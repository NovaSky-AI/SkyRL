import io
import os
from typing import Optional

import ray
import torch
import torch.distributed
from transformers import AutoConfig

from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
    should_use_meta_init,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    SKYRL_LORA_ADAPTER_NAME,
)
from skyrl.backends.skyrl_train.training_batch import (
    TrainingInputBatch,
)
from skyrl.backends.skyrl_train.utils.profiler import build_profiler_from_policy_cfg
from skyrl.backends.skyrl_train.weight_sync import LoraLoadRequest
from skyrl.backends.skyrl_train.workers.model_wrapper import (
    HFModelWrapper,
    get_llm_for_sequence_regression,
)
from skyrl.backends.skyrl_train.workers.worker import (
    CriticWorkerBase,
    PolicyWorkerBase,
    RefWorkerBase,
)


class FSDPPolicyWorkerBase(PolicyWorkerBase):
    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.strategy == "fsdp"
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.policy.fsdp_config,
            # Inference-only workers skip the optimizer entirely: passing None makes
            # FSDPStrategy.prepare return (model, None, None), avoiding the fp32 master
            # weights + AdamW state that would OOM memory-constrained nodes.
            optimizer_config=None if self.cfg.policy.inference_only_init else self.cfg.policy.optimizer_config,
            model_config=self.cfg.policy.model,
            fsdp_strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            micro_train_batch_size_per_gpu=self.cfg.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        self._is_lora = self.cfg.policy.model.lora.rank > 0

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        is_multimodal = hasattr(model_config, "vision_config") and model_config.vision_config is not None
        self._is_multimodal_lm_only = self.cfg.policy.language_model_only and is_multimodal
        use_meta = should_use_meta_init(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )

        wrapped_model = HFModelWrapper(
            model_path,
            use_flash_attention_2=self.cfg.flash_attn,
            bf16=self.cfg.policy.inference_only_init,
            lora_rank=self.cfg.policy.model.lora.rank,
            lora_alpha=self.cfg.policy.model.lora.alpha,
            lora_dropout=self.cfg.policy.model.lora.dropout,
            lora_init_method=self.cfg.policy.model.lora.init_method,
            target_modules=self.cfg.policy.model.lora.target_modules,
            exclude_modules=self.cfg.policy.model.lora.exclude_modules,
            sequence_parallel_size=self.cfg.policy.sequence_parallel_size,
            remove_microbatch_padding=self.cfg.remove_microbatch_padding,
            use_torch_compile=self.cfg.policy.use_torch_compile,
            model_config_kwargs=self.cfg.policy.model_config_kwargs,
            meta_init=use_meta,
            language_model_only=self.cfg.policy.language_model_only,
            logprobs_chunk_size=self.cfg.logprobs_chunk_size,
        )
        self._seq_parallel_monkey_patch(model=wrapped_model.model)

        if self.cfg.gradient_checkpointing:
            wrapped_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": self.cfg.gradient_checkpointing_use_reentrant}
            )

        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (wrapped_model, None, None),
        )
        if self.cfg.policy.inference_only_init:
            assert (
                self.optimizer is None and self.scheduler is None
            ), "inference_only_init should skip optimizer and scheduler construction"
        else:
            assert (
                self.optimizer is not None and self.scheduler is not None
            ), "FSDP preparation should create optimizer and scheduler"

        # Enable expandable_segments after init so model weights stay in IPC-compatible
        # standard CUDA memory; only subsequent activations use expandable segments.
        self._set_expandable_segments(True)

        # Created only on profiled ranks.
        self.profiler = build_profiler_from_policy_cfg(self.cfg)

    def _build_weight_source(self, dtype: "torch.dtype", backend: str):
        """``WeightSource`` over the FSDP-sharded policy model.

        ``self.model.model`` is the inner HF module, whose ``state_dict()`` keys
        are the names vLLM expects; the outer wrapper's are prefixed differently.
        ``weight_prefix`` covers the one case where they still differ: syncing a
        CausalLM backbone into a vLLM multimodal namespace.
        """
        weight_prefix = "language_model." if self._is_multimodal_lm_only else ""
        if backend == "sharded_rdt":
            # RDT pulls, so it needs the ownership + group channels its own
            # source subclass adds (see sharded_rdt/sharded_rdt_base.py).
            from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
                make_fsdp_weight_source,
            )

            return make_fsdp_weight_source(self.model.model, dtype, weight_prefix)

        from skyrl.backends.skyrl_train.weight_sync.sources import FsdpWeightSource

        return FsdpWeightSource(self.model.model, dtype, weight_prefix)

    async def _save_lora_adapters_and_sync(
        self,
        peft_model,
        lora_sync_path,
        inference_engine_client,
        lora_name: str = SKYRL_LORA_ADAPTER_NAME,
    ):
        """Collect LoRA parameters, save and call inference engine to load."""
        import json
        from dataclasses import asdict

        from safetensors.torch import save_file

        from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
            collect_lora_params,
        )

        lora_params = collect_lora_params(module=self.model.model)

        if torch.distributed.get_rank() == 0:
            os.makedirs(lora_sync_path, exist_ok=True)

            peft_config = asdict(peft_model.peft_config.get("default", {}))
            peft_config["task_type"] = peft_config["task_type"].value
            peft_config["peft_type"] = peft_config["peft_type"].value
            peft_config["target_modules"] = list(peft_config["target_modules"])

            # Save LoRA parameters and config
            save_file(lora_params, os.path.join(lora_sync_path, "adapter_model.safetensors"))
            with io.open(os.path.join(lora_sync_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(peft_config, f, ensure_ascii=False, indent=4)

            # Send LoRA disk loading request to inference engine.
            from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
                RemoteInferenceClient,
            )

            if isinstance(inference_engine_client, RemoteInferenceClient):
                await inference_engine_client.load_lora_adapter(lora_name, lora_sync_path)
            else:
                lora_request = LoraLoadRequest(lora_path=lora_sync_path, lora_name=lora_name)
                await inference_engine_client.update_named_weights(lora_request)

        torch.distributed.barrier()

    async def broadcast_to_inference_engines(
        self,
        inference_engine_client,
        inference_engine_cfg,
        model_id: Optional[str] = None,
    ):
        if inference_engine_client is None:
            inference_engine_client = self._weight_sync_inference_client

        # Check if this is a LoRA model
        peft_model = getattr(self.model.model, "_fsdp_wrapped_module", self.model.model)

        if self._is_lora:
            assert hasattr(peft_model, "peft_config"), "LoRA model should have peft_config"

            # Multi-tenant: per-adapter subdir + per-adapter vLLM name.
            # Single-tenant (model_id=None) keeps the legacy shared path +
            # name. _resolve_lora_sync_target (shared with Megatron, defined on
            # PolicyWorkerBase) basename-guards against a malformed model_id
            # escaping lora_sync_path even though api.py already validates IDs.
            cache_reset_task = self._reset_prefix_cache_task(inference_engine_client, inference_engine_cfg)
            torch.cuda.empty_cache()
            lora_name, lora_sync_path = self._resolve_lora_sync_target(model_id)
            await self._save_lora_adapters_and_sync(
                peft_model, lora_sync_path, inference_engine_client, lora_name=lora_name
            )
            if cache_reset_task is not None:
                await cache_reset_task
            if self.cfg.placement.colocate_all:
                torch.cuda.empty_cache()
            torch.distributed.barrier()
            return

        await self._sync_weights_to_inference_engines(inference_engine_client, inference_engine_cfg)

    def _set_pad_token_id(self, pad_token_id):
        # NOTE (sumanthrh): self.model -> HFModelWrapper; self.model.model -> AutoModelForCausalLM
        self.model.model.config.pad_token_id = pad_token_id

    def forward(
        self,
        data: TrainingInputBatch,
        loss_fn=None,
        loss_fn_config=None,
        return_per_token_outputs: bool = True,
    ) -> WorkerOutput:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(
            data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            return_per_token_outputs=return_per_token_outputs,
        )
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        return output


class FSDPCriticWorkerBase(CriticWorkerBase):
    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.strategy == "fsdp"
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.critic.fsdp_config,
            optimizer_config=self.cfg.critic.optimizer_config,
            fsdp_strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            micro_train_batch_size_per_gpu=self.cfg.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        use_meta = should_use_meta_init(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )

        critic = get_llm_for_sequence_regression(
            model_path,
            "critic",
            use_flash_attention_2=self.cfg.flash_attn,
            bf16=False,
            lora_rank=self.cfg.critic.model.lora.rank,
            lora_alpha=self.cfg.critic.model.lora.alpha,
            lora_dropout=self.cfg.critic.model.lora.dropout,
            target_modules=self.cfg.critic.model.lora.target_modules,
            exclude_modules=self.cfg.critic.model.lora.exclude_modules,
            value_head_prefix=self.cfg.algorithm.value_head_prefix,
            init_value_head=self.cfg.policy.model.path == self.cfg.critic.model.path,
            sequence_parallel_size=self.cfg.critic.sequence_parallel_size,
            remove_microbatch_padding=self.cfg.remove_microbatch_padding,
            model_config_kwargs=self.cfg.critic.model_config_kwargs,
            meta_init=use_meta,
        )
        self._seq_parallel_monkey_patch(model=critic, use_parent_class=True)

        if self.cfg.gradient_checkpointing:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": self.cfg.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (critic, None, None),
        )
        assert self.optimizer is not None

        self._set_expandable_segments(True)

    def _set_pad_token_id(self, pad_token_id):
        self.model.config.pad_token_id = pad_token_id

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> WorkerOutput:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        return output


class FSDPRefWorkerBase(RefWorkerBase):
    def init_model(self, model_path):
        assert self.cfg.strategy == "fsdp"
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.ref.fsdp_config,
            fsdp_strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            micro_train_batch_size_per_gpu=self.cfg.micro_train_batch_size_per_gpu,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        use_meta = should_use_meta_init(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )

        wrapped_model = HFModelWrapper(
            model_path,
            use_flash_attention_2=self.cfg.flash_attn,
            bf16=self.cfg.bf16,
            sequence_parallel_size=self.cfg.ref.sequence_parallel_size,
            remove_microbatch_padding=self.cfg.remove_microbatch_padding,
            model_config_kwargs=self.cfg.ref.model_config_kwargs,
            meta_init=use_meta,
            language_model_only=self.cfg.ref.language_model_only,
            logprobs_chunk_size=self.cfg.logprobs_chunk_size,
        )
        self._seq_parallel_monkey_patch(model=wrapped_model.model)

        self.model = strategy.prepare(wrapped_model)
        self.model.eval()

        self._set_expandable_segments(True)

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> WorkerOutput:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        return output


# Ray remote actors
PolicyWorker = ray.remote(num_gpus=1)(FSDPPolicyWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(FSDPCriticWorkerBase)
RefWorker = ray.remote(num_gpus=1)(FSDPRefWorkerBase)
