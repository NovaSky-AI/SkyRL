"""examples/ml4: the recipe yaml parses into a valid SkyRL config with the recipe's values."""

import os

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YAML = os.path.join(REPO, "examples", "ml4", "ml4_dapo_lora_fsdp.yaml")


@pytest.fixture(scope="module")
def cfg():
    from skyrl.train.config import SkyRLTrainConfig

    with open(YAML) as f:
        overrides = yaml.safe_load(f)
    overrides |= {
        "trainer.policy.model.path": "/models/ml4",
        "data.train_data": ["/data/dapo-math-17k-cleaned.parquet"],
        "data.val_data": ["/data/aime-2024-cleaned.parquet"],
    }
    return SkyRLTrainConfig.from_cli_overrides(overrides)


def test_recipe_values(cfg):
    t, g, a = cfg.trainer, cfg.generator, cfg.trainer.algorithm
    lora = t.policy.model.lora
    assert (lora.rank, lora.alpha, lora.sync_mode) == (32, 64, "memory")
    # shared-outer on the trainer, expanded per expert on the wire: vLLM's default (per-expert) format
    assert lora.experts_shared_outer_loras is True
    assert "enable_moe_shared_loras" not in g.inference_engine.engine_init_kwargs
    assert t.policy.model.sharded_load_fn == "loading.load_sharded"
    assert t.policy.model_config_kwargs == {
        "attn_implementation": "eager",
        "moe_grouped_mm": True,
        "logprob_chunk_size": 2048,
        "fp8_kv_cache": True,
        "rollout_moe_activation": "nvfp4_per_token",
    }
    assert (t.old_logprobs_from_update_forward, t.micro_train_batch_size_per_gpu) == (True, 2)
    assert (t.remove_microbatch_padding, t.flash_attn) == (False, False)
    assert (a.off_policy_correction.tis_ratio_type, a.off_policy_correction.token_tis_ratio_clip_high) == ("token", 2.0)
    assert (a.loss_reduction, a.eps_clip_low, a.eps_clip_high, a.use_kl_loss) == ("sequence_mean", 0.2, 0.28, False)
    opt = t.policy.optimizer_config
    assert (opt.lr, list(opt.adam_betas), opt.weight_decay, opt.num_warmup_steps) == (1e-5, [0.9, 0.98], 0.1, 0)
    assert (t.train_batch_size, t.policy_mini_batch_size, g.n_samples_per_prompt) == (32, 32, 8)
    assert (t.max_prompt_length, g.sampling_params.max_generate_length, g.sampling_params.logprobs) == (2048, 4096, 1)
    ev = g.eval_sampling_params
    assert (g.eval_n_samples_per_prompt, ev.temperature, ev.top_p, ev.max_generate_length) == (8, 1.0, 1.0, 4096)
    assert (t.eval_before_train, t.eval_interval, t.ckpt_interval, t.max_ckpts_to_keep) == (True, 10, 5, 2)
    ie = g.inference_engine
    assert (ie.num_engines, ie.tensor_parallel_size, ie.enforce_eager) == (4, 8, False)
    assert ie.engine_init_kwargs["limit_mm_per_prompt"] == {"image": 0}
    # trainer numerics match what the rollout serves: FP8 latent KV, per-token NVFP4 MoE activation
    assert ie.engine_init_kwargs["kv_cache_dtype"] == "fp8_ds_mla"
    assert t.policy.model_config_kwargs["fp8_kv_cache"] is True
    act = ie.engine_init_kwargs["quantization_config"]["moe"]["activation"]
    assert act == t.policy.model_config_kwargs["rollout_moe_activation"] == "nvfp4_per_token"


def test_recipe_constructs_the_model(cfg, tmp_path):
    """The recipe's model_config_kwargs on a checkpoint shaped like the real one
    (dsa.fp8_kv_cache=true, kv_lora_rank 512): AutoConfig + meta construction succeeds with
    the FP8 latent-KV simulation and NVFP4 QAT on; without the recipe's kwargs ml4_train
    refuses the config (no rollout MoE activation)."""
    from conftest import ML4_TRAIN_PATH

    if not ML4_TRAIN_PATH:
        pytest.skip("ML4_TRAIN_PATH not set")
    import json

    import torch
    from fake_nvfp4 import nvfp4_params, write_fake_nvfp4_ckpt
    from ml4_train.convert import convert
    from transformers import AutoConfig, AutoModelForCausalLM

    p = nvfp4_params()
    p["dsa"]["fp8_kv_cache"] = True
    p["kv_lora_rank"] = 512  # fp8_ds_mla layout
    ckpt, model_dir = str(tmp_path / "ckpt"), str(tmp_path / "model")
    write_fake_nvfp4_ckpt(ckpt, p)
    convert(ckpt, model_dir)
    cfg_path = os.path.join(model_dir, "config.json")
    c = json.load(open(cfg_path))
    assert c["fp8_kv_cache"] is True
    c.update(rollout_tp=2, rollout_moe_tp=2)  # set per run on the cluster model dir
    json.dump(c, open(cfg_path, "w"))

    def build(**kwargs):
        mc = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, **kwargs)
        with torch.device("meta"):
            return AutoModelForCausalLM.from_config(mc, trust_remote_code=True)

    m = build(**cfg.trainer.policy.model_config_kwargs)
    assert m.config.fp8_kv_cache is True
    assert all(mod.fp8_kv for mod in m.modules() if type(mod).__name__ == "ML4Attention")
    assert any(type(mod).__name__ == "NVFP4Linear" for mod in m.modules())
    with pytest.raises(ValueError, match="rollout_moe_activation"):
        build(attn_implementation="eager")


def test_lora_targets_are_the_ml4_contract(cfg):
    pytest.importorskip("ml4_train")
    from ml4_train.lora import LORA_TARGET_REGEX

    assert cfg.trainer.policy.model.lora.target_modules == LORA_TARGET_REGEX


def test_validate_inference_engine_cfg_accepts(cfg):
    from skyrl.train.utils.utils import validate_inference_engine_cfg

    validate_inference_engine_cfg(cfg)


def test_old_logprobs_from_update_forward_needs_one_update_per_batch(cfg):
    import copy

    from skyrl.train.utils.utils import validate_old_logprobs_from_update_forward

    validate_old_logprobs_from_update_forward(cfg)
    for key, value in [("policy_mini_batch_size", 16), ("update_epochs_per_batch", 2), ("strategy", "megatron")]:
        bad = copy.deepcopy(cfg)
        setattr(bad.trainer, key, value)
        with pytest.raises(ValueError, match="old_logprobs_from_update_forward"):
            validate_old_logprobs_from_update_forward(bad)


@pytest.mark.parametrize(
    "trainer_fp8,engine_kw,ok",
    [
        (True, {"kv_cache_dtype": "fp8_ds_mla"}, True),
        (False, {"kv_cache_dtype": "fp8_ds_mla"}, False),
        (True, {"kv_cache_dtype": "auto"}, False),
        (True, {"kv_cache_dtype": None}, False),
        (False, {"kv_cache_dtype": "bfloat16"}, True),
    ],
    ids=["recipe", "trainer-bf16", "engine-auto", "engine-unset", "both-bf16"],
)
def test_kv_cache_dtype_startup_check(cfg, trainer_fp8, engine_kw, ok):
    """validate_cfg aborts when the trainer's fp8_kv_cache and the engines' kv_cache_dtype disagree."""
    from conftest import ML4_TRAIN_PATH

    if not ML4_TRAIN_PATH:
        pytest.skip("ML4_TRAIN_PATH not set")
    import copy

    from skyrl.train.utils.utils import validate_ml4_kv_cache_dtype

    c = copy.deepcopy(cfg)
    c.trainer.policy.model_config_kwargs["fp8_kv_cache"] = trainer_fp8
    kw = dict(c.generator.inference_engine.engine_init_kwargs)
    kw.update(engine_kw)
    if engine_kw["kv_cache_dtype"] is None:
        kw.pop("kv_cache_dtype")
    c.generator.inference_engine.engine_init_kwargs = kw
    if ok:
        validate_ml4_kv_cache_dtype(c)
    else:
        with pytest.raises(ValueError, match="kv_cache_dtype"):
            validate_ml4_kv_cache_dtype(c)
