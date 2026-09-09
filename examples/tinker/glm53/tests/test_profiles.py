"""CPU checks for profile composition and GPU-free configuration rendering."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.tinker.glm53 import run_server as module


def build(profile):
    return module.build_config(profile, Path("/models/model"), Path("/state"), Path("/scratch/traces"))


def test_small_model_does_not_inherit_glm_attention_or_expert_settings():
    cfg = build("qwen3-0.6b")
    assert "trainer.policy.megatron_config.transformer_config_kwargs" not in cfg
    assert "trainer.policy.megatron_config.expert_model_parallel_size" not in cfg
    assert cfg["trainer.policy.torch_profiler_config"]["ranks"] == [0]
    assert cfg["trainer.placement.policy_num_gpus_per_node"] == 1
    assert cfg["generator.inference_engine.tensor_parallel_size"] == 1
    assert not cfg["trainer.policy.megatron_config.lora_config.merge_lora"]
    assert cfg["generator.inference_engine.engine_init_kwargs"]["model"] == "/models/model"


@pytest.mark.parametrize(
    "profile,nodes,tp,cp,pp,context",
    [
        ("glm52-32k-2n", 1, 8, 1, 1, 32768),
        ("glm53-32k-2n", 1, 8, 1, 1, 32768),
        ("glm53-256k-2n", 1, 4, 2, 1, 262144),
        ("glm53-256k-3n", 2, 8, 1, 2, 262144),
    ],
)
def test_profiles_preserve_nested_common_settings(profile, nodes, tp, cp, pp, context):
    cfg = build(profile)
    assert cfg["trainer.placement.policy_num_nodes"] == nodes
    assert nodes * 8 == tp * cp * pp
    assert cfg["trainer.policy.torch_profiler_config"]["ranks"] == list(range(nodes * 8))
    for name, value in (("tensor_model", tp), ("context", cp), ("pipeline_model", pp)):
        assert cfg[f"trainer.policy.megatron_config.{name}_parallel_size"] == value
    engine = cfg["generator.inference_engine.engine_init_kwargs"]
    assert engine["max_model_len"] == context
    assert engine["model"] == "/models/model"
    assert engine["moe_backend"] == "triton"
    assert engine["kv_cache_dtype"] == ("bfloat16" if profile == "glm53-32k-2n" else "auto")
    assert cfg["generator.inference_engine.model_dtype"] == "bfloat16"
    assert cfg["generator.inference_engine.gpu_memory_utilization"] <= 0.8
    assert "lora_target_modules" not in engine
    assert "generator.inference_engine.max_num_seqs" not in cfg
    assert {"linear_kv_up_proj", "linear_fc1"} <= set(cfg["trainer.policy.model.lora.target_modules"])
    attention = cfg["trainer.policy.megatron_config.transformer_config_kwargs"]
    assert attention["dsa_kernel_backend"] == "tilelang"
    assert attention["recompute_num_layers"] == 1
    if pp == 2:
        assert attention["num_layers_in_first_pipeline_stage"] == 38
    else:
        assert "num_layers_in_first_pipeline_stage" not in attention


def test_glm53_startup_overrides_leave_all_other_runtime_knobs_unchanged():
    glm52, glm53 = build("glm52-32k-2n"), build("glm53-32k-2n")
    assert glm53.pop("generator.inference_engine.max_num_batched_tokens") == 8192
    assert glm52.pop("generator.inference_engine.max_num_batched_tokens") == 32768
    engine53, engine52 = (
        glm53["generator.inference_engine.engine_init_kwargs"],
        glm52["generator.inference_engine.engine_init_kwargs"],
    )
    assert not engine53.pop("enable_flashinfer_autotune")
    assert "enable_flashinfer_autotune" not in engine52
    assert engine53.pop("kv_cache_dtype") == "bfloat16"
    assert engine52.pop("kv_cache_dtype") == "auto"
    assert glm52 == glm53


def test_dry_run_needs_no_download_or_gpu_imports():
    result = subprocess.run(
        [
            sys.executable,
            str(Path(module.__file__)),
            "glm53-256k-3n",
            "--model-path",
            "/nonexistent/model",
            "--state-dir",
            "/nonexistent/state",
            "--database-path",
            "/nonexistent/local/tinker.db",
            "--profile-dir",
            "/nonexistent/local/traces",
            "--print-config",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["trainer.placement.policy_num_nodes"] == 2


def test_profile_overrides_do_not_leak_between_calls():
    first = build("glm53-256k-3n")
    first["generator.inference_engine.engine_init_kwargs"]["kv_cache_dtype"] = "fp8"
    assert build("glm53-32k-2n")["generator.inference_engine.engine_init_kwargs"]["kv_cache_dtype"] == "bfloat16"


def test_profiling_has_no_warmup_gap_or_one_update_cutoff():
    profiler = build("glm53-32k-2n")["trainer.policy.torch_profiler_config"]
    assert profiler["enable"] and profiler["profile_memory"]
    assert not profiler["collect_kernel_summary"]
    assert [profiler[key] for key in ("skip_first", "wait", "warmup", "active", "repeat")] == [0, 0, 0, 1, 0]
    with pytest.raises(ValueError, match="absolute path"):
        module.build_config("glm53-32k-2n", Path("/m"), Path("/s"), Path("relative/traces"))


def test_new_profile_is_discovered_without_inheriting_glm_settings(tmp_path, monkeypatch):
    qwen = json.loads((module.CONFIG_DIR / "qwen3-0.6b.json").read_text())
    (tmp_path / "new-model.json").write_text(json.dumps(qwen))
    (tmp_path / "common.json").write_text(json.dumps({"glm_only": True}))
    monkeypatch.setattr(module, "CONFIG_DIR", tmp_path)
    assert module.get_profiles() == ["new-model"]
    assert "glm_only" not in build("new-model")
    qwen["extends"] = "common"
    (tmp_path / "new-model.json").write_text(json.dumps(qwen))
    inherited = build("new-model")
    assert inherited["glm_only"] is True
    assert "extends" not in inherited
