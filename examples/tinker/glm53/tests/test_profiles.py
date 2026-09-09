"""CPU-only profile merge and launcher dry-run checks."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("glm53_server_example", ROOT / "run_server.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestProfiles(unittest.TestCase):
    def test_small_model_does_not_inherit_glm_attention_or_expert_settings(self):
        cfg = module.build_config("qwen3-0.6b", Path("/models/qwen"), Path("/state/qwen"), Path("/scratch/traces"))
        self.assertNotIn("trainer.policy.megatron_config.transformer_config_kwargs", cfg)
        self.assertNotIn("trainer.policy.megatron_config.expert_model_parallel_size", cfg)
        self.assertEqual(cfg["trainer.policy.torch_profiler_config"]["ranks"], [0])
        self.assertEqual(cfg["trainer.placement.policy_num_gpus_per_node"], 1)
        self.assertEqual(cfg["generator.inference_engine.tensor_parallel_size"], 1)
        self.assertEqual(cfg["generator.inference_engine.engine_init_kwargs"]["model"], "/models/qwen")

    def test_profiles_preserve_nested_common_settings(self):
        for name, nodes, tp, cp, pp, context in (
            ("glm52-32k-2n", 1, 8, 1, 1, 32768),
            ("glm53-32k-2n", 1, 8, 1, 1, 32768),
            ("glm53-256k-2n", 1, 4, 2, 1, 262144),
            ("glm53-256k-3n", 2, 8, 1, 2, 262144),
        ):
            with self.subTest(profile=name):
                cfg = module.build_config(name, Path("/models/glm"), Path("/state/service"), Path("/scratch/traces"))
                self.assertEqual(cfg["trainer.placement.policy_num_nodes"], nodes)
                self.assertEqual(nodes * 8, tp * cp * pp)
                self.assertEqual(cfg["trainer.policy.torch_profiler_config"]["ranks"], list(range(nodes * 8)))
                self.assertEqual(cfg["trainer.policy.megatron_config.tensor_model_parallel_size"], tp)
                self.assertEqual(cfg["trainer.policy.megatron_config.context_parallel_size"], cp)
                self.assertEqual(cfg["trainer.policy.megatron_config.pipeline_model_parallel_size"], pp)
                engine = cfg["generator.inference_engine.engine_init_kwargs"]
                self.assertEqual(engine["max_model_len"], context)
                self.assertEqual(engine["model"], "/models/glm")
                self.assertEqual(engine["moe_backend"], "triton")
                self.assertEqual(engine["kv_cache_dtype"], "auto")
                self.assertEqual(cfg["generator.inference_engine.model_dtype"], "bfloat16")
                self.assertLessEqual(cfg["generator.inference_engine.gpu_memory_utilization"], 0.8)
                self.assertNotIn("lora_target_modules", engine)
                self.assertNotIn("generator.inference_engine.max_num_seqs", cfg)
                self.assertIn("linear_kv_up_proj", cfg["trainer.policy.model.lora.target_modules"])
                self.assertIn("linear_fc1", cfg["trainer.policy.model.lora.target_modules"])
                attention = cfg["trainer.policy.megatron_config.transformer_config_kwargs"]
                self.assertEqual(attention["dsa_kernel_backend"], "tilelang")
                self.assertEqual(attention["recompute_num_layers"], 1)
                if pp == 2:
                    self.assertEqual(attention["num_layers_in_first_pipeline_stage"], 38)
                else:
                    self.assertNotIn("num_layers_in_first_pipeline_stage", attention)

    def test_glm52_and_glm53_32k_profiles_have_identical_runtime_knobs(self):
        glm52 = module.build_config("glm52-32k-2n", Path("/models/glm52"), Path("/state"), Path("/traces"))
        glm53 = module.build_config("glm53-32k-2n", Path("/models/glm53"), Path("/state"), Path("/traces"))
        for cfg in (glm52, glm53):
            cfg.pop("trainer.policy.model.path")
            cfg["generator.inference_engine.engine_init_kwargs"].pop("model")
        self.assertEqual(glm52, glm53)

    def test_dry_run_needs_no_download_or_gpu_imports(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "run_server.py"),
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
        self.assertEqual(json.loads(result.stdout)["trainer.placement.policy_num_nodes"], 2)

    def test_profile_overrides_do_not_leak_between_calls(self):
        first = module.build_config("glm53-256k-3n", Path("/m"), Path("/s"), Path("/scratch/traces"))
        first["generator.inference_engine.engine_init_kwargs"]["kv_cache_dtype"] = "fp8"
        second = module.build_config("glm53-32k-2n", Path("/m"), Path("/s"), Path("/scratch/traces"))
        self.assertEqual(second["generator.inference_engine.engine_init_kwargs"]["kv_cache_dtype"], "auto")

    def test_profiling_has_no_warmup_gap_or_one_update_cutoff(self):
        profiled = module.build_config("glm53-32k-2n", Path("/m"), Path("/s"), Path("/scratch/traces"))
        profiler = profiled["trainer.policy.torch_profiler_config"]
        self.assertTrue(profiler["enable"])
        self.assertEqual((profiler["skip_first"], profiler["wait"], profiler["warmup"]), (0, 0, 0))
        self.assertEqual((profiler["active"], profiler["repeat"]), (1, 0))
        with self.assertRaisesRegex(ValueError, "absolute path"):
            module.build_config("glm53-32k-2n", Path("/m"), Path("/s"), Path("relative/traces"))


if __name__ == "__main__":
    unittest.main()
