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
    def test_profiles_preserve_nested_common_settings(self):
        for name, nodes, tp, cp, pp, context in (
            ("32k-2n", 1, 8, 1, 1, 32768),
            ("256k-2n", 1, 4, 2, 1, 262144),
            ("256k-3n", 2, 8, 1, 2, 262144),
        ):
            with self.subTest(profile=name):
                cfg = module.build_config(name, Path("/models/glm"), Path("/state/service"))
                self.assertEqual(cfg["trainer.placement.policy_num_nodes"], nodes)
                self.assertEqual(nodes * 8, tp * cp * pp)
                self.assertEqual(cfg["trainer.policy.megatron_config.tensor_model_parallel_size"], tp)
                self.assertEqual(cfg["trainer.policy.megatron_config.context_parallel_size"], cp)
                self.assertEqual(cfg["trainer.policy.megatron_config.pipeline_model_parallel_size"], pp)
                engine = cfg["generator.inference_engine.engine_init_kwargs"]
                self.assertEqual(engine["max_model_len"], context)
                self.assertEqual(engine["model"], "/models/glm")
                self.assertEqual(engine["moe_backend"], "triton")
                self.assertEqual(engine["kv_cache_dtype"], "auto")
                self.assertEqual(cfg["generator.inference_engine.model_dtype"], "bfloat16")
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

    def test_dry_run_needs_no_download_or_gpu_imports(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "run_server.py"),
                "256k-3n",
                "--model-path",
                "/nonexistent/model",
                "--state-dir",
                "/nonexistent/state",
                "--database-path",
                "/nonexistent/local/tinker.db",
                "--print-config",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["trainer.placement.policy_num_nodes"], 2)

    def test_profile_overrides_do_not_leak_between_calls(self):
        first = module.build_config("256k-3n", Path("/m"), Path("/s"))
        first["generator.inference_engine.engine_init_kwargs"]["kv_cache_dtype"] = "fp8"
        second = module.build_config("32k-2n", Path("/m"), Path("/s"))
        self.assertEqual(second["generator.inference_engine.engine_init_kwargs"]["kv_cache_dtype"], "auto")

    def test_profile_window_is_opt_in_and_does_not_change_model_recipe(self):
        control = module.build_config("32k-2n", Path("/m"), Path("/s"))
        profiled = module.build_config("32k-2n", Path("/m"), Path("/s"), Path("/scratch/traces"))
        profiler = profiled.pop("trainer.policy.torch_profiler_config")
        self.assertEqual(profiled, control)
        self.assertEqual((profiler["warmup"], profiler["active"], profiler["repeat"]), (1, 1, 1))
        with self.assertRaisesRegex(ValueError, "absolute path"):
            module.build_config("32k-2n", Path("/m"), Path("/s"), Path("relative/traces"))


if __name__ == "__main__":
    unittest.main()
