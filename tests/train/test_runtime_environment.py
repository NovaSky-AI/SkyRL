"""Check the runtime environment delivered by the training entrypoint."""

import os
import subprocess
import sys
import textwrap

import pytest
import ray


@pytest.mark.parametrize("strategy", ["fsdp", "megatron"])
def test_initialize_ray_delivers_kernel_cache_directories(tmp_path, strategy):
    caches = {
        "XDG_CACHE_HOME": str(tmp_path / "user cache"),
        "TILELANG_CACHE_DIR": str(tmp_path / "tilelang"),
        "TRITON_CACHE_DIR": str(tmp_path / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
    }
    # A separate driver can call initialize_ray without replacing the session
    # fixture's Ray connection. The raylet predates these driver-only variables.
    driver = textwrap.dedent(
        """
        import os
        import sys

        import ray

        from skyrl.train.utils.utils import initialize_ray
        from tests.train.util import example_dummy_config

        cfg = example_dummy_config()
        cfg.trainer.strategy = sys.argv[1]
        cfg.trainer.log_path = sys.argv[2]
        cfg.generator.inference_engine.tensor_parallel_size = 1
        cfg.trainer.placement.policy_num_gpus_per_node = 1
        cfg.trainer.placement.critic_num_gpus_per_node = 1
        cfg.trainer.placement.ref_num_gpus_per_node = 1

        # Defined here so deserializing the actor does not import training modules.
        @ray.remote(num_cpus=1, num_gpus=0)
        class CacheEnvironment:
            def __init__(self, names):
                self.values = {name: os.environ.get(name) for name in names}
                self.imported = [
                    name for name in (
                        "fla", "tilelang", "triton", "torch._inductor",
                        "transformers.models.qwen3_5",
                    ) if name in sys.modules
                ]

            def read(self):
                return self.values, self.imported

        names = sys.argv[3:]
        try:
            initialize_ray(cfg)
            actor = CacheEnvironment.remote(names)
            values, imported = ray.get(actor.read.remote(), timeout=60)
            assert not imported, imported
            assert values == {name: os.environ[name] for name in names}, values
        finally:
            ray.shutdown()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", driver, strategy, str(tmp_path / "logs"), *caches],
        env={
            **os.environ,
            **caches,
            "RAY_ADDRESS": ray.get_runtime_context().gcs_address,
            "CUDA_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
