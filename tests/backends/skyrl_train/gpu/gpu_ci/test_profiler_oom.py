"""Exercise CUDA allocator OOM export without exhausting physical GPU memory."""

import json

import pytest
import torch

from skyrl.backends.skyrl_train.utils.profiler import Profiler, flush_profile_on_oom
from skyrl.train.config import TorchProfilerConfig


def test_cuda_oom_exports_trace_and_keeps_profiling(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.cuda.current_device()
    original_fraction = torch.cuda.get_per_process_memory_fraction(device)
    limit = 64 * 1024**2
    total = torch.cuda.get_device_properties(device).total_memory
    profiler = Profiler(
        TorchProfilerConfig(
            enable=True,
            ranks=[0],
            save_path=str(tmp_path),
            activities=["cpu", "cuda"],
            warmup=0,
            active=1,
            repeat=0,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
    )

    class Worker:
        @flush_profile_on_oom
        def run(self, fail):
            with torch.profiler.record_function("before_oom" if fail else "after_oom"):
                value = torch.ones((512, 512), device=device)
                torch.mm(value, value)
                torch.cuda.synchronize(device)
            if fail:
                torch.empty(2 * limit, dtype=torch.uint8, device=device)

    worker = Worker()
    worker.profiler = profiler
    try:
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(limit / total, device)
        profiler.start()
        with pytest.raises(torch.OutOfMemoryError):
            worker.run(fail=True)
        torch.cuda.empty_cache()
        worker.run(fail=False)
        profiler.step()
    finally:
        profiler.stop()
        torch.cuda.set_per_process_memory_fraction(original_fraction, device)
        torch.cuda.empty_cache()

    traces = [json.loads(path.read_text()) for path in tmp_path.glob("*.pt.trace.json")]
    assert len(traces) >= 2
    for phase in ("before_oom", "after_oom"):
        matching = [trace for trace in traces if any(event["name"] == phase for event in trace["traceEvents"])]
        assert matching
        assert any(event.get("cat") == "kernel" for trace in matching for event in trace["traceEvents"])
