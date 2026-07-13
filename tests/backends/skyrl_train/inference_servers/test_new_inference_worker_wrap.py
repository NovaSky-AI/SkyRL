from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    NewInferenceWorkerWrap,
)


def test_worker_cuda_memory_stats_reports_device_wide_usage(monkeypatch):
    import skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap as worker_module

    monkeypatch.setattr(worker_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(worker_module.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(worker_module.torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(worker_module.torch.cuda, "mem_get_info", lambda _device: (300, 1000))
    monkeypatch.setattr(worker_module.torch.cuda, "memory_allocated", lambda _device: 100)
    monkeypatch.setattr(worker_module.torch.cuda, "memory_reserved", lambda _device: 200)

    worker = object.__new__(NewInferenceWorkerWrap)

    assert worker.skyrl_cuda_memory_stats() == {
        "cuda_available": True,
        "device": 2,
        "allocated_bytes": 100,
        "reserved_bytes": 200,
        "free_bytes": 300,
        "total_bytes": 1000,
        "device_used_bytes": 700,
    }


def test_worker_release_cuda_memory_clears_allocator_before_measuring(monkeypatch):
    import skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap as worker_module

    events = []
    monkeypatch.setattr(worker_module.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(worker_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(worker_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(worker_module.torch.cuda, "synchronize", lambda _device: events.append("sync"))
    monkeypatch.setattr(worker_module.torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(worker_module.torch.cuda, "ipc_collect", lambda: events.append("ipc_collect"))

    worker = object.__new__(NewInferenceWorkerWrap)
    monkeypatch.setattr(worker, "skyrl_cuda_memory_stats", lambda: {"device_used_bytes": 10})

    assert worker.skyrl_release_cuda_memory() == {"device_used_bytes": 10}
    assert events == ["gc", "sync", "empty_cache", "ipc_collect", "sync"]
