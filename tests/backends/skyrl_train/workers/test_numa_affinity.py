"""NUMA affinity of trainer workers: CPUs pinned to the GPU's node, memory only preferred there.

A strict membind OOM-kills on one node while the other is free (GB200: two ~478 GiB CPU nodes;
the colocated vLLM's pinned sleep backup plus the trainer's host offload can fill one node).
"""

from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.workers import worker as worker_mod


class _FakeNuma:
    def __init__(self):
        self.calls = []
        for name in (
            "numa_parse_nodestring",
            "numa_run_on_node_mask",
            "numa_set_membind",
            "numa_set_preferred",
            "numa_num_configured_nodes",
        ):
            setattr(self, name, _Fn(self, name))


class _Fn:
    def __init__(self, owner, name):
        self.owner, self.name, self.argtypes, self.restype = owner, name, None, None

    def __call__(self, *a):
        self.owner.calls.append((self.name, a))
        return 0


@pytest.mark.parametrize("local_rank,node", [(0, 0), (3, 0), (4, 1), (7, 1)], ids=["r0", "r3", "r4", "r7"])
def test_memory_is_preferred_not_bound(monkeypatch, local_rank, node):
    fake = _FakeNuma()
    monkeypatch.setattr(worker_mod, "CDLL", lambda _path: fake)
    monkeypatch.setattr(worker_mod, "_SET_AFFINITY", False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setattr(
        worker_mod.torch.cuda, "get_device_properties", lambda _i: (_ for _ in ()).throw(RuntimeError())
    )
    real_listdir, real_open = worker_mod.os.listdir, open

    def listdir(p):
        return ["node0", "node1", "node2"] if p == "/sys/devices/system/node" else real_listdir(p)

    def fake_open(p, *a, **k):
        if str(p).startswith("/sys/devices/system/node/node"):
            import io

            return io.StringIO("" if "node2" in str(p) else "0-71")
        return real_open(p, *a, **k)

    monkeypatch.setattr(worker_mod.os, "listdir", listdir)
    monkeypatch.setattr("builtins.open", fake_open)
    worker_mod.DistributedTorchRayActor._set_numa_affinity(SimpleNamespace(_local_rank=local_rank), local_rank)

    names = [c[0] for c in fake.calls]
    assert "numa_set_membind" not in names
    assert ("numa_set_preferred", (node,)) in fake.calls
    assert "numa_run_on_node_mask" in names
