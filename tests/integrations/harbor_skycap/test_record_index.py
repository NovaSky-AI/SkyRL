"""The per-step W&B index of skycap records: what each version holds, and that logging fails open.

W&B is a fake module with the surface the callback uses (``Artifact``, ``run.log_artifact``); nothing
reaches the network. The records come from real trials through a real skycap server.
"""

import json
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("skycap")
pytest.importorskip("harbor")

import fsspec  # noqa: E402
import orjson  # noqa: E402
import pytest_asyncio  # noqa: E402
import torch  # noqa: E402
import zstandard  # noqa: E402
from loguru import logger  # noqa: E402

from examples.train_integrations.harbor_skycap import record_index  # noqa: E402
from examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap import (  # noqa: E402
    HarborSkycapConfig,
)
from examples.train_integrations.harbor_skycap.harbor_generator import (  # noqa: E402
    HarborSkycapGenerator,
)
from examples.train_integrations.harbor_skycap.record_index import (  # noqa: E402
    RecordLog,
    SkycapRecordIndex,
)
from skyrl.train.utils.callbacks import CallbackInput  # noqa: E402
from tests.integrations.harbor_skycap import (
    test_harbor_skycap as harbor_tests,  # noqa: E402
)
from tests.integrations.harbor_skycap.test_harbor_skycap import (  # noqa: E402
    batch,
    generator_cfg,
    harbor_cfg,
    service_for,
)

# The end-to-end tests' fixtures: a mock router, a skycap server in front of it, and fake Harbor trials.
router = harbor_tests.router
skycap = harbor_tests.skycap
trials = harbor_tests.trials

pytestmark = pytest.mark.integrations


class FakeArtifact:
    def __init__(self, name, type, metadata=None) -> None:
        self.name, self.type, self.metadata = name, type, dict(metadata or {})
        self.files = {}
        self.references = []

    def add_file(self, local_path, name) -> None:
        self.files[name] = Path(local_path).read_bytes()

    def add_reference(self, uri, name=None, checksum=True) -> None:
        self.references.append((uri, name, checksum))

    def index(self) -> list:
        return json.loads(self.files["step.json"])


class FakeRun:
    """``log_artifact`` raises ``failures`` errors first, and waits on ``hang`` when given one."""

    def __init__(self, failures=0, error=ConnectionError, hang=None, wait_error=None) -> None:
        self.id = "run/1"
        self.logged = []
        self.calls = 0
        self.failures, self.error, self.hang, self.wait_error = failures, error, hang, wait_error

    def log_artifact(self, artifact, aliases):
        self.calls += 1
        if self.hang is not None:
            self.hang.wait(30)
        if self.failures:
            self.failures -= 1
            raise self.error("W&B unreachable")
        self.logged.append((artifact, aliases))

        def wait(timeout):
            if self.wait_error is not None:
                raise self.wait_error("still uploading")
            return artifact

        return SimpleNamespace(wait=wait)


def wandb_trainer(run=None, backend="wandb"):
    run = run or FakeRun()
    wandb = SimpleNamespace(Artifact=FakeArtifact, run=run, errors=SimpleNamespace())
    return SimpleNamespace(tracker=SimpleNamespace(backend=backend, logger=wandb)), run


def step_end(step=3, trajectory_ids=None, loss_masks=None) -> CallbackInput:
    """``on_step_end``'s input: one loss-mask row per trajectory id, plus a padding row as the trainer adds."""
    batch = None
    if loss_masks is not None:
        rows = [[int(any(mask))] for mask in loss_masks] + [[1]]
        batch = {"loss_mask": torch.tensor(rows)}
    return CallbackInput(
        global_step=step, epoch=0, total_steps=9, steps_per_epoch=9, batch=batch, trajectory_ids=trajectory_ids
    )


def index(records, *, phases=("train",), **options) -> SkycapRecordIndex:
    options.setdefault("backoff", 0.0)
    return SkycapRecordIndex(records, list(phases), **options)


@pytest.fixture
def warnings():
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    yield messages
    logger.remove(sink)


@pytest_asyncio.fixture
async def make_generator(skycap):
    made = []

    def make(records, service=None) -> HarborSkycapGenerator:
        url = (service or skycap).url
        made.append(
            HarborSkycapGenerator(generator_cfg(), harbor_cfg(), [url], SimpleNamespace(weight_version=7), records)
        )
        return made[-1]

    yield make
    for gen in made:
        await gen.close()


@pytest.mark.asyncio
async def test_a_step_is_one_version_however_many_generate_calls_it_took(skycap, trials, make_generator) -> None:
    records = RecordLog()
    gen = make_generator(records)
    # Dynamic sampling generates more than once per step; the index covers all of it.
    first = await gen.generate(batch("linear"), disable_tqdm=True)
    second = await gen.generate(batch("crash"), disable_tqdm=True)
    ids = first["trajectory_ids"] + second["trajectory_ids"]
    masks = first["loss_masks"] + second["loss_masks"]
    trainer, run = wandb_trainer()
    callback = index(records)
    callback.on_step_end(trainer, step_end(trajectory_ids=ids, loss_masks=masks), None)
    assert callback.close()

    ((artifact, aliases),) = run.logged
    assert artifact.name == "skycap-records-train-run-1" and artifact.type == "skycap-records"
    assert aliases == ["train-step-3", "latest"]
    rows = {(row["instance_id"], row["attempt"]): row for row in artifact.index()}
    # "crash" failed twice: the first attempt was superseded by the second, which was masked.
    assert {key: (row["superseded"], row["trained"]) for key, row in rows.items()} == {
        ("linear", 0): (False, True),
        ("crash", 0): (True, False),
        ("crash", 1): (False, False),
    }
    linear = rows[("linear", 0)]
    record_dir = skycap.server.record_dir
    assert linear["phase"] == "train" and linear["status"] == "finished" and linear["repetition_id"] == 0
    assert linear["annotations"] == {"reward": 1.0, "stop_reason": "complete"}
    assert linear["record"] == {"path": str(record_dir / f"{linear['id']}.json.zst"), "mirror": None}
    # Local-only records are indexed, never uploaded.
    assert artifact.references == [] and set(artifact.files) == {"step.json"}
    assert artifact.metadata == {
        "global_step": 3,
        "training_phase": "train",
        "run_id": "run/1",
        "num_trajectories": 3,
        "num_trained": 1,
        "num_superseded": 1,
        "num_recorded": 3,
        "num_referenced": 0,
    }
    assert records.take("train") == []


@pytest.mark.asyncio
async def test_a_trajectory_dropped_by_dynamic_sampling_is_untrained(skycap, trials, make_generator) -> None:
    records = RecordLog()
    out = await make_generator(records).generate(batch("linear", "summarize"), disable_tqdm=True)
    kept = [(t, m) for t, m in zip(out["trajectory_ids"], out["loss_masks"]) if t.instance_id == "linear"]
    trainer, run = wandb_trainer()
    callback = index(records)
    callback.on_step_end(trainer, step_end(trajectory_ids=[t for t, _ in kept], loss_masks=[m for _, m in kept]), None)
    assert callback.close()

    ((artifact, _),) = run.logged
    assert {row["instance_id"]: row["trained"] for row in artifact.index()} == {"linear": True, "summarize": False}


@pytest.mark.asyncio
async def test_mirrored_records_are_referenced_not_copied(router, tmp_path, trials, make_generator) -> None:
    mirror = f"memory://skycap-{uuid.uuid4().hex[:8]}"
    service = service_for(
        router, tmp_path / "record", record_mirror=mirror, record_mirror_config={"exclude": ["experts"]}
    )
    service.start()
    try:
        records = RecordLog()
        await make_generator(records, service).generate(batch("linear"), disable_tqdm=True)
    finally:
        service.stop()
    trainer, run = wandb_trainer()
    callback = index(records)
    callback.on_step_end(trainer, step_end(), None)
    assert callback.close()

    ((artifact, _),) = run.logged
    (row,) = artifact.index()
    name = f"{row['id']}.json.zst"
    assert row["record"] == {"path": str(tmp_path / "record" / name), "mirror": f"{mirror}/{name}"}
    assert artifact.references == [(f"{mirror}/{name}", f"records/{name}", False)]
    assert set(artifact.files) == {"step.json"} and artifact.metadata["num_referenced"] == 1
    # The mirror config reached the server: its copy left the routed experts out, and says so.
    with fsspec.open(f"{mirror}/{name}", "rb") as handle:
        mirrored = orjson.loads(zstandard.ZstdDecompressor().decompressobj().decompress(handle.read()))
    assert mirrored["omitted_sidecars"] == ["experts"] and "experts" not in mirrored["sidecars"]
    # Without the batch's trajectory ids the trainer said nothing about what trained.
    assert row["trained"] is None


@pytest.mark.asyncio
async def test_eval_goes_to_its_own_artifact_only_when_asked(skycap, trials, make_generator) -> None:
    eval_batch = {**batch("linear"), "batch_metadata": SimpleNamespace(global_step=3, training_phase="eval")}
    for phases, expected in ((["train"], []), (["train", "eval"], ["skycap-records-eval-run-1"])):
        records = RecordLog()
        await make_generator(records).generate(eval_batch, disable_tqdm=True)
        trainer, run = wandb_trainer()
        callback = index(records, phases=phases)
        callback.on_eval_end(trainer, step_end(), None)
        assert callback.close()
        assert [artifact.name for artifact, _ in run.logged] == expected
        assert records.take("eval") == []
    ((artifact, aliases),) = run.logged
    assert aliases == ["eval-step-3", "latest"]
    assert [(row["phase"], row["trained"]) for row in artifact.index()] == [("eval", False)]


@pytest.mark.asyncio
async def test_nothing_is_logged_without_wandb(skycap, trials, make_generator) -> None:
    records = RecordLog()
    await make_generator(records).generate(batch("linear"), disable_tqdm=True)
    trainer, run = wandb_trainer(backend="tensorboard")
    callback = index(records)
    callback.on_step_end(trainer, step_end(), None)
    assert callback.close()
    assert run.logged == [] and records.take("train") == []


def entries(records: RecordLog, *names: str) -> RecordLog:
    """``records`` with a finished, local-only trajectory per name."""
    for name in names:
        result = SimpleNamespace(status="finished", record=None)
        trajectory = SimpleNamespace(id=f"tr_{name}", result=result, finishing={"reward": 1.0})
        records.add("train", SimpleNamespace(instance_id=name, repetition_id=0), 0, trajectory)
    return records


def test_an_error_that_may_pass_is_retried(warnings) -> None:
    trainer, run = wandb_trainer(FakeRun(failures=2))
    callback = index(entries(RecordLog(), "a"))
    callback.on_step_end(trainer, step_end(), None)
    assert callback.close()
    assert len(run.logged) == 1 and run.calls == 3
    assert callback.stats()["logged"] == 1 and callback.stats()["retried"] == 2


def test_a_wandb_that_keeps_failing_fails_open(warnings) -> None:
    for failures, error, calls in ((3, ConnectionError, 3), (1, ValueError, 1)):
        trainer, run = wandb_trainer(FakeRun(failures=failures, error=error))
        callback = index(entries(RecordLog(), "a"))
        callback.on_step_end(trainer, step_end(), None)  # returns: the step goes on
        assert callback.close()
        assert run.logged == [] and run.calls == calls
        assert callback.stats()["failed"] == 1
    assert any("logging skycap-records-train-run-1:train-step-3 failed" in message for message in warnings)


def test_a_version_wandb_accepted_is_never_retried(warnings) -> None:
    trainer, run = wandb_trainer(FakeRun(wait_error=TimeoutError))
    callback = index(entries(RecordLog(), "a"))
    callback.on_step_end(trainer, step_end(), None)
    assert callback.close()
    # A retry would log a second version of the same step.
    assert run.calls == 1 and callback.stats()["failed"] == 1 and callback.stats()["retried"] == 0


def test_a_slow_wandb_is_abandoned_not_retried_and_never_holds_up_the_step(warnings) -> None:
    hang = threading.Event()
    trainer, run = wandb_trainer(FakeRun(hang=hang))
    callback = index(entries(RecordLog(), "a"), timeout=0.2)
    started = time.monotonic()
    callback.on_step_end(trainer, step_end(), None)
    assert time.monotonic() - started < 0.2
    assert callback.close()
    hang.set()
    time.sleep(0.05)
    assert run.calls == 1
    assert callback.stats()["timed_out"] == 1 and callback.stats()["failed"] == 1
    assert any("abandoned" in message for message in warnings)


def test_a_full_queue_drops_and_shutdown_stops_at_its_deadline(warnings) -> None:
    hang = threading.Event()
    trainer, run = wandb_trainer(FakeRun(hang=hang))
    records = RecordLog()
    callback = index(records, queue_size=1, timeout=30.0)
    entries(records, "a")
    callback.on_step_end(trainer, step_end(step=1), None)
    deadline = time.monotonic() + 10
    while run.calls == 0:  # the worker holds step 1; the queue is empty
        assert time.monotonic() < deadline
        time.sleep(0.01)
    for step in (2, 3):
        entries(records, "a")
        callback.on_step_end(trainer, step_end(step=step), None)
    assert callback.stats()["dropped"] == 1  # step 3

    started = time.monotonic()
    assert not callback.close(timeout=0.2)
    assert time.monotonic() - started < 5.0
    # Step 2 was still queued at the deadline; step 1 is left to finish on its own.
    assert callback.stats()["dropped"] == 2 and callback.stats()["pending"] == 1
    hang.set()
    assert any("queue full" in message for message in warnings)
    assert any("shutdown deadline" in message for message in warnings)


def test_a_bug_in_the_callback_raises_into_the_step(monkeypatch) -> None:
    def broken(*args):
        raise KeyError("attempt")

    monkeypatch.setattr(record_index, "index_rows", broken)
    trainer, run = wandb_trainer()
    callback = index(entries(RecordLog(), "a"))
    with pytest.raises(KeyError):
        callback.on_step_end(trainer, step_end(), None)
    assert callback.close() and run.calls == 0


def test_the_config() -> None:
    cfg = HarborSkycapConfig().skycap
    assert cfg.wandb.enabled and cfg.wandb.phases == ["train"] and cfg.record_mirror is None
    assert cfg.record_mirror_config == {}
    with pytest.raises(ValueError, match="unknown phases"):
        SkycapRecordIndex(RecordLog(), ["train", "test"])
