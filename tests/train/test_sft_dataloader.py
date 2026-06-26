"""CPU tests for the SFT stateful dataloader and custom samplers.

Covers the RFC test plan:
  - default random dataloader order is deterministic for identical seeds
  - random dataloader order differs across different seeds
  - sequential sampler yields examples in order
  - sequential sampler state_dict/load_state_dict resumes at the next sample
  - custom sampler state is included in the dataloader checkpoint
  - data-mixing / curriculum samplers resume correctly

The custom-sampler path is exercised with a small test-local sampler (the core
library ships only ``StatefulSequentialSampler``; curriculum learning is a
user-supplied example under ``examples/train/sft/curriculum_sampler.py``, which
is loaded by file path here to keep it covered).

Run::

    uv run --extra dev --extra skyrl-train pytest tests/train/test_sft_dataloader.py -v
"""

import importlib.util
import random
from pathlib import Path

import pytest
import torch

from skyrl.train.config import SFTConfig
from skyrl.train.dataset.samplers import StatefulSequentialSampler, import_sampler_class
from skyrl.train.sft_trainer import SFTTrainer

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _ExampleKwargSampler(torch.utils.data.Sampler[int]):
    """Test-local custom sampler: a deterministic shuffled plan, checkpointable.

    Exists to exercise the ``sampler='custom'`` path (kwargs forwarding + state
    in the dataloader checkpoint) without depending on any shipped example.
    """

    def __init__(self, data_source, num_samples=None, seed=0):
        n = len(data_source)
        self.num_samples = num_samples if num_samples is not None else n
        self.position = 0
        rng = random.Random(seed)
        plan = []
        while len(plan) < self.num_samples:
            order = list(range(n))
            rng.shuffle(order)
            plan.extend(order)
        self._plan = plan[: self.num_samples]

    def __iter__(self):
        while self.position < len(self._plan):
            idx = self._plan[self.position]
            self.position += 1
            yield idx
        self.position = 0

    def __len__(self):
        return len(self._plan)

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


_CUSTOM_SAMPLER_PATH = f"{__name__}._ExampleKwargSampler"


def _load_curriculum_cls():
    """Import the example CurriculumLearningSampler by file path."""
    path = Path(__file__).resolve().parents[2] / "examples" / "train" / "sft" / "curriculum_sampler.py"
    spec = importlib.util.spec_from_file_location("curriculum_sampler_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CurriculumLearningSampler


def _make_trainer(**overrides) -> SFTTrainer:
    """Build a bare SFTTrainer (no setup/Ray/GPU) for dataloader-building tests.

    Bypasses ``__init__`` so we can exercise ``build_train_sampler`` /
    ``build_train_dataloader`` with a trivial identity collator and a plain
    list dataset.
    """
    cfg = SFTConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    trainer = object.__new__(SFTTrainer)
    trainer.sft_cfg = cfg
    # Identity collator: returns the list of sampled items so we can inspect order.
    trainer.collator = lambda examples, batch_size: list(examples)
    return trainer


def _flatten(batches) -> list:
    out = []
    for b in batches:
        out.extend(b)
    return out


# ---------------------------------------------------------------------------
# StatefulSequentialSampler (core)
# ---------------------------------------------------------------------------


class TestStatefulSequentialSampler:
    def test_yields_in_order(self):
        sampler = StatefulSequentialSampler(list(range(5)))
        assert list(sampler) == [0, 1, 2, 3, 4]

    def test_len(self):
        assert len(StatefulSequentialSampler(list(range(7)))) == 7

    def test_resets_after_exhaustion(self):
        sampler = StatefulSequentialSampler(list(range(3)))
        assert list(sampler) == [0, 1, 2]
        # A fresh pass starts from the top again.
        assert list(sampler) == [0, 1, 2]

    def test_state_dict_resumes_at_next_sample(self):
        sampler = StatefulSequentialSampler(list(range(10)))
        it = iter(sampler)
        first = [next(it), next(it), next(it)]
        assert first == [0, 1, 2]
        state = sampler.state_dict()
        assert state == {"position": 3}

        resumed = StatefulSequentialSampler(list(range(10)))
        resumed.load_state_dict(state)
        assert list(resumed) == [3, 4, 5, 6, 7, 8, 9]


# ---------------------------------------------------------------------------
# import_sampler_class
# ---------------------------------------------------------------------------


class TestImportSamplerClass:
    def test_imports_class(self):
        cls = import_sampler_class("skyrl.train.dataset.samplers.StatefulSequentialSampler")
        assert cls is StatefulSequentialSampler

    def test_imports_test_local_class(self):
        cls = import_sampler_class(_CUSTOM_SAMPLER_PATH)
        assert cls is _ExampleKwargSampler

    def test_rejects_bare_name(self):
        with pytest.raises(ValueError, match="dotted path"):
            import_sampler_class("StatefulSequentialSampler")


# ---------------------------------------------------------------------------
# build_train_sampler dispatch
# ---------------------------------------------------------------------------


class TestBuildTrainSampler:
    def test_random_returns_none(self):
        trainer = _make_trainer(sampler="random")
        assert trainer.build_train_sampler(list(range(10))) is None

    def test_sequential(self):
        trainer = _make_trainer(sampler="sequential")
        sampler = trainer.build_train_sampler(list(range(10)))
        assert isinstance(sampler, StatefulSequentialSampler)

    def test_custom(self):
        trainer = _make_trainer(
            sampler="custom",
            sampler_class_path=_CUSTOM_SAMPLER_PATH,
            sampler_kwargs={"num_samples": 8, "seed": 0},
        )
        sampler = trainer.build_train_sampler(list(range(10)))
        assert isinstance(sampler, _ExampleKwargSampler)
        assert len(sampler) == 8

    def test_custom_requires_class_path(self):
        trainer = _make_trainer(sampler="custom", sampler_class_path=None)
        with pytest.raises(ValueError, match="sampler_class_path"):
            trainer.build_train_sampler(list(range(10)))

    def test_unknown_sampler(self):
        trainer = _make_trainer(sampler="bogus")
        with pytest.raises(ValueError, match="Unknown sampler"):
            trainer.build_train_sampler(list(range(10)))


# ---------------------------------------------------------------------------
# build_train_dataloader: determinism + checkpoint/resume
# ---------------------------------------------------------------------------


class TestRandomDataloader:
    def test_deterministic_for_same_seed(self):
        data = list(range(40))
        dl_a = _make_trainer(sampler="random", batch_size=4, seed=123).build_train_dataloader(data)
        dl_b = _make_trainer(sampler="random", batch_size=4, seed=123).build_train_dataloader(data)
        assert _flatten(dl_a) == _flatten(dl_b)

    def test_differs_for_different_seed(self):
        data = list(range(40))
        dl_a = _make_trainer(sampler="random", batch_size=4, seed=123).build_train_dataloader(data)
        dl_b = _make_trainer(sampler="random", batch_size=4, seed=999).build_train_dataloader(data)
        assert _flatten(dl_a) != _flatten(dl_b)

    def test_drop_last(self):
        # 41 items, batch_size 4 -> 10 full batches, last partial dropped.
        data = list(range(41))
        dl = _make_trainer(sampler="random", batch_size=4, seed=1).build_train_dataloader(data)
        assert len(dl) == 10

    def test_resume_mid_epoch(self):
        data = list(range(40))
        dl = _make_trainer(sampler="random", batch_size=4, seed=123).build_train_dataloader(data)
        it = iter(dl)
        next(it)
        next(it)
        state = dl.state_dict()
        rest_full = [b for b in it]

        dl2 = _make_trainer(sampler="random", batch_size=4, seed=123).build_train_dataloader(data)
        dl2.load_state_dict(state)
        rest_resumed = [b for b in dl2]
        assert rest_full == rest_resumed


class TestSequentialDataloader:
    def test_yields_in_order(self):
        data = list(range(20))
        dl = _make_trainer(sampler="sequential", batch_size=5).build_train_dataloader(data)
        assert _flatten(dl) == list(range(20))

    def test_resume_mid_epoch(self):
        data = list(range(20))
        dl = _make_trainer(sampler="sequential", batch_size=5).build_train_dataloader(data)
        it = iter(dl)
        first = next(it)
        assert first == [0, 1, 2, 3, 4]
        state = dl.state_dict()
        rest_full = [b for b in it]

        dl2 = _make_trainer(sampler="sequential", batch_size=5).build_train_dataloader(data)
        dl2.load_state_dict(state)
        rest_resumed = [b for b in dl2]
        assert rest_full == rest_resumed
        assert _flatten(rest_resumed) == list(range(5, 20))


class TestCustomSamplerDataloaderResume:
    """Custom sampler state is captured in the dataloader checkpoint."""

    def _build(self, data):
        trainer = _make_trainer(
            sampler="custom",
            batch_size=4,
            sampler_class_path=_CUSTOM_SAMPLER_PATH,
            sampler_kwargs={"num_samples": 36, "seed": 5},
        )
        return trainer.build_train_dataloader(data)

    def test_state_in_checkpoint(self):
        data = list(range(12))
        dl = self._build(data)
        it = iter(dl)
        next(it)
        next(it)
        state = dl.state_dict()
        # The custom sampler's position is nested in the dataloader state.
        assert "_sampler_iter_state" in state

    def test_resume_mid_epoch(self):
        data = list(range(12))
        dl = self._build(data)
        it = iter(dl)
        next(it)
        next(it)
        state = dl.state_dict()
        rest_full = [b for b in it]

        dl2 = self._build(data)
        dl2.load_state_dict(state)
        rest_resumed = [b for b in dl2]
        assert rest_full == rest_resumed


def test_random_dataloader_uses_shuffle_not_sampler():
    """The 'random' path must rely on shuffle=True (sampler is None)."""
    trainer = _make_trainer(sampler="random", batch_size=4, seed=1)
    dl = trainer.build_train_dataloader(list(range(20)))
    # torch DataLoader replaces a None sampler with an internal RandomSampler
    # because shuffle=True; confirm shuffling actually happened.
    assert _flatten(dl) != list(range(20))
    assert isinstance(dl.generator, torch.Generator)


# ---------------------------------------------------------------------------
# Example: CurriculumLearningSampler (shipped under examples/, loaded by path)
# ---------------------------------------------------------------------------


class TestCurriculumExample:
    def test_len_matches_num_samples(self):
        cls = _load_curriculum_cls()
        sampler = cls(list(range(30)), lengths=[10, 10, 10], num_samples=24)
        assert len(sampler) == 24

    def test_defaults_to_dataset_length(self):
        cls = _load_curriculum_cls()
        assert len(cls(list(range(12)))) == 12

    def test_validates_lengths_sum(self):
        cls = _load_curriculum_cls()
        with pytest.raises(ValueError, match="must equal len"):
            cls(list(range(30)), lengths=[10, 10])

    def test_rejects_nonpositive_lengths(self):
        cls = _load_curriculum_cls()
        with pytest.raises(ValueError, match="lengths must be > 0"):
            cls(list(range(10)), lengths=[10, 0])

    def test_deterministic_for_same_seed(self):
        cls = _load_curriculum_cls()
        a = list(cls(list(range(30)), lengths=[10, 10, 10], num_samples=30, seed=7))
        b = list(cls(list(range(30)), lengths=[10, 10, 10], num_samples=30, seed=7))
        assert a == b

    def test_differs_for_different_seed(self):
        cls = _load_curriculum_cls()
        a = list(cls(list(range(30)), lengths=[10, 10, 10], num_samples=30, seed=1))
        b = list(cls(list(range(30)), lengths=[10, 10, 10], num_samples=30, seed=2))
        assert a != b

    def test_progressive_unlocking(self):
        cls = _load_curriculum_cls()
        # First stage only unlocks subset 0 (indices [0,10)); last stage unlocks all.
        plan = list(cls(list(range(30)), lengths=[10, 10, 10], num_samples=30, seed=0))
        first_stage = plan[:10]
        last_stage = plan[20:]
        assert all(idx < 10 for idx in first_stage), first_stage
        assert any(idx >= 20 for idx in last_stage), last_stage

    def test_state_dict_resume(self):
        cls = _load_curriculum_cls()
        sampler = cls(list(range(30)), lengths=[10, 10, 10], num_samples=30, seed=3)
        it = iter(sampler)
        consumed = [next(it) for _ in range(7)]
        state = sampler.state_dict()
        assert state == {"position": 7}
        rest = list(it)

        resumed = cls(list(range(30)), lengths=[10, 10, 10], num_samples=30, seed=3)
        resumed.load_state_dict(state)
        assert list(resumed) == rest
        assert consumed + rest == list(cls(list(range(30)), lengths=[10, 10, 10], num_samples=30, seed=3))
