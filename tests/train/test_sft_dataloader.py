"""CPU tests for the SFT stateful dataloader and samplers.

Covers the stateful-dataloader RFC test plan:
  - default random dataloader order is deterministic for identical seeds
  - random dataloader order differs across different seeds
  - sequential sampler yields examples in order
  - sequential sampler state_dict/load_state_dict resumes at the next sample
  - custom sampler state is included in the dataloader checkpoint
  - data-mixing / curriculum samplers resume correctly

and the multi-dataset RFC (#1875) test plan:
  - user-specified train_dataset_weights: per-batch source representation
    tracks the weights
  - default weights: equal mixing across datasets
  - mixing ratios are independent of the individual dataset sizes
  - DataMixingSampler draws a fresh plan every epoch and resumes exactly
    (mid-epoch and across epoch boundaries) via its RNG state

The custom-sampler path is exercised with small test-local samplers. The
``CurriculumLearningSampler`` example under
``examples/train/sft/curriculum_sampler.py`` is loaded by file path.

Run::

    uv run --extra dev --extra skyrl-train pytest tests/train/test_sft_dataloader.py -v
"""

import importlib.util
import random
from pathlib import Path

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from skyrl.train.config import SFTConfig
from skyrl.train.dataset.samplers import (
    DataMixingSampler,
    StatefulSequentialSampler,
    import_sampler_class,
)
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


class _LengthsAwareSampler(_ExampleKwargSampler):
    """Test-local custom sampler that accepts the ``lengths`` kwarg the trainer
    injects when multiple training datasets are configured."""

    def __init__(self, data_source, lengths=None, num_samples=None, seed=0):
        super().__init__(data_source, num_samples=num_samples, seed=seed)
        self.lengths = lengths


_LENGTHS_SAMPLER_PATH = f"{__name__}._LengthsAwareSampler"


def _load_example_cls(filename: str, class_name: str):
    """Import a sampler class from an examples/train/sft/*.py file by path."""
    path = Path(__file__).resolve().parents[2] / "examples" / "train" / "sft" / filename
    spec = importlib.util.spec_from_file_location(f"{filename[:-3]}_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def _load_curriculum_cls():
    """Import the example CurriculumLearningSampler by file path."""
    return _load_example_cls("curriculum_sampler.py", "CurriculumLearningSampler")


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

    def test_random_single_source_lengths_returns_none(self):
        trainer = _make_trainer(sampler="random")
        assert trainer.build_train_sampler(list(range(10)), dataset_lengths=[10]) is None

    def test_random_multi_dataset_returns_mixing_sampler(self):
        trainer = _make_trainer(sampler="random", train_dataset_weights=[0.8, 0.2], seed=11)
        sampler = trainer.build_train_sampler(list(range(20)), dataset_lengths=[10, 10])
        assert isinstance(sampler, DataMixingSampler)
        assert len(sampler) == 20
        # Seeded from the config so runs are reproducible.
        other = _make_trainer(sampler="random", train_dataset_weights=[0.8, 0.2], seed=11).build_train_sampler(
            list(range(20)), dataset_lengths=[10, 10]
        )
        assert list(sampler) == list(other)

    def test_random_multi_dataset_defaults_to_equal_weights(self):
        # Bare trainers skip config normalization; equal mixing is the fallback.
        trainer = _make_trainer(sampler="random")
        sampler = trainer.build_train_sampler(list(range(20)), dataset_lengths=[10, 10])
        assert isinstance(sampler, DataMixingSampler)

    def test_custom_multi_dataset_injects_lengths(self):
        trainer = _make_trainer(
            sampler="custom",
            sampler_class_path=_LENGTHS_SAMPLER_PATH,
            sampler_kwargs={"num_samples": 8},
        )
        sampler = trainer.build_train_sampler(list(range(20)), dataset_lengths=[12, 8])
        assert isinstance(sampler, _LengthsAwareSampler)
        assert sampler.lengths == [12, 8]

    def test_custom_multi_dataset_user_lengths_win(self):
        trainer = _make_trainer(
            sampler="custom",
            sampler_class_path=_LENGTHS_SAMPLER_PATH,
            sampler_kwargs={"num_samples": 8, "lengths": [1, 19]},
        )
        sampler = trainer.build_train_sampler(list(range(20)), dataset_lengths=[12, 8])
        assert sampler.lengths == [1, 19]

    def test_custom_single_dataset_does_not_inject_lengths(self):
        trainer = _make_trainer(
            sampler="custom",
            sampler_class_path=_LENGTHS_SAMPLER_PATH,
            sampler_kwargs={"num_samples": 8},
        )
        sampler = trainer.build_train_sampler(list(range(20)), dataset_lengths=[20])
        assert sampler.lengths is None


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

    def test_keeps_partial_tail_batch(self):
        # drop_last=False: 41 items, batch_size 4 -> ceil(41/4) = 11 batches
        # (the trailing partial batch is padded in collate, not dropped).
        data = list(range(41))
        dl = _make_trainer(sampler="random", batch_size=4, seed=1).build_train_dataloader(data)
        assert len(dl) == 11

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


class TestEmptyDataloaderInvariant:
    """With drop_last=False a sub-batch_size dataset is padded into one batch
    (not dropped); the dataloader is only empty when the sampler yields nothing,
    which is the condition train() guards against (opaque StopIteration)."""

    def test_dataset_smaller_than_batch_size_is_one_batch(self):
        # Built-in sampler: fewer examples than batch_size -> 1 (padded) batch.
        dl = _make_trainer(sampler="random", batch_size=4).build_train_dataloader(list(range(2)))
        assert len(dl) == 1

    def test_custom_sampler_num_samples_below_batch_size_is_one_batch(self):
        # Custom sampler shrinks the effective length via num_samples, but the
        # tail is padded rather than dropped -> 1 batch.
        dl = _make_trainer(
            sampler="custom",
            batch_size=4,
            sampler_class_path=_CUSTOM_SAMPLER_PATH,
            sampler_kwargs={"num_samples": 2},
        ).build_train_dataloader(list(range(100)))
        assert len(dl) == 1

    def test_empty_dataset_is_zero_batches(self):
        # Only a truly empty dataset yields 0 batches (the guarded condition).
        # Uses the sequential sampler: the built-in random path errors earlier
        # (torch's RandomSampler rejects an empty data_source at construction).
        dl = _make_trainer(sampler="sequential", batch_size=4).build_train_dataloader([])
        assert len(dl) == 0


class TestTailBatchPadding:
    """The final short batch is padded up to batch_size (via the real collator),
    with padded rows masked out of the loss, so every example is trained on."""

    @staticmethod
    def _example(n_in=6, n_act=3):
        return {
            "input_ids": list(range(1, n_in + 1)),
            "attention_mask": [1] * n_in,
            "num_actions": n_act,
            "loss_mask": [1] * n_act,
        }

    def _trainer_with_real_collator(self, batch_size=4, micro=2):
        import types

        from skyrl.train.dataset.collators import DefaultCollator

        trainer = _make_trainer(sampler="sequential", batch_size=batch_size)
        trainer.collator = DefaultCollator(types.SimpleNamespace(pad_token_id=0), micro)
        return trainer

    def test_tail_batch_padded_and_masked(self):
        trainer = self._trainer_with_real_collator(batch_size=4)
        data = [self._example() for _ in range(6)]  # 1 full batch + tail of 2
        dl = trainer.build_train_dataloader(data)
        assert len(dl) == 2  # ceil(6/4), nothing dropped

        last = list(dl)[-1]
        # Padded up to batch_size rows...
        assert last["sequences"].shape[0] == 4
        # ...with the 2 padded rows masked out of the loss (loss_mask == 0),
        # while the 2 real rows keep their (scaled, nonzero) loss_mask.
        loss_mask = last["loss_mask"]
        assert (loss_mask[2:] == 0).all()
        assert (loss_mask[:2] != 0).any()

    def test_full_batches_unaffected(self):
        trainer = self._trainer_with_real_collator(batch_size=4)
        data = [self._example() for _ in range(8)]  # exactly 2 full batches
        dl = trainer.build_train_dataloader(data)
        assert len(dl) == 2
        for batch in dl:
            assert batch["sequences"].shape[0] == 4
            assert (batch["loss_mask"] != 0).any()


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


# ---------------------------------------------------------------------------
# DataMixingSampler (core)
# ---------------------------------------------------------------------------


class TestDataMixingSampler:
    # data = two sources of 10 each: indices [0,10) = source 0, [10,20) = source 1.
    DATA = list(range(20))
    LENGTHS = [10, 10]

    def _make(self, weights=(0.7, 0.3), num_samples=40, seed=3, data=None, lengths=None):
        return DataMixingSampler(
            data if data is not None else self.DATA,
            lengths=lengths if lengths is not None else self.LENGTHS,
            weights=list(weights),
            num_samples=num_samples,
            seed=seed,
        )

    def test_len_matches_num_samples(self):
        sampler = self._make(weights=[0.5, 0.5], num_samples=64)
        assert len(sampler) == 64

    def test_num_samples_defaults_to_dataset_length(self):
        sampler = DataMixingSampler(self.DATA, lengths=self.LENGTHS, weights=[0.5, 0.5])
        assert len(sampler) == 20

    def test_validates_lengths_sum(self):
        with pytest.raises(ValueError, match="must equal len"):
            DataMixingSampler(self.DATA, lengths=[10, 5], weights=[0.5, 0.5])

    def test_validates_weights_align(self):
        with pytest.raises(ValueError, match="must align"):
            DataMixingSampler(self.DATA, lengths=self.LENGTHS, weights=[1.0])

    def test_rejects_degenerate_weights(self):
        with pytest.raises(ValueError, match="non-negative with a positive sum"):
            DataMixingSampler(self.DATA, lengths=self.LENGTHS, weights=[0.0, 0.0])

    def test_deterministic_for_same_seed(self):
        a = list(self._make(weights=[0.8, 0.2], num_samples=200, seed=1))
        b = list(self._make(weights=[0.8, 0.2], num_samples=200, seed=1))
        assert a == b

    def test_differs_for_different_seed(self):
        a = list(self._make(weights=[0.8, 0.2], num_samples=200, seed=1))
        b = list(self._make(weights=[0.8, 0.2], num_samples=200, seed=2))
        assert a != b

    def test_weighting_biases_the_mix(self):
        # Source 0 weighted 4x source 1 -> ~80% of draws from indices [0,10).
        plan = list(self._make(weights=[0.8, 0.2], num_samples=4000, seed=7))
        frac_source0 = sum(1 for idx in plan if idx < 10) / len(plan)
        assert 0.75 < frac_source0 < 0.85, frac_source0

    def test_equal_weights_balanced(self):
        plan = list(self._make(weights=[0.5, 0.5], num_samples=4000, seed=7))
        frac_source0 = sum(1 for idx in plan if idx < 10) / len(plan)
        assert 0.45 < frac_source0 < 0.55, frac_source0

    def test_size_imbalance_still_matches_weights(self):
        # Source 0 is tiny (2 examples) but weighted equally; source-level mix
        # should still be ~50/50 because weight is divided across examples.
        data = list(range(12))  # source 0: [0,2), source 1: [2,12)
        plan = list(self._make(data=data, lengths=[2, 10], weights=[0.5, 0.5], num_samples=4000, seed=9))
        frac_source0 = sum(1 for idx in plan if idx < 2) / len(plan)
        assert 0.45 < frac_source0 < 0.55, frac_source0

    def test_fresh_plan_each_epoch(self):
        sampler = self._make()
        epoch1, epoch2 = list(sampler), list(sampler)
        assert epoch1 != epoch2, "each epoch must draw a fresh plan"
        # ...but the whole multi-epoch stream is reproducible from the seed.
        other = self._make()
        assert [list(other), list(other)] == [epoch1, epoch2]

    def test_state_dict_resume(self):
        sampler = self._make()
        it = iter(sampler)
        consumed = [next(it) for _ in range(11)]
        state = sampler.state_dict()
        assert state["position"] == 11
        assert "generator_state" in state
        rest = list(it)

        # A different construction seed: the checkpointed generator state alone
        # must determine the restored plan.
        resumed = self._make(seed=999)
        resumed.load_state_dict(state)
        assert list(resumed) == rest
        assert consumed + rest == list(self._make())

    def test_mid_epoch_resume_matches_across_epoch_boundary(self):
        sampler = self._make()
        it = iter(sampler)
        [next(it) for _ in range(11)]
        state = sampler.state_dict()
        # Uninterrupted: rest of the current epoch, then the full next epoch.
        expected = list(it) + list(sampler)

        resumed = self._make(seed=999)
        resumed.load_state_dict(state)
        assert list(resumed) + list(resumed) == expected

    def test_between_epoch_resume_draws_same_next_epoch(self):
        sampler = self._make()
        list(sampler)  # exhaust epoch 1
        state = sampler.state_dict()
        epoch2 = list(sampler)

        resumed = self._make(seed=999)
        resumed.load_state_dict(state)
        assert list(resumed) == epoch2

    def test_position_only_state_loads(self):
        # A position-only state (no generator_state key) loads with a warning
        # and resumes the cursor against the freshly-seeded generator.
        sampler = self._make()
        sampler.load_state_dict({"position": 5})
        assert sampler.position == 5
        assert len(list(sampler)) == 35  # resumes the cursor best-effort

    def _make_dl(self):
        sampler = self._make(num_samples=40, seed=5)
        return StatefulDataLoader(
            self.DATA,
            batch_size=4,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            collate_fn=lambda x: list(x),
        )

    def test_resume_through_stateful_dataloader(self):
        """The sampler's state is captured in the dataloader checkpoint and resumes."""
        dl = self._make_dl()
        it = iter(dl)
        next(it)
        next(it)
        state = dl.state_dict()
        assert "_sampler_iter_state" in state
        rest_full = [b for b in it]

        dl2 = self._make_dl()
        dl2.load_state_dict(state)
        rest_resumed = [b for b in dl2]
        assert rest_full == rest_resumed

    def test_dataloader_epochs_differ_and_resume_after_boundary(self):
        """Fresh plans survive the trainer's epoch-boundary iter() re-creation,
        and a checkpoint taken in a later epoch resumes exactly."""
        dl = self._make_dl()
        epoch1 = list(dl)
        it = iter(dl)  # epoch 2: the sampler re-plans on the fresh iterator
        first_batch = next(it)
        state = dl.state_dict()
        rest_full = [b for b in it]
        assert [first_batch] + rest_full != epoch1, "epoch 2 must be a fresh plan"

        dl2 = self._make_dl()
        dl2.load_state_dict(state)
        rest_resumed = [b for b in dl2]
        assert rest_full == rest_resumed


# ---------------------------------------------------------------------------
# Multi-dataset mixing through build_train_dataloader (RFC #1875 test plan)
# ---------------------------------------------------------------------------


class TestMultiDatasetMixing:
    """Per-batch source representation tracks ``train_dataset_weights``,
    independent of the individual dataset sizes."""

    @staticmethod
    def _source0_fraction(trainer, data, dataset_lengths, boundary, epochs=10):
        dl = trainer.build_train_dataloader(data, dataset_lengths=dataset_lengths)
        drawn = []
        for _ in range(epochs):
            drawn.extend(_flatten(dl))
        return sum(1 for idx in drawn if idx < boundary) / len(drawn)

    def test_user_weights_reflected_in_batches(self):
        # Two sources of 100 each, weighted 80/20.
        data = list(range(200))
        trainer = _make_trainer(sampler="random", batch_size=20, seed=3, train_dataset_weights=[0.8, 0.2])
        frac = self._source0_fraction(trainer, data, [100, 100], boundary=100)
        assert 0.75 < frac < 0.85, frac

    def test_default_weights_mix_equally(self):
        data = list(range(200))
        trainer = _make_trainer(sampler="random", batch_size=20, seed=3)
        frac = self._source0_fraction(trainer, data, [100, 100], boundary=100)
        assert 0.45 < frac < 0.55, frac

    def test_mixing_independent_of_dataset_sizes(self):
        # Source 0 has 20 examples, source 1 has 180; equal weights should
        # still yield ~50/50 representation in batches.
        data = list(range(200))
        trainer = _make_trainer(sampler="random", batch_size=20, seed=3, train_dataset_weights=[0.5, 0.5])
        frac = self._source0_fraction(trainer, data, [20, 180], boundary=20)
        assert 0.45 < frac < 0.55, frac
