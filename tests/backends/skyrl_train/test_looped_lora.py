import pytest

from skyrl.train.looped_lora import (
    LayerExecution,
    build_looped_lora_schedule,
    get_lora_only_executions_by_physical_layer,
)


@pytest.mark.parametrize(
    ("repeat_count", "expected_length", "expected_lora_only"),
    [(1, 36, 0), (2, 44, 8), (4, 60, 24)],
)
def test_middle_section_repeats_only_lora_after_first_pass(
    repeat_count: int,
    expected_length: int,
    expected_lora_only: int,
) -> None:
    schedule = build_looped_lora_schedule(
        36,
        [{"start_layer": 14, "end_layer": 22, "repeat_count": repeat_count}],
    )

    assert len(schedule) == expected_length
    assert sum(execution.lora_only for execution in schedule) == expected_lora_only
    assert [execution.physical_layer for execution in schedule if not execution.lora_only] == list(range(36))


def test_sections_can_have_independent_repeat_counts() -> None:
    schedule = build_looped_lora_schedule(
        36,
        [
            {"start_layer": 0, "end_layer": 4, "repeat_count": 2},
            {"start_layer": 4, "end_layer": 20, "repeat_count": 3},
            {"start_layer": 20, "end_layer": 36, "repeat_count": 1},
        ],
    )

    assert len(schedule) == 72
    assert schedule[:4] == tuple(LayerExecution(layer, False) for layer in range(4))
    assert schedule[4:8] == tuple(LayerExecution(layer, True) for layer in range(4))
    assert schedule[8:24] == tuple(LayerExecution(layer, False) for layer in range(4, 20))
    assert schedule[24:40] == tuple(LayerExecution(layer, True) for layer in range(4, 20))
    assert schedule[40:56] == tuple(LayerExecution(layer, True) for layer in range(4, 20))
    assert schedule[56:] == tuple(LayerExecution(layer, False) for layer in range(20, 36))


def test_unconfigured_gaps_execute_once() -> None:
    schedule = build_looped_lora_schedule(
        8,
        [{"start_layer": 2, "end_layer": 4, "repeat_count": 2}],
    )

    assert schedule == (
        LayerExecution(0, False),
        LayerExecution(1, False),
        LayerExecution(2, False),
        LayerExecution(3, False),
        LayerExecution(2, True),
        LayerExecution(3, True),
        LayerExecution(4, False),
        LayerExecution(5, False),
        LayerExecution(6, False),
        LayerExecution(7, False),
    )


def test_lora_only_executions_get_distinct_logical_indices() -> None:
    schedule = build_looped_lora_schedule(
        6,
        [{"start_layer": 2, "end_layer": 4, "repeat_count": 3}],
    )

    assert get_lora_only_executions_by_physical_layer(6, schedule) == (
        (),
        (),
        (4, 6),
        (5, 7),
        (),
        (),
    )


@pytest.mark.parametrize(
    "sections",
    [
        [{"start_layer": 4, "end_layer": 4, "repeat_count": 2}],
        [{"start_layer": -1, "end_layer": 4, "repeat_count": 2}],
        [{"start_layer": 2, "end_layer": 4, "repeat_count": 0}],
        [{"start_layer": 2, "end_layer": 7, "repeat_count": 2}],
        [
            {"start_layer": 2, "end_layer": 5, "repeat_count": 2},
            {"start_layer": 4, "end_layer": 6, "repeat_count": 2},
        ],
    ],
)
def test_invalid_sections_fail_before_model_start(
    sections: list[dict[str, int]],
) -> None:
    with pytest.raises(ValueError):
        build_looped_lora_schedule(6, sections)
