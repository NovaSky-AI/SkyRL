"""project / run / trajectory.

A trajectory is one attempt, naming the project and run it belongs to, the task
it attempted and the step that produced it. These are fields rather than
annotations, because every question an RL run is read with -- one task across
steps, one step across tasks, the reward surface over both -- filters or groups
by them.

A **run is derived**. There is no run record, nothing creates one, and nothing
writes to one: it is the grouping its trajectories imply, computed when a
listing is read. So there is no counter to keep in step with its members, and
no way for a run and its trajectories to disagree.
"""

from __future__ import annotations

import pytest


async def attempt(stack, *, run_id, task_id, step, reward=None, project="rl", **fields):
    created = await stack.create_trajectory(
        project=project, run_id=run_id, task_id=task_id, step=step, **fields
    )
    await stack.chat(created, [{"role": "user", "content": f"{task_id}@{step}"}])
    await stack.finish(
        created["id"], annotations={"reward": reward} if reward is not None else {}
    )
    return created


# -- creation ---------------------------------------------------------------
async def test_a_run_is_created_by_the_first_trajectory_that_names_it(stack):
    """An RL harness knows its run id before it knows how many attempts there
    will be, so a separate create call would be one more thing to get wrong."""
    assert (await stack.get("/v1/runs")).json()["data"] == []

    created = await stack.create_trajectory(project="rl", run_id="run-7", task_id="t1", step=0)
    assert created["id"]

    runs = (await stack.get("/v1/runs")).json()["data"]
    assert [run["id"] for run in runs] == ["run-7"]
    assert runs[0]["project"] == "rl"
    assert runs[0]["trajectory_count"] == 1


async def test_a_second_trajectory_joins_the_run_rather_than_replacing_it(stack):
    await stack.create_trajectory(project="rl", run_id="run-7", task_id="t1", step=0)
    await stack.create_trajectory(project="rl", run_id="run-7", task_id="t2", step=0)

    run = await stack.run("run-7")
    assert run["trajectory_count"] == 2
    assert run["task_count"] == 2
    assert run["steps"] == [0]


async def test_a_run_is_derived_and_has_nothing_to_write_to(stack):
    """A run is not a record. It is the grouping its trajectories imply, so
    there is no row to create, to re-parent, or to annotate -- and no route
    that pretends otherwise."""
    await stack.create_trajectory(project="rl", run_id="run-7", task_id="t1", step=0)

    assert (await stack.run("run-7"))["trajectory_count"] == 1
    assert (
        await stack.patch("/v1/runs/run-7/metadata", {"annotations": {"lr": 1e-5}})
    ).status_code in (404, 405)
    assert (await stack.post("/v1/runs", {"id": "run-8"})).status_code in (404, 405)


async def test_a_trajectory_need_not_belong_to_a_run(stack):
    """`skyrl-capture run` against an agent harness is one trial and nothing
    more; a synthetic run around it would be a row nobody asked for."""
    created = await stack.create_trajectory(project="one-off")
    record = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert record["run_id"] is None
    assert record["task_id"] is None and record["step"] is None
    assert (await stack.get("/v1/runs")).json()["data"] == []


async def test_a_failed_create_leaves_no_empty_run(stack):
    """The run is inserted in the trajectory's own transaction."""
    rejected = await stack.post(
        "/v1/trajectories", {"project": "rl", "run_id": "run-9", "trajectory_id": "no/slash"}
    )
    assert rejected.status_code == 400
    assert await stack.run("run-9") is None


# -- the axes ---------------------------------------------------------------
async def test_trajectories_filter_by_every_level(stack):
    await attempt(stack, run_id="a", task_id="t1", step=0)
    await attempt(stack, run_id="a", task_id="t1", step=1)
    await attempt(stack, run_id="a", task_id="t2", step=1)
    await attempt(stack, run_id="b", task_id="t1", step=0)

    async def ids(query):
        return sorted(
            item["task_id"] + "@" + str(item["step"])
            for item in (await stack.get(f"/v1/trajectories?{query}")).json()["data"]
        )

    assert await ids("run_id=a") == ["t1@0", "t1@1", "t2@1"]
    assert await ids("run_id=a&task_id=t1") == ["t1@0", "t1@1"]
    assert await ids("run_id=a&step=1") == ["t1@1", "t2@1"]
    assert await ids("run_id=a&task_id=t1&step=1") == ["t1@1"]
    assert await ids("project=rl&step=0") == ["t1@0", "t1@0"]


async def test_step_zero_is_a_filter_and_not_a_missing_value(stack):
    """Step 0 is the first training step, and falsy. Filtering on it has to
    mean step 0 rather than 'no step given'."""
    await attempt(stack, run_id="a", task_id="t1", step=0)
    await attempt(stack, run_id="a", task_id="t1", step=1)

    listed = (await stack.get("/v1/trajectories?run_id=a&step=0")).json()["data"]
    assert [item["step"] for item in listed] == [0]


# -- run-scoped metadata lives on the trajectories --------------------------
async def test_what_belonged_to_a_run_is_annotated_on_its_trajectories(stack):
    """Hyperparameters and a git sha used to be run metadata. With runs
    derived, they are annotations on the attempts -- which is also what an
    export row carries, so a reader of the data has them without a join."""
    created = await stack.create_trajectory(
        project="rl", run_id="a", task_id="t1", step=0, annotations={"lr": 1e-5, "git_sha": "abc"}
    )

    row = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert row["annotations"] == {"lr": 1e-5, "git_sha": "abc"}
    assert (await stack.run("a"))["trajectory_count"] == 1


# -- exports ----------------------------------------------------------------
async def test_a_run_is_an_export_scope(stack):
    """"The rows this training run produced" is the export an RL harness wants."""
    await attempt(stack, run_id="a", task_id="t1", step=0, reward=1.0)
    await attempt(stack, run_id="a", task_id="t2", step=0, reward=0.0)
    await attempt(stack, run_id="b", task_id="t1", step=0, reward=1.0)

    job = await stack.run_export(run="a", format="graph")
    assert job["status"] == "ready"
    assert len(job["selected_trajectory_ids"]) == 2
    assert job["run"] == "a"


async def test_an_exported_row_says_which_run_and_step_produced_it(stack):
    """A training row that cannot say that cannot be compared with another
    step's."""
    created = await attempt(stack, run_id="a", task_id="t1", step=3, reward=1.0)
    rows = await stack.export_lines(trajectory=created["id"], format="graph")
    assert rows[0]["run_id"] == "a"
    assert rows[0]["task_id"] == "t1"
    assert rows[0]["step"] == 3


@pytest.mark.parametrize("scopes", [{}, {"project": "rl", "run": "a"}])
async def test_an_export_takes_exactly_one_scope(stack, scopes):
    await attempt(stack, run_id="a", task_id="t1", step=0)
    response = await stack.post("/v1/exports", {"format": "graph", **scopes})
    assert response.status_code == 400
    assert "exactly one of project, run or trajectory" in response.json()["detail"]


async def test_exporting_an_unknown_run_is_a_404(stack):
    assert (await stack.post("/v1/exports", {"format": "graph", "run": "nope"})).status_code == 404


async def test_a_reward_is_stored_as_written_whatever_it_is(stack_builder, tmp_path):
    """A reward is a free-form annotation. Nothing coerces it, so a listing
    off a record and a listing off the running service agree -- including on
    the cases that tempt a coercion: `True` is not 1.0, and a string reward is
    a string.
    """
    from skyrl_capture.reader.records import RecordReader, TrajectoryQuery

    root = tmp_path / "traces"
    stack = await stack_builder(record_dir=root)
    rewards = {"t1": 1e-07, "t2": 0.5, "t3": "great", "t4": -2, "t5": True}
    for task, reward in rewards.items():
        await attempt(stack, run_id="a", task_id=task, step=0, reward=reward)
    await stack.settle()

    reader = RecordReader(root)
    await reader.refresh()
    page = reader.list_trajectories(TrajectoryQuery(run_id="a", limit=50))
    stored = {row["task_id"]: row["annotations"].get("reward") for row in page.items}
    assert stored["t1"] == pytest.approx(1e-07), "scientific notation is still a number"
    assert stored["t2"] == 0.5 and stored["t4"] == -2
    assert stored["t3"] == "great"
    assert stored["t5"] is True

    served = {
        row["task_id"]: row["annotations"].get("reward")
        for row in (await stack.get("/v1/trajectories?run_id=a")).json()["data"]
    }
    assert served == stored, "one reader, so one answer"
