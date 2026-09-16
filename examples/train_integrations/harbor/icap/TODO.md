# Review of the SkyRL-side icap integration

Three fixes from the 2026-09-15 Harbor run are in `harbor_generator.py`. They
are correct. This lists what was still wrong or missing; items 1-4 are now
fixed, item 5 is open.

## The three fixes (good)

**Trajectory names now carry a run id and the step.** `_session_id` used SkyRL's
`TrajectoryID` alone, which is unique only *within* a step -- instance 0,
repetition 0 comes round again every step and on every re-run -- while capture
keeps a caller-supplied name for the life of its database. Step 2 died on
`409 trajectory '0_1' already exists`. The SkyRL id stays the last segment, so
capture's session and SkyRL's are still legibly the same one.

**The trajectory key moved into `llm_kwargs`.** Terminus-2 takes `api_base` as
its own parameter but has no `api_key` one: it forwards `llm_kwargs` to the
LiteLLM constructor and swallows anything else. A key set beside `api_base` was
accepted, ignored, and the route answered 401 with nothing in the trial config
to explain it.

> Worth knowing this is *per agent*, not general. `_with_api_base` now encodes
> Terminus-2's convention. A different Harbor agent may take the key elsewhere,
> and the failure mode is a 401 that looks like a capture problem. The `TypeError`
> guard on a non-mapping `llm_kwargs` is good; a comment naming the agent this
> layout belongs to would be better.

## Still to do

### ✅ 1. One failed trial kills the whole batch

> **Fixed.** `_trial` never raises: it retries, then masks. `_attempt` holds one
> attempt on its own trajectory, and `generate` fills any empty slot with a
> masked outcome rather than letting `compose` hit `None`.

`generate()` runs trials in an `asyncio.TaskGroup`, which cancels every sibling
when any task raises. `_trial` catches broadly, but only *inside* the try: the
`create_trajectory` call above it and the `trajectory.finish` in the `finally`
are both outside. So a 409, a dropped connection, or a slow control plane during
creation takes down the entire step, not one rollout.

The proven integration does the opposite -- a failed trajectory is masked, and
its whole prompt group with it, conservatively. This one should match that:
catch around the trajectory lifecycle too and return a `TrialOutcome` with
`stop_reason="error"` and no rows, which `compose` already knows how to mask.

Related: `outcomes` is pre-filled with `None`, so anything that does slip
through surfaces as `AttributeError: 'NoneType' has no attribute 'rows'` inside
`compose`, which names the wrong culprit.

### ✅ 2. `max_retries` is accepted and never used

> **Fixed.** It is the attempt count now, and a retry gets a **fresh**
> trajectory -- capture's graph is append-only, so reusing the name would
> interleave two rollouts into one record. An agent timeout is masked without
> retrying, matching the sibling integration.

It is stored in `__init__` and read nowhere. The proven integration retries a
trial up to `MAX_NUM_RETRIES_PER_TRIAL` before giving up, which matters because
a Harbor trial failure is often environmental -- a sandbox that did not come up.
Either implement it or drop the parameter; a config knob that does nothing is
worse than its absence.

### ✅ 3. `_run_harbor` is still `NotImplementedError`

> **Fixed.** Moved in from `work_icap`, so the class in this tree can be run.

The working implementation is `WiredICapHarborGenerator` in
`work_icap/src/work_icap/entrypoints/run_icap.py`: create a `Trial`, run it, and
map Harbor's exception types onto `reward` / `stop_reason`. That is ~20 lines
and it belongs here, not in an experiment repo. Keeping the hole means the class
in this tree cannot actually be run.

### ✅ 4. The shim should disappear

> **Done, on the capture side.** `type: "skyrl"` is a registered upstream kind
> now, so `run_icap.py` points a target straight at the router and
> `tito_shim.py` is deleted.

`ICapExp.get_generator` starts `work_icap`'s `tito_shim` in a thread, because
capture's `tokens` target speaks a different wire than SkyRL's router. That is
being fixed on the capture side -- see `inference-capture/TODO.md`, Todo 10 --
by registering SkyRL's wire as an upstream kind. When that lands, a target is
`type: "skyrl"` pointed straight at `{router_url}`, and `serve_in_thread`, the
`SHIM_PORT` plumbing and `get_node_ip()` all go.

### ⬜ 5. In-process `CaptureService` is now possible

`ICapHarborGenerator` takes a `capture_service` and only uses `.base_url`, so it
works with either an in-process service or a URL to a separate `icap serve`.
Until now only the second was available to SkyRL, because capture capped
`transformers<5` and SkyRL pins `>=5.6.1`. That cap is lifted, so the in-process
hook the design note describes can finally be tried -- and the two-venv setup in
`work_icap/scripts/00_setup.sh` can collapse to one.
