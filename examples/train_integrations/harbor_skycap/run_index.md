# The run index

The Harbor integration knows more about a run than skycap does: which
trajectories made up each training step, and what became of them. It writes
that as a run index: each step's `step.json` in the W&B index artifact, and,
once pulled, `index/<phase>/step-<N>.json` beside the records. It is an
optional companion to skycap's records (their format is skycap's
[`docs/format.md`](../../../skycap/docs/format.md)). A reader must
work without it, from the records alone, and must not assume it lists every
record in the directory, or that every record it lists is there.

The index is one file per step and phase, `index/<phase>/step-<N>.json` in the
record directory, where `<phase>` is `train` or `eval` and `<N>` is the global
step, without zero-padding (`index/train/step-12.json`). It is plain JSON, not
compressed. Records are only the top-level `*.json.zst` files, so the `index/`
directory never reads as a trajectory.

```json
{"format_version": 1, "run": "9bp7pkra", "phase": "train", "step": 12,
 "rows": [{"id": "tr_ab12", "instance_id": "task-7", "repetition_id": 0, "attempt": 0,
           "status": "finished", "annotations": {"reward": 1.0},
           "superseded": false, "trained": true,
           "record": {"path": "/data/record/tr_ab12.json.zst",
                      "mirror": "s3://bucket/run-7/tr_ab12.json.zst",
                      "files": ["tr_ab12.tokens.zst", "tr_ab12.json.zst"]}}]}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `format_version` | int | `1`. As for the document, adding a field does not change it |
| `run` | string | the run's id (e.g. the W&B run id) |
| `phase` | string | `train` or `eval` |
| `step` | int | the global step |
| `rows` | array | one per trajectory attempt opened in this step and phase (below) |

A row:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | the trajectory id: its record is `{id}.json.zst` |
| `instance_id` | string | the task (prompt) the attempt ran |
| `repetition_id` | int | which of the instance's samples in the step |
| `attempt` | int | the attempt number for this `(instance_id, repetition_id)`, from 0; a retry is a new trajectory |
| `status` | string or null | skycap's status at finish; null when no finish was answered |
| `annotations` | object or null | what the trajectory was finished with (e.g. `reward`) |
| `superseded` | bool | a later attempt of the same `(instance_id, repetition_id)` exists in this step |
| `trained` | bool or null | the final attempt had trainable tokens in the step's batch. `false` for a superseded attempt and in `eval`; null when the trainer didn't say |
| `record` | object or null | skycap finish's `record` (format.md, "Where a trajectory's record is"): `host`, `path`, `mirror` and `files`; null when no record was written |

`record.files` may be absent in an index written before it existed; such a
record is still found by its `id`. A row's record may be missing from the
directory (a mirror that lost it, a record never copied): a reader shows the
row without it.
