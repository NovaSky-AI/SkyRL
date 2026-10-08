"""Pulling a run's records and step index back from W&B: a fake public API, no network."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.train_integrations.harbor_skycap import record_index
from examples.train_integrations.harbor_skycap.record_index import (
    build_version,
    main,
    parse_ref,
    pull,
)

pytestmark = pytest.mark.integrations

MIRROR = "s3://bucket/run-7"
REF = "team/proj/skycap-records-train-run-1"


def row(trajectory_id, *, files=("tokens", "json"), mirror=True, attempt=0):
    names = [f"{trajectory_id}.{kind}.zst" for kind in files]
    document = f"{trajectory_id}.json.zst"
    return {
        "id": trajectory_id,
        "instance_id": f"task-{trajectory_id}",
        "repetition_id": 0,
        "attempt": attempt,
        "status": "finished",
        "annotations": {"reward": 1.0},
        "superseded": False,
        "trained": True,
        "record": {
            "host": "10.0.0.5",
            "path": f"/data/record/{document}",
            "mirror": f"{MIRROR}/{document}" if mirror else None,
            "files": names,
        },
    }


class FakeEntry:
    def __init__(self, artifact, name) -> None:
        self.artifact, self.name = artifact, name

    def download(self, root=None):
        if self.name in self.artifact.unreachable:
            raise OSError(f"cannot fetch {self.name}")
        target = Path(root) / self.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.artifact.contents[self.name])
        return str(target)


class FakeArtifact:
    """A logged version as the public API hands it back: step.json and a reference per record file.

    ``unreachable`` names entries W&B can't fetch; ``broken`` makes the whole-artifact download raise.
    """

    def __init__(self, version, phase, step, rows, unreachable=(), broken=False) -> None:
        built = build_version("run-1", phase, step, rows)
        self.version = version
        self.aliases = built.aliases
        self.contents = {"step.json": built.index}
        for uri, name in built.references:
            self.contents[name] = f"bytes of {uri}".encode()
        self.unreachable, self.broken = set(unreachable), broken
        self.downloads = 0

    def download(self, root=None, allow_missing_references=False):
        self.downloads += 1
        if self.broken:
            raise RuntimeError("the download failed")
        for name in self.contents:
            if name in self.unreachable:
                if not allow_missing_references:
                    raise OSError(f"cannot fetch {name}")
                continue
            FakeEntry(self, name).download(root)
        return root

    def get_entry(self, name):
        if name not in self.contents:
            raise KeyError(name)
        return FakeEntry(self, name)


class FakeApi:
    def __init__(self, *artifacts) -> None:
        self.versions = list(artifacts)
        self.calls = []

    def artifact(self, name):
        self.calls.append(("artifact", name))
        path, alias = name.rsplit(":", 1)
        for artifact in self.versions:
            if alias == artifact.version or alias in artifact.aliases:
                return artifact
        raise ValueError(f"no artifact {name}")

    def artifacts(self, type_name, name):
        self.calls.append(("artifacts", type_name, name))
        return iter(self.versions)


def test_an_alias_pulls_that_version_as_a_record_directory(tmp_path: Path) -> None:
    rows = [row("tr_a"), row("tr_b", files=("tokens", "experts", "json")), row("tr_c", mirror=False)]
    api = FakeApi(FakeArtifact("v0", "train", 2, []), FakeArtifact("v1", "train", 3, rows))
    summary = pull(f"{REF}:train-step-3", tmp_path / "out", api=api)

    assert api.calls == [("artifact", f"{REF}:train-step-3")]
    out = tmp_path / "out"
    names = ["tr_a.tokens.zst", "tr_a.json.zst", "tr_b.tokens.zst", "tr_b.experts.zst", "tr_b.json.zst"]
    assert sorted(path.name for path in out.iterdir() if path.is_file()) == sorted(names)
    assert (out / "tr_b.experts.zst").read_bytes() == f"bytes of {MIRROR}/tr_b.experts.zst".encode()
    # step.json is the run index, at index/<phase>/step-<N>.json.
    step = json.loads((out / "index" / "train" / "step-3.json").read_bytes())
    assert (step["format_version"], step["run"], step["phase"], step["step"]) == (1, "run-1", "train", 3)
    assert [r["id"] for r in step["rows"]] == ["tr_a", "tr_b", "tr_c"]
    # A local-only record is indexed, with nothing to pull.
    assert summary.versions == ["skycap-records-train-run-1:v1"]
    assert (summary.records, summary.files, summary.unchanged) == (2, 5, 0)
    assert summary.local_only == ["skycap-records-train-run-1:v1/tr_c at 10.0.0.5:/data/record/tr_c.json.zst"]
    assert summary.missing == []
    # Nothing of the scratch directory is left.
    assert sorted(path.name for path in out.iterdir() if path.is_dir()) == ["index"]


def test_no_alias_pulls_every_version_and_leaves_identical_files_alone(tmp_path: Path) -> None:
    train = FakeArtifact("v0", "train", 1, [row("tr_a")])
    # The same record again in a later step: already there, and identical.
    again = FakeArtifact("v1", "train", 2, [row("tr_a", attempt=0), row("tr_d")])
    evaluation = FakeArtifact("v2", "eval", 2, [row("tr_e")])
    out = tmp_path / "out"
    summary = pull(REF, out, api=FakeApi(train, again, evaluation))

    assert summary.versions == [f"skycap-records-train-run-1:{v}" for v in ("v0", "v1", "v2")]
    assert sorted(str(path.relative_to(out)) for path in (out / "index").rglob("*.json")) == [
        "index/eval/step-2.json",
        "index/train/step-1.json",
        "index/train/step-2.json",
    ]
    assert (summary.records, summary.files, summary.unchanged) == (4, 6, 2)
    # A second pull changes nothing.
    before = {path: path.stat().st_mtime_ns for path in out.rglob("*") if path.is_file()}
    again_summary = pull(REF, out, api=FakeApi(train, again, evaluation))
    assert again_summary.files == 0 and again_summary.unchanged == 8
    assert {path: path.stat().st_mtime_ns for path in out.rglob("*") if path.is_file()} == before


def test_a_record_that_cannot_be_fetched_is_skipped_whole_and_the_rest_go_on(tmp_path: Path) -> None:
    rows = [row("tr_a"), row("tr_b")]
    artifact = FakeArtifact("v0", "train", 3, rows, unreachable={"records/tr_b.tokens.zst"})
    out = tmp_path / "out"
    summary = pull(REF, out, api=FakeApi(artifact))
    # tr_b's document was fetched, but without its sidecar it isn't moved in.
    assert sorted(path.name for path in out.glob("*.zst")) == ["tr_a.json.zst", "tr_a.tokens.zst"]
    assert summary.records == 1
    assert summary.missing == ["skycap-records-train-run-1:v0/tr_b: could not fetch tr_b.tokens.zst"]
    assert (out / "index" / "train" / "step-3.json").exists()
    assert "1 missing" in summary.report() and "tr_b.tokens.zst" in summary.report()


def test_a_failed_download_falls_back_to_one_entry_at_a_time(tmp_path: Path) -> None:
    artifact = FakeArtifact("v0", "train", 3, [row("tr_a"), row("tr_b")], broken=True)
    artifact.unreachable = {"records/tr_a.json.zst"}
    summary = pull(REF, tmp_path, api=FakeApi(artifact))
    assert artifact.downloads == 1
    assert sorted(path.name for path in tmp_path.glob("*.zst")) == ["tr_b.json.zst", "tr_b.tokens.zst"]
    assert summary.records == 1 and len(summary.missing) == 1


def test_a_bad_version_is_reported_and_the_others_still_pull(tmp_path: Path) -> None:
    bad = FakeArtifact("v0", "train", 1, [row("tr_a")], unreachable={"step.json"})
    good = FakeArtifact("v1", "train", 2, [row("tr_b")])
    summary = pull(REF, tmp_path, api=FakeApi(bad, good))
    assert summary.versions == ["skycap-records-train-run-1:v1"]
    assert len(summary.failed_versions) == 1 and summary.failed_versions[0].startswith("skycap-records-train-run-1:v0")


def test_the_command_exits_non_zero_only_when_nothing_was_pulled(tmp_path: Path, capsys) -> None:
    good = FakeArtifact("v0", "train", 1, [row("tr_a")])
    assert main(["pull", REF, str(tmp_path / "out")], api=FakeApi(good)) == 0
    assert "versions: 1 pulled, 0 failed" in capsys.readouterr().out
    # Only local-only records: the index is still pulled.
    local = FakeArtifact("v0", "train", 1, [row("tr_c", mirror=False)])
    assert main(["pull", REF, str(tmp_path / "local")], api=FakeApi(local)) == 0
    # No such alias: nothing pulled.
    assert main(["pull", f"{REF}:nope", str(tmp_path / "none")], api=FakeApi(good)) == 1
    assert "1 failed" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["pull", "proj/artifact", str(tmp_path)], api=FakeApi())


def test_parse_ref() -> None:
    assert parse_ref("e/p/a") == ("e/p/a", None)
    assert parse_ref("e/p/a:train-step-3") == ("e/p/a", "train-step-3")
    for bad in ("p/a", "e/p/a/b", "e//a:v1"):
        with pytest.raises(ValueError):
            parse_ref(bad)


def test_wandb_is_imported_only_when_no_api_is_given(tmp_path: Path, monkeypatch) -> None:
    made = []
    fake = SimpleNamespace(Api=lambda: made.append(1) or FakeApi(FakeArtifact("v0", "train", 1, [row("tr_a")])))
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
    assert record_index.pull(REF, tmp_path).records == 1 and made == [1]


def test_a_file_name_that_isnt_a_plain_name_is_refused(tmp_path: Path) -> None:
    """step.json comes from the artifact: a name like ../victim.txt must not move a file out of out_dir."""
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"original")
    rows = [row("tr_a"), row("tr_evil", files=("json",))]
    rows[1]["record"]["files"] = ["../victim.txt"]
    summary = pull(f"{REF}:v0", tmp_path / "out", api=FakeApi(FakeArtifact("v0", "train", 1, rows)))

    assert victim.read_bytes() == b"original"
    assert summary.records == 1 and len(summary.missing) == 1 and "not pulled" in summary.missing[0]
    assert (tmp_path / "out" / "tr_a.json.zst").is_file()


def test_a_record_that_cant_move_in_doesnt_stop_the_others(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / "tr_a.json.zst").mkdir(parents=True)  # a directory where tr_a's document goes
    rows = [row("tr_a"), row("tr_b")]
    summary = pull(f"{REF}:v0", out, api=FakeApi(FakeArtifact("v0", "train", 1, rows)))

    assert summary.records == 1 and len(summary.missing) == 1 and "tr_a" in summary.missing[0]
    # tr_a's sidecar, moved in before its document failed, is taken back out; tr_b and the index are in.
    assert not (out / "tr_a.tokens.zst").exists()
    assert (out / "tr_b.tokens.zst").is_file() and (out / "tr_b.json.zst").is_file()
    assert (out / "index" / "train" / "step-1.json").is_file()
