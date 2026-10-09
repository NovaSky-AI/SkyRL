"""SWE-bench Multimodal as Harbor tasks, the issue's screenshots inline as images.

Each instance becomes a Harbor task directory:

- ``instruction.md``: the issue, each screenshot replaced by a mini-swe-agent image tag
  (``model.multimodal_regex``) pointing at a local copy of the image. skycap renders prompts on the
  training cluster, so the copies must be at the same path on every node that runs a skycap server
  (``--image-dir``, default ``~/data/swebench_multimodal/images``).
- ``task.toml``: the instance's prebuilt SWE-bench image (``swebench/sweb.eval.x86_64.*``), no build.
- ``tests/``: swebench's own eval script for the instance, and ``grade.py``, which scores its log.

Only the dev split has its tests (the test split's are hidden behind sb-cli). By default only the repos
swebench grades pass-and-fail are kept: in Chart.js, p5.js and marked a test counts as passing unless the
log says it failed, so a rollout that breaks the build or kills the test runner scores as solved.

    uv run --isolated --with "swebench==4.1.0" --with datasets --with pillow \\
        examples/train_integrations/harbor_skycap/swebench_multimodal/prepare_tasks.py \\
        --output-dir ~/data/swebench_multimodal/tasks
"""

import argparse
import hashlib
import io
import json
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from PIL import Image
from swebench.harness.constants import FAIL_ONLY_REPOS
from swebench.harness.test_spec.test_spec import make_test_spec

HERE = Path(__file__).resolve().parent
DATASET = "princeton-nlp/SWE-bench_Multimodal"
#: mini-swe-agent's default ``multimodal_regex`` matches this; the agent config turns it on.
IMAGE_TAG = "<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>{url}</MSWEA_MULTIMODAL_CONTENT>"

INSTRUCTION = """\
<issue>
{problem_statement}
</issue>

The repository is checked out at /testbed. Edit its source files to resolve the issue.
Do not modify the existing tests.
The sandbox has {cpus} CPU(s) and {memory_gb} GB of memory, but reports the host's CPUs: run Jest with
--maxWorkers=2 (or --runInBand), or its workers run out of memory and your command is killed.
"""

TASK_TOML = """\
schema_version = "1.0"

[task]
name = "swebench-multimodal/{instance_id}"
description = "SWE-bench Multimodal: {repo}"
keywords = ["swe-bench", "multimodal"]

[agent]
network_mode = "public"
timeout_sec = {agent_timeout}

[verifier]
network_mode = "public"
timeout_sec = {verifier_timeout}

[environment]
docker_image = "docker.io/{image}"
cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = {storage_mb}
gpus = 0
"""

TEST_SH = """\
#!/bin/bash
# swebench's eval script for this instance, then its log graded the swebench way (grade.py).
set -uo pipefail
mkdir -p /logs/verifier
bash /tests/eval.sh > /logs/verifier/test_output.log 2>&1
if ! command -v uv > /dev/null; then
  curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh > /dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
if uv run --no-project --python 3.12 --with "swebench==4.1.0" python /tests/grade.py \\
    /tests/config.json /logs/verifier/test_output.log /logs/verifier/report.json; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

SOLVE_SH = """\
#!/bin/bash
set -euo pipefail
cd /testbed
git apply - <<'__SOLUTION__'
{patch}__SOLUTION__
"""


#: Instances whose gold patch doesn't pass their own FAIL_TO_PASS tests in a Daytona sandbox: they can't be solved.
UNSOLVABLE = frozenset({"Automattic__wp-calypso-21635", "diegomura__react-pdf-1552"})

#: Jest starts a worker per CPU it sees, and a Daytona sandbox shows the host's (64): a whole suite in a
#: 4 GiB sandbox loses its workers ("Call retries were exceeded") and tests that pass at the base commit fail.
JEST = "./node_modules/.bin/jest"


def cap_test_workers(eval_script: str, workers: int) -> str:
    """``eval_script`` with every Jest run capped at ``workers`` workers. Raises if it runs no Jest."""
    if JEST not in eval_script:
        raise ValueError(f"the eval script runs no {JEST}; cap its test runner's workers here")
    return eval_script.replace(JEST, f"{JEST} --maxWorkers={workers}")


def image_urls(row: Dict) -> List[str]:
    return json.loads(row["image_assets"] or "{}").get("problem_statement", [])


def download(url: str, out: Path) -> bool:
    """The image at ``url`` as a PNG at ``out``. False if it can't be fetched or decoded (e.g. an SVG)."""
    if out.exists():
        return True
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - the dataset's own asset URLs
            image = Image.open(io.BytesIO(response.read()))
            image.load()
    except Exception as error:  # noqa: BLE001 - one bad asset skips that image, not the task
        print(f"  skipping {url}: {type(error).__name__}: {error}")
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out, "PNG")
    return True


def inline_images(text: str, local: Dict[str, str]) -> str:
    """``text`` with each downloaded image (markdown, ``<img>``, or a bare URL) replaced by its tag."""
    for url, path in local.items():
        tag = IMAGE_TAG.format(url=f"file://{path}")
        escaped = re.escape(url)
        text, n = re.subn(rf"!\[[^\]]*\]\(\s*{escaped}\s*\)", tag, text)
        if not n:
            text, n = re.subn(rf"<img\b[^>]*?src=[\"']{escaped}[\"'][^>]*>", tag, text)
        if not n:
            text = text.replace(url, tag)
    return text


def build(row: Dict, out_dir: Path, image_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    instance_id = row["instance_id"]
    local = {}
    for index, url in enumerate(image_urls(row)):
        digest = hashlib.sha256(url.encode()).hexdigest()[:12]
        path = image_dir / instance_id / f"{index}_{digest}.png"
        if download(url, path):
            local[url] = str(path)
    if not local:
        print(f"{instance_id}: no usable images, skipped")
        return None

    spec = make_test_spec(row, namespace="swebench")
    task = out_dir / instance_id
    if task.exists():
        shutil.rmtree(task)
    (task / "tests").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "environment").mkdir()

    problem = inline_images(row["problem_statement"].strip(), local)
    (task / "instruction.md").write_text(
        INSTRUCTION.format(problem_statement=problem, cpus=args.cpus, memory_gb=args.memory_mb // 1024)
    )
    (task / "task.toml").write_text(
        TASK_TOML.format(
            instance_id=instance_id,
            repo=row["repo"],
            image=spec.instance_image_key,
            agent_timeout=args.agent_timeout,
            verifier_timeout=args.verifier_timeout,
            cpus=args.cpus,
            memory_mb=args.memory_mb,
            storage_mb=args.storage_mb,
        )
    )
    (task / "tests" / "config.json").write_text(json.dumps(row, indent=2))
    (task / "tests" / "eval.sh").write_text(cap_test_workers(spec.eval_script, args.test_workers))
    shutil.copy(HERE / "grade.py", task / "tests" / "grade.py")
    (task / "tests" / "test.sh").write_text(TEST_SH)
    (task / "tests" / "test.sh").chmod(0o755)
    (task / "solution" / "solve.sh").write_text(SOLVE_SH.format(patch=row["patch"].rstrip("\n") + "\n"))
    (task / "solution" / "solve.sh").chmod(0o755)
    print(f"{instance_id}: {len(local)} image(s)")
    return task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=Path("~/data/swebench_multimodal/images"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--instance-ids", nargs="+", default=None, help="Only these instances")
    parser.add_argument("--repos", nargs="+", default=None, help="Only these repos, e.g. diegomura/react-pdf")
    parser.add_argument(
        "--include-fail-only-repos",
        action="store_true",
        help="Also keep Chart.js, p5.js and marked, whose grading can't tell a crashed test run from a pass",
    )
    parser.add_argument("--agent-timeout", type=int, default=3000)
    parser.add_argument("--verifier-timeout", type=int, default=1800)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=4096)
    parser.add_argument("--storage-mb", type=int, default=10240)
    parser.add_argument("--test-workers", type=int, default=2, help="Jest workers in the verifier's test run")
    args = parser.parse_args()

    out_dir = args.output_dir.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_dataset(DATASET, split=args.split)
    built = 0
    for row in rows:
        if args.instance_ids and row["instance_id"] not in args.instance_ids:
            continue
        if args.repos and row["repo"] not in args.repos:
            continue
        if row["repo"] in FAIL_ONLY_REPOS and not args.include_fail_only_repos:
            continue
        if row["instance_id"] in UNSOLVABLE:
            continue
        built += build(row, out_dir, image_dir, args) is not None
    print(f"{built} tasks in {out_dir}, images in {image_dir}")


if __name__ == "__main__":
    main()
