# Example script to run simplecoder on swebench

import argparse
import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, TypedDict, Any

from datasets import load_dataset
import simplecoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DjangoPackage:

    @classmethod
    def init_cmd(cls, repo_dir) -> str:
        return f"""
if [ ! -d "$HOME/{repo_dir}/.venv" ]; then
    pushd "$HOME/{repo_dir}"
        uv venv --python 3.11 && uv pip install -e .
    popd
fi
source $HOME/{repo_dir}/.venv/bin/activate
"""

    def __init__(self, executor):
        self.executor = executor

    def apply_patches(self, repo_dir):

        # There are certain older versions of django that use pyproject.toml which uv
        # cannot handle, in those cases we just delete it.
        pyproject_path = os.path.join(repo_dir, "pyproject.toml")
        if os.path.exists(pyproject_path):
            with open(pyproject_path, "r") as f:
                content = f.read()
                if "[project]" not in content:
                    os.remove(pyproject_path)

        # There is a test that often fails with "filename too long", so we deactivate it
        test_path = os.path.join(repo_dir, "tests/file_storage/tests.py")
        with open(test_path, "r") as f:
            new_contents = f.read().replace("def test_extended_length_storage(self):", "def _test_extended_length_storage(self):")
        with open(test_path, "w") as f:
            f.write(new_contents)

        # This test would fail with UnicodeEncodeError: 'utf-8' codec can't encode characters in position 12-13: surrogates not allowed
        test_path = os.path.join(repo_dir, "tests/mail/tests.py")
        with open(test_path, "r") as f:
            new_contents = f.read().replace("def test_send_unicode(self):", "def _test_send_unicode(self):")
            new_contents = new_contents.replace("def test_dont_base64_encode(self):", "def _test_dont_base64_encode(self):")
        with open(test_path, "w") as f:
            f.write(new_contents)
    
    def run_tests(self, repo_dir):
        # For now just run this in a uv venv to be fully compatible with setup.py, but this can be customized based
        # on the package
        return self.executor.execute(
            "uv run ./tests/runtests.py",
            working_dir=os.path.join("/home/skyrl", repo_dir),
        )


class SweBenchInstance(TypedDict):
    repo: str
    base_commit: str
    patch: str
    problem_statement: str


class SweBenchResult(TypedDict):
    patch: str


class SweBenchRunner:

    def __init__(self, model):
        self.model = model
        # Patches to apply to repos to make their tests pass
        self.packages = {
            "django/django": DjangoPackage,
        }

    def _get_task(self, instance: SweBenchInstance) -> str:
        repo_name = instance["repo"]
        problem_statement = instance["problem_statement"]
        hints_text = instance.get("hints_text", "")

        # TODO: Incorporate the other parts?

        return problem_statement

    def _git_checkout(self, org, repo, base_commit, working_dir):
        repo_dir = f"{org}__{repo}"
        subprocess.run(
            f"mkdir -p {repo_dir} && cd {repo_dir} && "
            f"git init && git remote add origin https://github.com/{org}/{repo} && "
            f"git fetch --depth 1 origin {base_commit} && git checkout FETCH_HEAD && cd ..",
            shell=True,
            cwd=working_dir,
            check=True,
        )
        return repo_dir

    def run_instance(self, instance: SweBenchInstance):

        base_commit = instance["base_commit"]
        org, repo = instance["repo"].split("/")
        
        with tempfile.TemporaryDirectory(delete=False) as working_dir:
            logging.info(f"running agent in working directory {working_dir}")
            repo_dir = self._git_checkout(org, repo, base_commit, working_dir)
            package_cls = self.packages[instance["repo"]]
            executor = simplecoder.BubblewrapExecutor(working_dir, init_cmd=package_cls.init_cmd(repo_dir))
            coder = simplecoder.SimpleCoder(os.environ["OPENAI_API_KEY"], self.model, executor)

            task = self._get_task(instance)
            coder.run(task)

            # Extract the patch
            result = subprocess.run(
                f"cd {repo_dir} && git diff && cd ..",
                shell=True,
                capture_output=True,
                text=True,
                cwd=working_dir,
                check=True,
            )

            return {"patch": result.stdout}

    def run_tests(self, instance: SweBenchInstance, patches: List[str]):
        base_commit = instance["base_commit"]
        org, repo = instance["repo"].split("/")

        with tempfile.TemporaryDirectory(delete=False) as home_dir:
            logging.info(f"running tests with home directory {home_dir}")
            repo_dir = self._git_checkout(org, repo, base_commit, home_dir)

            # Apply the patches
            for patch in patches:
                with tempfile.NamedTemporaryFile(delete=False, mode="w", prefix="agent-test-patch-", suffix=".patch") as f:
                    f.write(patch)
                logging.info(f"applying patch from {f.name} to {home_dir}/{repo_dir}")
                subprocess.run(
                    ["git", "apply", f.name],
                    cwd=os.path.join(home_dir, repo_dir),
                    check=True
                )

            package_cls = self.packages[instance["repo"]]
            executor = simplecoder.BubblewrapExecutor(home_dir, init_cmd=package_cls.init_cmd(repo_dir))
            package = package_cls(executor)
            package.apply_patches(os.path.join(home_dir, repo_dir))
            return package.run_tests(repo_dir)

    def run_dataset(self, dataset):
        pass

def main():
    parser = argparse.ArgumentParser(description="Solve SWE-bench challenges")

    parser.add_argument("--model", default="o4-mini", help="Model to use (default: o4-mini)")
    parser.add_argument("--max-iterations", type=int, default=30, 
                       help="Maximum iterations per instance (default: 30)")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                       help="Dataset to use (default: princeton-nlp/SWE-bench_Lite)")

    args = parser.parse_args()
    runner = SweBenchRunner("o4-mini")
    dataset = load_dataset(args.dataset)
    # TODO: Right now we only run a single instance, run all of them going forward
    instance = dataset["test"][100]
    print("instance", instance)
    run_result = runner.run_instance(instance)
    print("run_result", run_result)
    # test_result = runner.run_tests(instance, run_result["patch"])
    patches = [instance["test_patch"], run_result["patch"]]
    test_result = runner.run_tests(instance, patches)
    print("test_result", test_result)

    # for instance in dataset["test"]:
    #     if instance["repo"] == "django/django":
    #         r = runner.run_tests(instance, None)
    #         print("r", r)

if __name__ == "__main__":
    main()