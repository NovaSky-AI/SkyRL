from swesmith.constants import (
    TEST_OUTPUT_END,
    TEST_OUTPUT_START,
)
from swesmith.harness.utils import get_test_command
from types import SimpleNamespace

def swe_smith_get_instance_docker_image(instance):
    image_name = instance['image_name']
    return f"jyangballin/{image_name.replace('__', '_1776_')}"

def make_test_spec(instance) -> str:
    env_name = "testbed"
    repo_directory = f"/{env_name}"
    test_command, _ = get_test_command(instance)
    eval_script = ("\n".join(
                [
                    "#!/bin/bash",
                    "set -uxo pipefail",
                    "source /opt/miniconda3/bin/activate",
                    f"conda activate {env_name}",
                    f"cd {repo_directory}",
                    f": '{TEST_OUTPUT_START}'",
                    test_command,
                    f": '{TEST_OUTPUT_END}'",
                ]
            )
            + "\n")
    return SimpleNamespace(eval_script=eval_script)