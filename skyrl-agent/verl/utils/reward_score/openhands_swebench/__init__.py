# Modified from https://github.com/All-Hands-AI/OpenHands/blob/main/evaluation/benchmarks/swe_bench/eval_infer.py
# commit around: 1a7003a7056f5b459207c2e2d2c6db4646967c53
# (DL): Deleted logic, in addition to dataset processing:
# 1. test spec processing, assume ground truth is already provided (reference: prime_code)
# 2. 
import json
import os
import tempfile
import time

from pathlib import Path
from typing import cast

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from swegym.harness.grading import get_eval_report
from swegym.harness.run_evaluation import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
)

from openhands.core.config import (
    AppConfig,
    SandboxConfig
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation
from openhands.utils.async_utils import call_async_from_sync


from swebench.harness.constants import (
    SWEbenchInstance,
    KEY_INSTANCE_ID
)


# TODO: migrate all swe-bench docker to ghcr.io/openhands
DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')

"""Mapping instance_id to resource_factor.
Different instances may have different resource requirements.
e.g., some instances may require more memory/CPU to run inference.
This file tracks the resource requirements of different instances.
"""

import json
import os

from openhands.core.logger import openhands_logger as logger

# OpenHands evaluation/benchmarks/swe_bench/resource/mapping.py
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNTIME_RESOURCE_FACTOR = int(
    os.environ.get('DEFAULT_RUNTIME_RESOURCE_FACTOR', 1)
)

"""
# Adapt from https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/utils.py
def load_swebench_dataset(
    name="princeton-nlp/SWE-bench", split="test", instance_ids=None
) -> list[SWEbenchInstance]:
    """"""
    Load SWE-bench dataset from Hugging Face Datasets or local .json/.jsonl file
    """"""
    # check that all instance IDs are in the dataset
    if instance_ids:
        instance_ids = set(instance_ids)
    # Load from local .json/.jsonl file
    if name.endswith(".json") or name.endswith(".jsonl"):
        dataset = json.loads(Path(name).read_text())
        dataset_ids = {instance[KEY_INSTANCE_ID] for instance in dataset}
    else:
        # Load from Hugging Face Datasets
        if name.lower() in {"swe-bench", "swebench", "swe_bench"}:
            name = "princeton-nlp/SWE-bench"
        elif name.lower() in {
            "swe-bench-lite",
            "swebench-lite",
            "swe_bench_lite",
            "swe-bench_lite",
            "lite",
        }:
            name = "princeton-nlp/SWE-bench_Lite"
        elif "swe-gym" in name.lower():
            name = "SWE-Gym/SWE-Gym"
        if (Path(name) / split / "dataset_info.json").exists():
            dataset = cast(Dataset, load_from_disk(Path(name) / split))
        else:
            dataset = cast(Dataset, load_dataset(name, split=split))
        dataset_ids = {instance[KEY_INSTANCE_ID] for instance in dataset}
    if instance_ids:
        if instance_ids - dataset_ids:
            raise ValueError(
                (
                    "Some instance IDs not found in dataset!"
                    f"\nMissing IDs:\n{' '.join(instance_ids - dataset_ids)}"
                )
            )
        dataset = [
            instance
            for instance in dataset
            if instance[KEY_INSTANCE_ID] in instance_ids
        ]
    return [cast(SWEbenchInstance, instance) for instance in dataset]
"""

# dataset to resource mapping
_global_resource_mapping: dict[str, dict[str, float]] = {}
def get_resource_mapping(dataset_name: str) -> dict[str, float]:
    if dataset_name not in _global_resource_mapping:
        file_path = os.path.join(CUR_DIR, f'{dataset_name}.json')
        if not os.path.exists(file_path):
            logger.warning(f'Resource mapping for {dataset_name} not found.')
            return None

        with open(file_path, 'r') as f:
            _global_resource_mapping[dataset_name] = json.load(f)
        logger.info(f'Loaded resource mapping for {dataset_name}')
    return _global_resource_mapping[dataset_name]

def get_instance_resource_factor(dataset_name: str, instance_id: str) -> int:
    resource_mapping = get_resource_mapping(dataset_name)
    if resource_mapping is None:
        return DEFAULT_RUNTIME_RESOURCE_FACTOR
    return int(resource_mapping.get(instance_id, DEFAULT_RUNTIME_RESOURCE_FACTOR))

def get_default_sandbox_config_for_eval() -> SandboxConfig:
    return SandboxConfig(
        use_host_network=False,
        # large enough timeout, since some testcases take very long to run
        timeout=300,
        api_key=os.environ.get('ALLHANDS_API_KEY', None),
        remote_runtime_api_url=os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL'),
        keep_runtime_alive=False,
        remote_runtime_init_timeout=3600,
        remote_runtime_api_timeout=120,
        remote_runtime_enable_retries=True,
        remote_runtime_class='sysbox',
    )

def get_instance_docker_image(instance_id: str) -> str:
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace(
        '__', '_s_'
    )  # to comply with docker image naming convention
    return (DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name).lower()

def process_git_patch(patch):
    if not isinstance(patch, str):
        return ''

    if not patch.strip():
        print(f'Skipping empty patch....')
        # skip empty patches
        return ''

    patch = patch.replace('\r\n', '\n')
    # There might be some weird characters at the beginning of the patch
    # due to some OpenHands inference command outputs

    # FOR EXAMPLE:
    # git diff --no-color --cached 895f28f9cbed817c00ab68770433170d83132d90
    # [A[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[K0
    # diff --git a/django/db/models/sql/.backup.query.py b/django/db/models/sql/.backup.query.py
    # new file mode 100644
    # index 0000000000..fc13db5948

    # We "find" the first line that starts with "diff" and then we remove lines before it
    lines = patch.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('diff --git'):
            patch = '\n'.join(lines[i:])
            break

    patch = patch.rstrip() + '\n'  # Make sure the last line ends with a newline
    return patch


def get_config(data_source, instance_id) -> AppConfig:
    # We use a different instance image for the each instance of swe-bench eval
    base_container_image = get_instance_docker_image(instance_id)
    logger.info(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
    )
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image
    sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
        dataset_name=data_source,
        instance_id=instance_id,
    )
    config = AppConfig(
        run_as_openhands=False,
        runtime="remote",
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    return config

# modified from the process_instance in eval_infer.py
def compute_score(
    model_patch,
    test_spec,
    instance_id,
    data_source,
    ):
    model_patch = process_git_patch(model_patch)
    config = get_config(data_source, instance_id)
    instance = {"instance_id": instance_id, "model_patch": model_patch, "test_spec": test_spec, "test_result": {}}

    instance['test_result']['report'] = {
        'empty_generation': False,
        'resolved': False,
        'failed_apply_patch': False,
        'error_eval': False,
        'test_timeout': False,
    }

    # print(f"(DL Debug in openhands swebench init) Inside compute score with model_patch: {model_patch} {model_patch == ""}")
    if model_patch == '':
        # print(f"(DL Debug in openhands swebench init) Inside compute score empty branch with instance {instance}")
        instance['test_result']['report']['empty_generation'] = True
        return instance['test_result']['report']['resolved'] # (DL): This is 0, but for compatibility, we return the resolved value

    try:
        runtime = create_runtime(config)
        call_async_from_sync(runtime.connect)
        # Get patch and save it to /tmp/patch.diff
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch file
            patch_file_path = os.path.join(temp_dir, 'patch.diff')
            with open(patch_file_path, 'w') as f:
                f.write(model_patch)
            runtime.copy_to(patch_file_path, '/tmp')
            # Eval script
            eval_script_path = os.path.join(temp_dir, 'eval.sh')
            with open(eval_script_path, 'w') as f:
                f.write(test_spec.eval_script)
            runtime.copy_to(eval_script_path, '/tmp')

        # Set +x
        action = CmdRunAction(command='chmod +x /tmp/eval.sh')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0

        print(f"(DL Debug in openhands swebench init) Inside compute score L253")
        # Apply patch
        exec_command = (
            'cd /testbed && '
            "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
            "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
            "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
            "echo 'APPLY_PATCH_FAIL')))"
        )
        action = CmdRunAction(command=exec_command)
        action.set_hard_timeout(600)
        obs = runtime.run_action(action)
        assert isinstance(obs, CmdOutputObservation)
        apply_patch_output = obs.content
        assert isinstance(apply_patch_output, str)
        instance['test_result']['apply_patch_output'] = apply_patch_output

        # print(f"(DL Debug in openhands swebench init) Inside compute score L269")
        if 'APPLY_PATCH_FAIL' in apply_patch_output:
            logger.info(f'[{instance_id}] {APPLY_PATCH_FAIL}:\n{apply_patch_output}')
            instance['test_result']['report']['failed_apply_patch'] = True

            return instance['test_result']['report']['resolved']
        elif 'APPLY_PATCH_PASS' in apply_patch_output:
            logger.info(f'[{instance_id}] {APPLY_PATCH_PASS}:\n{apply_patch_output}')

            # Run eval script in background and save output to log file
            log_file = '/tmp/eval_output.log'
            action = CmdRunAction(command=f'/tmp/eval.sh > {log_file} 2>&1 & echo $!')
            action.set_hard_timeout(300)  # Short timeout just to get the process ID
            obs = runtime.run_action(action)

            # print(f"(DL Debug in openhands swebench init) Inside compute score L284")
            if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
                pid = obs.content.split()[-1].strip()
                logger.info(
                    f'[{instance_id}] Evaluation process started with PID: {pid}'
                )

                # Poll for completion
                start_time = time.time()
                timeout = 1800  # 30 minutes
                while True:
                    seconds_elapsed = time.time() - start_time
                    if seconds_elapsed > timeout:
                        logger.info(
                            f'[{instance_id}] Evaluation timed out after {timeout} seconds'
                        )
                        instance['test_result']['report']['test_timeout'] = True
                        break
                    check_action = CmdRunAction(
                        command=f'ps -p {pid} > /dev/null; echo $?'
                    )
                    check_action.set_hard_timeout(300)
                    check_obs = runtime.run_action(check_action)
                    if (
                        isinstance(check_obs, CmdOutputObservation)
                        and check_obs.content.split()[-1].strip() == '1'
                    ):
                        logger.info(
                            f'[{instance_id}] Evaluation process completed after {seconds_elapsed} seconds'
                        )
                        break
                    logger.info(
                        f'[{instance_id}] [{seconds_elapsed:.0f}s] Evaluation still running, waiting...'
                    )
                    time.sleep(30)  # Wait for 30 seconds before checking again
                # print(f"(DL Debug in openhands swebench init) Inside compute score L320")

                # Read the log file
                cat_action = CmdRunAction(command=f'cat {log_file}')
                cat_action.set_hard_timeout(300)
                cat_obs = runtime.run_action(cat_action)

                # Grade answer
                if isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0:
                    test_output = cat_obs.content
                    assert isinstance(test_output, str)
                    instance['test_result']['test_output'] = test_output

                    # Get report from test output
                    logger.info(f'[{instance_id}] Grading answer...')
                    # print(f"(DL Debug in openhands swebench init) Inside compute score L335")
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Create a directory structure that matches the expected format
                        # NOTE: this is a hack to make the eval report format consistent
                        # with the original SWE-Bench eval script
                        log_dir = os.path.join(temp_dir, 'logs', instance_id.lower())
                        os.makedirs(log_dir, exist_ok=True)
                        test_output_path = os.path.join(log_dir, 'test_output.txt')
                        with open(test_output_path, 'w') as f:
                            f.write(test_output)
                        try:
                            _report = get_eval_report(
                                test_spec=test_spec,
                                prediction={
                                    'model_patch': model_patch,
                                    'instance_id': instance_id,
                                },
                                log_path=test_output_path,
                                include_tests_status=True,
                            )
                            report = _report[instance_id]
                            logger.info(
                                f"[{instance_id}] report: {report}\nResult for {instance_id}: resolved: {report['resolved']}"
                            )
                            instance['test_result']['report']['resolved'] = report[
                                'resolved'
                            ]
                        except Exception as e:
                            logger.error(
                                f'[{instance_id}] Error when getting eval report: {e}'
                            )
                            instance['test_result']['report']['resolved'] = False
                            instance['test_result']['report']['error_eval'] = True
                # print(f"(DL Debug in openhands swebench init) Inside compute score L367")
            else:
                logger.info(f'[{instance_id}] Error when starting eval:\n{obs.content}')
                instance['test_result']['report']['error_eval'] = True

            return instance['test_result']['report']['resolved']
        else:
            logger.info(
                f'[{instance_id}] Unexpected output when applying patch:\n{apply_patch_output}'
            )
            return instance['test_result']['report']['resolved']
            # (DL): This is not raised, we should always return a number for training
            # raise RuntimeError(
            #     instance_id,
            #     f'Unexpected output when applying patch:\n{apply_patch_output}',
            #     logger,
            # )
    finally:
        runtime.close()
