from loguru import logger 
import swebench
import tempfile

def evaluate_result(runtime, instance, run_results, instance_id, trajectory_id, dataset) -> bool:
        """Apply patch and evaluate the solution."""
        from swebench.harness.grading import get_eval_report
        from swebench.harness.constants import (
            APPLY_PATCH_FAIL,
            APPLY_PATCH_PASS,
        )
        from swebench.harness.test_spec import (
            make_test_spec,
        )
        
        model_patch = run_results.get('git_patch', None)
        if not model_patch:
            raise Exception(f"No git patch found for instance {instance_id}, trajectory {trajectory_id}")
        
        test_spec = make_test_spec(instance=instance)
        model_patch = process_git_patch(model_patch)
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

        # Apply patch
        if 'swe-smith' in dataset:
            # need to fetch and checkout the branch first
            exec_command = (
                "cd /testbed && "
                "git fetch && "
                f"git checkout {instance['instance_id']} && "
                "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
                "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "echo 'APPLY_PATCH_FAIL')))"
            )
        else:
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
        # instance['test_result']['apply_patch_output'] = apply_patch_output

        if 'APPLY_PATCH_FAIL' in apply_patch_output:
            raise Exception(f"Instance {instance_id}, trajectory {trajectory_id} {APPLY_PATCH_FAIL}:\n{apply_patch_output}")
        elif 'APPLY_PATCH_PASS' in apply_patch_output:
            logger.info(f'[{instance_id}, {trajectory_id}] {APPLY_PATCH_PASS}:\n{apply_patch_output}')

            # Run eval script in background and save output to log file
            log_file = '/tmp/eval_output.log'
            action = CmdRunAction(command=f'/tmp/eval.sh > {log_file} 2>&1 & echo $!')
            action.set_hard_timeout(300)  # Short timeout just to get the process ID
            obs = runtime.run_action(action)

            if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
                pid = obs.content.split()[-1].strip()
                logger.info(
                    f'[{instance_id}, {trajectory_id}] Evaluation process started with PID: {pid}'
                )

                # Poll for completion
                start_time = time.time()
                timeout = 1200  # 20 minutes
                while True:
                    seconds_elapsed = time.time() - start_time
                    if seconds_elapsed > timeout:
                        raise Exception(
                            f'[{instance_id}, {trajectory_id}] Evaluation timed out after {timeout} seconds'
                        )
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
                            f'[{instance_id}, {trajectory_id}] Evaluation process completed after {seconds_elapsed} seconds'
                        )
                        break
                    logger.info(
                        f'[{instance_id}, {trajectory_id}] [{seconds_elapsed:.0f}s] Evaluation still running, waiting...'
                    )
                    time.sleep(30)  # Wait for 30 seconds before checking again

                # Read the log file
                cat_action = CmdRunAction(command=f'cat {log_file}')
                cat_action.set_hard_timeout(300)
                cat_obs = runtime.run_action(cat_action)

                # Grade answer
                if isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0:
                    test_output = cat_obs.content
                    assert isinstance(test_output, str)
                    # instance['test_result']['test_output'] = test_output

                    # Get report from test output
                    logger.info(f'[{instance_id}, {trajectory_id}] Grading answer...')
                    
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
                            extra_kwargs = {}
                            if 'swe-smith' in dataset:
                                # SWE-Gym uses a different version of the package, hence a different eval report argument
                                extra_kwargs['test_log_path'] = test_output_path
                            else:
                                extra_kwargs['log_path'] = test_output_path
                            
                            if 'swe-smith' in dataset:
                                extra_kwargs['inst'] = instance
                            else:
                                extra_kwargs['test_spec'] = test_spec
                                extra_kwargs['include_tests_status'] = True
                            
                            _report = get_eval_report(
                                prediction={
                                    'model_patch': model_patch,
                                    'instance_id': instance_id,
                                },
                                **extra_kwargs,
                            )
                            # in swe-smith, the report is a single dict
                            # in swe-gym and swe-bench, the report is a dict with instance_id
                            report = _report if 'swe-smith' in dataset else _report[instance_id]
                            logger.info(
                                f"[{instance_id}, {trajectory_id}] report: {report}\nResult for [{instance_id}, {trajectory_id}]: resolved: {report['resolved']}"
                            )
                            return report['resolved']
                        except Exception as e:
                            logger.error(
                                f'[{instance_id}, {trajectory_id}] Error when getting eval report: {e}'
                            )
                            return False
            else:
                raise Exception(f'[{instance_id}, {trajectory_id}] Error when starting eval:\n{obs.content}')
        else:
            raise Exception(
                f'[{instance_id}] Unexpected output when applying patch:\n{apply_patch_output}'
            )