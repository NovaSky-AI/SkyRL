from email import message
from skyrl_agent.tasks.base import BaseTask
from typing import Dict, Any
import time
from skyrl_agent.tools.osworld_tools import OSWorldActionTool
import asyncio
import json
from loguru import logger
from skyrl_agent.tasks.osworld.desktop_env_interface import DesktopEnvInterface, DesktopEnvRay
from skyrl_agent.tasks.osworld.desktop_env.desktop_env import DesktopEnv
from typing import List
from io import BytesIO
import base64


SYS_PROMPT_IN_A11Y_OUT_CODE = """
You are an agent which follow my instruction and perform desktop computer tasks as instructed.
You have good knowledge of computer and good internet connection and assume your code will run on a computer for controlling the mouse and keyboard.
For each step, you will get an observation of the desktop by accessibility tree, which is based on AT-SPI library. And you will predict the action of the computer based on the accessibility tree.

You can use the osworld_action tool to perform the action grounded to the observation.

You are required to use `pyautogui` while using the osworld_action tool, but DO NOT use the `pyautogui.locateCenterOnScreen` function to locate the element you want to operate with since we have no image of the element you want to operate with. DO NOT USE `pyautogui.screenshot()` to make screenshot in the tool call.
Return one tool call to perform the action each time, be time efficient.
You need to to specify the coordinates by yourself based on current observation, but you should be careful to ensure that the coordinates are correct.

Specially, it is also allowed to call the finish tool to finish the task:
When you think the task can not be done, call the finish tool with answer="FAIL" in the format of
<function=finish>
<parameter=answer>FAIL</parameter>
</function> 
When you think the task is done, call the finish tool with answer="DONE" in the format of
<function=finish>
<parameter=answer>DONE</parameter>
</function>
DO NOT EASILY CALL THE FINISH TOOL, TRY YOUR BEST TO DO THE TASK;

My computer's password is 'password', feel free to use it when you need sudo rights.
First give the current screenshot and previous things we did a short reflection, then CALL THE TOOLS I ASKED FOR. NEVER EVER RETURN ME ANYTHING ELSE.
""".strip()

MAX_RETRY_TIMES = 10

class OSWorldTask(BaseTask):
    @classmethod
    async def initialize_runtime(cls, instance: Dict[str, Any]):
        runtime = instance.get('runtime')
        cfg = instance.pop('cfg')
        vision_is_active = cfg.vision_is_active
        instance['vision_is_active'] = vision_is_active
        
        start_time = time.time()
        
        last_exception = None
        for attempt in range(MAX_RETRY_TIMES):
            try:
                await runtime.reset(task_config=instance)
                break
            except Exception as e:
                last_exception = e
                print(f"Failed to reset the runtime, retrying... (attempt {attempt + 1}/{MAX_RETRY_TIMES})")
                print(f"Error: {str(e)}")
                if attempt < MAX_RETRY_TIMES - 1:  # Don't sleep after last attempt
                    await asyncio.sleep(1)
        else:
            # If we exit the loop without breaking, all retries failed
            print(f"All {MAX_RETRY_TIMES} retry attempts failed")
            raise last_exception
        
        await asyncio.sleep(5) # wait for the environment to be ready
        initial_obs = await runtime._get_obs_async()
        
        reset_time = time.time() - start_time
        print(f"[Runtime Reset] Reset completed in {reset_time:.2f} seconds")
        if vision_is_active:
            return cls.pil_to_base64(initial_obs["screenshot"])
        if initial_obs["accessibility_tree"] is None:
            raise ValueError("Accessibility tree is None")
        initial_acc_tree = OSWorldActionTool.linearize_accessibility_tree(initial_obs["accessibility_tree"], "ubuntu") # fixme: refer to the platform of the runtime/agent
        if initial_acc_tree:
            initial_acc_tree = OSWorldActionTool.trim_accessibility_tree(initial_acc_tree, 10000) # fixme: add arguments for max tokens
        cls.initial_acc_tree = initial_acc_tree
        return initial_acc_tree
    
    @classmethod
    async def get_instruction(cls, instance: Dict[str, Any]):
        instance = cls.osworld_data_preprocess(instance)
        initial_observation = await cls.initialize_runtime(instance)
        instance['initial_observation'] = initial_observation
        vision_is_active = instance.get('vision_is_active', False)

        key = "instruction" if "instruction" in instance else "prompt"

        assert 'initial_observation' in instance, "initial_observation is required"
        initial_observation = instance['initial_observation']
        
        if isinstance(instance[key], str):
            instruction = instance[key] # instance here should be json.load(example.json) examples from osworld
            system_message = SYS_PROMPT_IN_A11Y_OUT_CODE + "\nYou are asked to complete the following task: {}".format(instruction)
            messages = [
                {
                    "role": "system",
                    "content": system_message
                }
            ]
        elif isinstance(instance[key], list):
            messages = instance[key]
            instruction = messages[0]["content"]
            system_message_str = SYS_PROMPT_IN_A11Y_OUT_CODE + "\nYou are asked to complete the following task: {}".format(instruction)
            messages = [
                {
                    "role": "system",
                    "content": system_message_str
                }
            ]
        if vision_is_active:
            user_content = [
                {"type": "text", "text": "Here is the current desktop screenshot. What's the next step to help with the task?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{initial_observation}"}}
            ]
        else:
            user_content = "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(initial_observation)
        messages.append({
            "role": "user",
            "content":  user_content
            })
        return messages
    
    @classmethod
    def complete_runtime(cls, runtime: DesktopEnvRay, instance: Dict[str, Any]):
        runtime.close()
        
    @classmethod
    async def evaluate_result(cls, result: any, instance: any, data_source: str, instance_id: int, trajectory_id: int) -> float:
        """
        result is returned by the self.runtime.evaluate() in the osworld_react_agent.py
        """
        runtime = instance['runtime']
        if result != "DONE" and result != "FAIL":
            await runtime.step_async("DONE", 0.2)
        else:
            await runtime.step_async(result, 0.2)
        result = await runtime.evaluate()
        return result
    
    @classmethod
    def osworld_data_preprocess(cls, instance: Dict[str, Any]):
        json_columns = ['config', 'evaluator', 'related_apps']
        for col in json_columns:
            if col in instance:
                try:
                    value = instance[col]
                    if isinstance(value, str):
                        if value and value != "":
                            instance[col] = json.loads(value)
                        elif col == 'config' or col == 'evaluator' or col == 'source' or col == 'related_apps':
                            instance[col] = []
                        else:
                            instance[col] = None
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse JSON for column {col}: {e}")
                    logger.warning(f"Instance: {instance}")
                    instance[col] = []
        return instance
    
    @classmethod
    async def initialize_shared_env(cls, cfg: Dict[str, Any]):
        batch_size = 8
        total_agents = cfg.dispatcher.max_parallel_agents
        shared_env = []
        print("creating shared env")
        
        for i in range(0, total_agents, batch_size):
            # Calculate how many agents to create in this batch
            current_batch_size = min(batch_size, total_agents - i)
            
            # Create tasks for current batch
            desktop_env_tasks = [
                cls._create_and_start_desktop_env(i + j, cfg) 
                for j in range(current_batch_size)
            ]
            
            # Execute current batch concurrently
            batch_envs = await asyncio.gather(*desktop_env_tasks)
            shared_env.extend(batch_envs)
            
            logger.info(f"Created batch {i//batch_size + 1}: {current_batch_size} DesktopEnv instances (Total: {len(shared_env)}/{total_agents})")
        return shared_env
        
    @classmethod
    async def _create_and_start_desktop_env(cls, env_id: int, cfg: Dict[str, Any]) -> DesktopEnvInterface:
        """
        Async wrapper to create and start a DesktopEnv instance.
        Uses Ray for distributed execution if enabled.
        """
        print("creating and starting desktop env with id ", env_id)
        if cfg.generator.use_cpu_node:  # Add this flag to your config
            # Create Ray remote actor
            runtime = DesktopEnvRay.remote(
                action_space="pyautogui",
                provider_name="docker",
                screen_size=(1920, 1080),
                headless=True,
                os_type="Ubuntu",
                require_a11y_tree=not cfg.generator.vision_is_active,
                env_id=env_id,
                path_to_vm = cfg.generator.path_to_vm
            )
            print(runtime)
            # Wrap in interface first, then start the emulator
            interface = DesktopEnvInterface(runtime, cpu_node=True)
            await interface._start_emulator_async()
            
            return interface
        else:
            # Original local execution
            runtime = DesktopEnv(
                action_space="pyautogui",
                provider_name="docker",
                screen_size=(1920, 1080),
                headless=True,
                os_type="Ubuntu",
                require_a11y_tree=not cfg.generator.vision_is_active,
                env_id=env_id,
                path_to_vm = cfg.generator.path_to_vm
            )
            
            await runtime._start_emulator_async()
            
            return DesktopEnvInterface(runtime, cpu_node=False)
        
        
    @classmethod
    async def close_shared_env(cls, shared_env: List[DesktopEnvInterface]):
        if shared_env:
            # Close all environments in parallel
            close_tasks = []
            for env in shared_env:
                # Run each close operation in a thread pool to avoid blocking
                close_tasks.append(asyncio.to_thread(env.close))
            
            # Wait for all close operations to complete
            await asyncio.gather(*close_tasks, return_exceptions=True)
            
            # Clean up any remaining reserved ports after closing all environments
            from skyrl_agent.tasks.osworld.desktop_env.providers.docker.provider import DockerProvider
            DockerProvider.cleanup_all_reserved_ports()
            while shared_env:
                shared_env.pop()
                
    @classmethod
    def get_task_dependent_context_management(cls, agent, traj_config):
        assert getattr(traj_config.generator_cfg, "history_length", None) is not None, "history_length is required"
        history_length = int(traj_config.generator_cfg.history_length)
        assert history_length > 0, "history_length must be greater than 0"
        
        if (2 + history_length * 2) < len(agent.history.messages):
            new_messages = [agent.history.messages[0]] + agent.history.messages[-history_length * 2 - 1:]
            agent.history.initialize(new_messages)
            
    @classmethod
    def pil_to_base64(cls, image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")  # 你可以改成 "JPEG" 等格式
        return base64.b64encode(buffer.getvalue()).decode("utf-8")