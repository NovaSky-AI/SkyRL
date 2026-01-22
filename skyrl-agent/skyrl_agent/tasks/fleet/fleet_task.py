"""
Fleet Task - Task implementation for Fleet-hosted environments.

This module provides a task class for running agents on Fleet environments
with MCP tool support and verifier-based evaluation.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from skyrl_agent.tasks.base import BaseTask

logger = logging.getLogger(__name__)


class FleetTask(BaseTask):
    """Task implementation for Fleet-hosted environments.

    This task:
    1. Creates a Fleet environment instance via FleetTaskEnv
    2. Provides the task prompt as instruction
    3. Exposes MCP tools for agent interaction
    4. Evaluates results using the task verifier
    """

    @classmethod
    async def initialize_runtime(
        cls,
        instance: Dict[str, Any],
        api_key: Optional[str] = None,
        ttl_seconds: int = 600,
        max_iterations: int = 50,
        **kwargs,
    ) -> Dict[str, Any]:
        """Initialize the Fleet environment runtime.

        Args:
            instance: Task instance dict with keys:
                - task_key: Unique task identifier
                - prompt: Task instruction
                - env_key: Environment key (e.g., "booking")
                - env_version: Environment version
                - data_key: Optional data key
                - data_version: Optional data version
                - verifier_code: Python code for verification
            api_key: Fleet API key (defaults to FLEET_API_KEY env var)
            ttl_seconds: Instance TTL in seconds
            max_iterations: Max agent iterations

        Returns:
            Runtime dict with:
                - env: FleetTaskEnv instance
                - tools: FleetMCPTools instance
                - task_config: Original task config
        """
        try:
            from envs.fleet_env import FleetTaskEnv
        except ImportError as e:
            raise ImportError(
                "Fleet support requires OpenEnv. Install with:\n"
                "pip install git+https://github.com/fleet-ai/OpenEnv.git@deniz/fleet_client"
            ) from e

        api_key = api_key or os.environ.get("FLEET_API_KEY")
        if not api_key:
            raise ValueError("Fleet API key required (pass api_key or set FLEET_API_KEY)")

        # Build task config from instance
        task_config = {
            "task_key": instance.get("task_key", instance.get("key", "unknown")),
            "prompt": instance.get("prompt", ""),
            "env_key": instance.get("env_key", instance.get("env_id", "")),
            "env_version": instance.get("env_version", instance.get("version", "")),
            "data_key": instance.get("data_key", instance.get("data_id")),
            "data_version": instance.get("data_version"),
            "verifier_code": instance.get("verifier_code", instance.get("verifier_func", "")),
            "task_modality": instance.get("task_modality", "tool_use"),
        }

        logger.info(f"Initializing Fleet environment for task: {task_config['task_key']}")

        # Create FleetTaskEnv
        env = FleetTaskEnv(
            task_config=task_config,
            api_key=api_key,
            ttl_seconds=ttl_seconds,
            max_steps=max_iterations,
        )

        # Reset to create the environment and get tools
        obs = await env.reset_async()
        tools = obs.get("tools", [])

        logger.info(f"Fleet environment initialized with {len(tools)} tools")

        return {
            "env": env,
            "tools": tools,
            "task_config": task_config,
            "observation": obs,
        }

    @classmethod
    def get_instruction(cls, instance: Dict[str, Any], **kwargs) -> List[Dict[str, str]]:
        """Get the initial instruction for the agent.

        Args:
            instance: Task instance dict with 'prompt' key

        Returns:
            List of messages in OpenAI format
        """
        prompt = instance.get("prompt", "")
        task_key = instance.get("task_key", instance.get("key", "unknown"))

        instruction = f"""You are an AI assistant helping with a task.

Task ID: {task_key}

{prompt}

Use the available tools to complete this task. When you have completed the task, use the appropriate tool to indicate completion.
"""

        return [{"role": "user", "content": instruction}]

    @classmethod
    def complete_runtime(
        cls,
        runtime: Dict[str, Any],
        instance: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Complete/finalize the runtime.

        Args:
            runtime: Runtime dict from initialize_runtime
            instance: Task instance

        Returns:
            Dict with completion info
        """
        env = runtime.get("env")
        if env:
            try:
                env.close()
                logger.info(f"Closed Fleet environment for task: {instance.get('task_key', 'unknown')}")
            except Exception as e:
                logger.warning(f"Error closing Fleet environment: {e}")

        return {"status": "completed"}

    @classmethod
    async def evaluate_result(
        cls,
        instance: Dict[str, Any],
        runtime: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        **kwargs,
    ) -> bool:
        """Evaluate the agent's result using the task verifier.

        Args:
            instance: Task instance
            runtime: Runtime dict with env and tools
            trajectory: Agent trajectory (conversation history)

        Returns:
            True if task completed successfully, False otherwise
        """
        env = runtime.get("env")
        if not env:
            logger.warning("No environment in runtime, cannot evaluate")
            return False

        task_config = runtime.get("task_config", {})
        verifier_code = task_config.get("verifier_code", "")

        if not verifier_code:
            logger.warning("No verifier code, returning False")
            return False

        try:
            # Execute verifier
            reward = await env._compute_reward()
            success = reward > 0

            logger.info(
                f"Evaluation result for {task_config.get('task_key', 'unknown')}: "
                f"reward={reward}, success={success}"
            )

            return success
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return False


def load_fleet_tasks(
    tasks_file: str,
    max_tasks: Optional[int] = None,
    env_key_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load Fleet tasks from a JSON file.

    Args:
        tasks_file: Path to JSON file with tasks
        max_tasks: Optional limit on number of tasks
        env_key_filter: Optional filter by env_key

    Returns:
        List of task dicts
    """
    with open(tasks_file) as f:
        tasks = json.load(f)

    # Handle both list format and {"tasks": [...]} format
    if isinstance(tasks, dict) and "tasks" in tasks:
        tasks = tasks["tasks"]

    # Filter by env_key if specified
    if env_key_filter:
        tasks = [t for t in tasks if t.get("env_key") == env_key_filter]

    # Limit number of tasks
    if max_tasks and max_tasks > 0:
        tasks = tasks[:max_tasks]

    logger.info(f"Loaded {len(tasks)} Fleet tasks from {tasks_file}")

    return tasks
