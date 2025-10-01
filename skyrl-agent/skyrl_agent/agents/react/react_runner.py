import pandas as pd

from skyrl_agent.agents.react.react_agent import ReActAgent

from skyrl_agent.agents.base import BaseTrajectory


class ReActTrajectory(BaseTrajectory):
    async def initialize_trajectory(self):
        pass

    async def generate_trajectory(self) -> None:
        data = self.data
        instance_id = data["instance_id"] if data["instance_id"] else self.cfg.instance_id
        instance = pd.Series(data["instance"])
        # self.agent = ReActAgent(traj_config=self.cfg, infer_engine=self.infer_engine, tokenizer=self.tokenizer)
        self.agent: ReActAgent = self.agent_cls(
            traj_config=self.cfg,
            infer_engine=self.infer_engine,
            tokenizer=self.tokenizer,
        )

        # sys + user messages
        instruction = self.task.get_instruction(instance)

        finish_reason, result = await self.agent.run(instruction)
        # Optional tool profile snapshot (env-gated inside agent)
        tool_profile = None
        try:
            tool_profile = self.agent.get_tool_profile()
        except Exception:
            tool_profile = None
        self.result = {
            "instance_id": instance_id,
            "trajectory_id": self.cfg.trajectory_id,
            "messages": self.agent.get_messages(),
            "results": result,
            "finish_reason": finish_reason,
            "state": {"tool_profile": tool_profile} if tool_profile is not None else {},
        }

    async def evaluate_trajectory(self) -> None:
        instance_id = self.cfg.instance_id
        trajectory_id = self.cfg.trajectory_id
        data = self.data
        instance_id = data["instance_id"] if data["instance_id"] else self.cfg.instance_id
        instance = data["instance"]
        # print(f"[react_runner] instance_id={instance_id} original instance type: {type(instance).__name__}")
        # if isinstance(instance, dict):
        #     # print(f"[react_runner] instance keys: {list(instance.keys())}")
        # if not isinstance(instance, (dict, pd.Series)):
        #     # print(f"[react_runner] Converting to Series for instance_id={instance_id}")
        #     instance = pd.Series(instance)
        result = self.result.get("results")

        try:
            eval_result = await self.task.evaluate_result(
                result,
                instance,
                data["data_source"],
                instance_id,
                trajectory_id,
            )
            self.result["reward"] = eval_result
        except Exception as e:
            print(f"Error evaluating result: {e}")
            self.result["reward"] = 0
            self.result["eval_error"] = str(e)
