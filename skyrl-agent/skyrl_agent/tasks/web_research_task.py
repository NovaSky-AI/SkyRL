"""
Web Research Task - Uses search_engine and web_browser tools together
"""

from typing import Dict, Any, List
from skyrl_agent.tasks.base import BaseTask


class WebResearchTask(BaseTask):
    """
    A task that combines search_engine and web_browser tools for web research.
    This task uses the existing tools without modification.
    """

    @classmethod
    async def initialize_runtime(cls, *args, **kwargs) -> Any:
        """Initialize the runtime for the web research task"""
        # No special runtime initialization needed
        return {}

    @classmethod
    def get_instruction(cls, instance: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Get the initial instruction for the agent in OpenAI messages format.
        The agent will use tools sequentially as per ReActAgent's design.
        """
        # Handle pandas Series or dict input
        import pandas as pd

        # Extract prompt - simplified since our unified dataset has 'prompt' field
        if isinstance(instance, pd.Series):
            # Use the prompt field from our unified dataset
            prompt = instance.get("prompt", "")

            # Debug output (optional)
            if prompt:
                print(f"[DEBUG] Using prompt from unified dataset (length: {len(prompt)})")
            else:
                # Fallback to Question field if using old format
                prompt = instance.get("Question", "")
                if prompt:
                    print("[DEBUG] Using Question field as fallback")
        else:
            # Handle dict-like objects
            prompt = instance.get("prompt", instance.get("Question", "")) if hasattr(instance, "get") else str(instance)

        # Debug: show raw prompt source when enabled
        import os as _os

        _debug = _os.getenv("SKYAGENT_DEBUG_LOG", "0") == "1"

        # Handle different input formats
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            prompt = [prompt]
        elif not isinstance(prompt, list):
            prompt = [{"role": "user", "content": str(prompt)}]

        # Check if there's already a system message
        has_system = any(msg.get("role") == "system" for msg in prompt)

        if not has_system:
            # Add system prompt that explains how to use the tools
            system_prompt = {
                "role": "system",
                "content": """You are a precise, helpful assistant. You decide whether to use tools (search_engine, web_browser) based on the task.

Use search_engine to find relevant sources.
Use web_browser to visit URLs for detailed information.
If you DID call search_engine, do NOT answer from snippets—open at least one source with web_browser first.

Always end by calling the finish tool with exactly one \\boxed{...} as the final answer.

Typical workflow:
1) search_engine → get URLs
2) web_browser → extract key facts from sources
3) finish → \\boxed{YOUR_ANSWER}
""",
            }
            prompt = [system_prompt] + prompt

        if _debug:
            # Print system + first user message preview
            def _preview(msg, w=300):
                s = str(msg).replace("\n", " ")
                return s[:w] + ("..." if len(s) > w else "")

            if isinstance(prompt, list) and len(prompt) > 0:
                sys_msg = next((m for m in prompt if m.get("role") == "system"), prompt[0])
                usr_msg = next((m for m in prompt if m.get("role") == "user"), prompt[min(1, len(prompt) - 1)])
                print("[WebResearchTask] System prompt preview:", _preview(sys_msg.get("content", "")))
                print("[WebResearchTask] User prompt preview:", _preview(usr_msg.get("content", "")))
        return prompt

    @classmethod
    def complete_runtime(cls, *args, **kwargs) -> Dict[str, Any]:
        """Complete or finalize the runtime for the task"""
        # No special cleanup needed
        return {}

    @classmethod
    async def evaluate_result(
        cls, result: Any, instance: Any, data_source: str = None, instance_id: int = None, trajectory_id: int = None
    ) -> float:
        """
        STEM evaluation using LLM judge for textbook data; otherwise simple pass-through.
        """
        import pandas as pd

        # No result => incorrect
        if result is None or (isinstance(result, str) and not result.strip()):
            return 0.0

        # Debug logging for instance type and content
        print(f"[evaluate_result] instance_id={instance_id} instance type: {type(instance).__name__}")

        # Normalize instance into a dict
        try:
            if isinstance(instance, pd.Series):
                print(f"[evaluate_result] Converting Series to dict for instance_id={instance_id}")
                instance = instance.to_dict()
        except Exception as e:
            print(f"[evaluate_result] Failed to convert Series for instance_id={instance_id}: {e}")
            pass

        # Log instance structure after normalization
        if isinstance(instance, dict):
            print(f"[evaluate_result] instance_id={instance_id} keys: {list(instance.keys())}")
            if "reward_model" in instance:
                rm_type = type(instance["reward_model"]).__name__
                print(f"[evaluate_result] reward_model type: {rm_type}")
            if "extra_info" in instance:
                ei_type = type(instance["extra_info"]).__name__
                print(f"[evaluate_result] extra_info type: {ei_type}")

        # Determine data_source
        ds = data_source
        if not ds and isinstance(instance, dict):
            ds = instance.get("data_source")

        # Apply LLM judge for STEM textbook and HLE data
        ds_lower = str(ds or "").lower()
        if ds_lower in ["stem_textbook", "hle"]:
            try:
                # Extract fields with robust GT fallback (align with training-side logic)
                import json as _json

                rm_raw = instance.get("reward_model") if isinstance(instance, dict) else None
                ei_raw = instance.get("extra_info") if isinstance(instance, dict) else None
                # Handle JSON-serialized dicts
                if isinstance(rm_raw, str):
                    try:
                        rm = _json.loads(rm_raw)
                        print(f"[evaluate_result] Parsed reward_model from JSON for instance_id={instance_id}")
                    except Exception as e:
                        print(f"[evaluate_result] Failed to parse reward_model JSON for instance_id={instance_id}: {e}")
                        rm = {}
                else:
                    rm = rm_raw or {}
                if isinstance(ei_raw, str):
                    try:
                        ei = _json.loads(ei_raw)
                        print(f"[evaluate_result] Parsed extra_info from JSON for instance_id={instance_id}")
                    except Exception as e:
                        print(f"[evaluate_result] Failed to parse extra_info JSON for instance_id={instance_id}: {e}")
                        ei = {}
                else:
                    ei = ei_raw or {}

                # Prefer: nested instance.answer -> top-level answer -> extra_info.reference_answer -> reward_model.ground_truth
                ground_truth = ""
                try:
                    if isinstance(instance, dict):
                        nested_inst = instance.get("instance") if isinstance(instance.get("instance"), dict) else {}
                        if nested_inst and nested_inst.get("answer"):
                            ground_truth = str(nested_inst.get("answer"))
                except Exception:
                    ground_truth = ""
                if not ground_truth and isinstance(instance, dict) and instance.get("answer"):
                    ground_truth = str(instance.get("answer"))
                if not ground_truth and isinstance(ei, dict) and ei.get("reference_answer"):
                    ground_truth = str(ei.get("reference_answer"))
                if not ground_truth and isinstance(rm, dict) and rm.get("ground_truth"):
                    ground_truth = str(rm.get("ground_truth"))

                if not ground_truth:
                    # No GT → cannot judge; return 0 and emit a concise trace
                    try:
                        print(
                            f"[rollout-judge-skip] instance_id={instance_id} traj={trajectory_id} ds={ds_lower} reason=missing_ground_truth"
                        )
                        # Debug why ground truth is missing
                        print(f"  [DEBUG] rm keys: {list(rm.keys()) if isinstance(rm, dict) else 'not dict'}")
                        print(f"  [DEBUG] ei keys: {list(ei.keys()) if isinstance(ei, dict) else 'not dict'}")
                        if isinstance(rm, dict) and "ground_truth" in rm:
                            print(f"  [DEBUG] rm.ground_truth exists but is: {rm['ground_truth']!r}")
                        if isinstance(ei, dict) and "reference_answer" in ei:
                            print(f"  [DEBUG] ei.reference_answer exists but is: {ei['reference_answer']!r}")
                    except Exception:
                        pass
                    return 0.0

                # Ensure question/subject available for judge
                question = ei.get("question") or (instance.get("raw_prompt", "") if isinstance(instance, dict) else "")
                subject = ei.get("subject", "")
                judge_extra = dict(ei)
                judge_extra.setdefault("question", question)
                judge_extra.setdefault("subject", subject)

                # Call synchronous judge safely from async context
                from skyrl_agent.dispatcher.async_utils import call_sync_from_async

                # Use the original minimal web-search STEM judge (as before)
                from skyrl_agent.tasks.verifiers.web_search.stem_llm_judge import compute_score as _compute_score

                score = await call_sync_from_async(
                    _compute_score,
                    ds,
                    str(result),
                    ground_truth,
                    judge_extra,
                )
                # Lightweight trace to confirm rollout-side LLM judge usage
                try:
                    import os as _os

                    _base = _os.environ.get("STEM_LLM_JUDGE_URL", "") or _os.environ.get("MATH_LLM_JUDGE_URL", "") or ""
                    print(
                        f"[rollout-judge-used] instance_id={instance_id} traj={trajectory_id} ds={ds_lower} score={float(score)} base={_base}"
                    )
                except Exception:
                    pass
                return float(score)
            except Exception as e:
                print(f"[eval-error] STEM LLM judge failed: {e}")
                return 0.0
        else:
            # For non-STEM datasets, do not call judge; emit a concise trace once per item when result exists
            try:
                if result:
                    print(
                        f"[rollout-judge-skip] instance_id={instance_id} traj={trajectory_id} ds={ds_lower} reason=non_stem_dataset"
                    )
            except Exception:
                pass

        # Default behavior for other data sources: success if result exists
        return 1.0 if result else 0.0
