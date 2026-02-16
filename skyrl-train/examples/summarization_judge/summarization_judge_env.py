"""
Summarization Judge Environment.

This environment uses an LLM grader to evaluate summaries against a 12-criterion rubric.
The grader returns scores 1-5 for each criterion, which are aggregated into a 0-1 reward.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig
from openai import OpenAI
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


GRADING_SYSTEM_PROMPT = """## ROLE & CONTEXT

You are a deterministic summarization agent operating within the **Grading** module.

**Your function**: Evaluate completed summaries against standardized 12-criterion rubric with 1-5 scoring

You process source documents to create or evaluate summaries that are factually accurate, appropriately scoped, and 
tailored to user requirements. All outputs must be traceable to source material—you never fabricate, speculate, or 
extrapolate beyond explicit content.

---

## USER REQUIREMENTS

The user provides these parameters to guide your work:

- **purpose**: Why this summary exists (e.g., 'board update', 'research abstract')
- **audience**: Who will read it (expertise level, role, context)
- **tone**: Desired voice/style (e.g., 'formal', 'conversational', 'neutral')
- **target_words**: Approximate length (flexible guideline, not strict limit)
- **focus_areas**: Optional emphasis instructions (what to highlight or minimize)

**Your job**: Honor these requirements while maintaining factual accuracy.
When purpose/audience suggest a structure (e.g., 'research abstract' → methods/results format), infer appropriately.
When tone guidance is vague, default to objective/factual voice.

---

## SCORING SCALE (1-5)

- **5: Perfect** - Truly flawless; you actively searched for issues and found NONE
- **4: Good** - Minor, verifiable issues only; still high quality
- **3: Acceptable** - Meets minimum bar; noticeable concerns but functional
- **2: Not Acceptable** - Clear, demonstrable problems requiring major work
- **1: Critical Failure** - Objectively unusable; fundamental violations

---

## 12-CRITERION RUBRIC

### Critical Criteria (6)
1. **factual_fidelity**: Every claim supported by source; no fabrication
2. **core_coverage**: Thesis + major findings + key recommendations present
3. **tone_fidelity**: Voice matches requested tone throughout
4. **standalone_usability**: Reader unfamiliar with source understands context/outcomes
5. **numeric_integrity**: Quantitative data matches source exactly
6. **temporal_accuracy**: Timeframes accurate; no extrapolation beyond source

### Major Criteria (3)
7. **proportionality**: Emphasis distribution mirrors source importance
8. **clarity**: Plain language, clear antecedents, logical flow
9. **audience_calibration**: Jargon/detail level appropriate for target audience

### Minor Criteria (3)
10. **purposeful_brevity**: Within target length (±20%); no filler
11. **terminology_consistency**: One canonical term per concept throughout
12. **ambiguity_transparency**: Source contradictions flagged neutrally

---

## YOUR TASK

Evaluate the summary using the **12-criterion rubric** (1-5 scale) and produce a **Score (1-5)** for each criteria,
along with a detailed **Rationale** citing specific evidence from the summary.

**Output format**: a JSON object with:
- "criteria": list of {"name": str, "score": int, "rationale": str}
- "reasoning": summary of evaluation across all criteria
- "score": aggregated score from 0.0 to 1.0 (average of all criteria scores divided by 5)

---

## CONSTRAINTS

### You Must NOT
1. Generate or revise content
2. Provide fix suggestions (grading evaluates only)
3. Downplay failures

### You Must
1. Evaluate objectively using 1-5 scale per rubric
2. Provide concise rationales
3. Apply aggregate logic consistently
4. Maintain neutral tone

**END OF GRADING MODULE**"""


@dataclass
class SummarizationJudgeEnvConfig:
    """Configuration for the summarization judge environment."""
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    api_key_env_var: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 4096


class SummarizationJudgeEnv(BaseTextEnv):
    """
    Environment for summarization tasks with LLM-as-a-judge grading.

    The grader evaluates summaries on a 12-criterion rubric and returns
    an aggregated score from 0.0 to 1.0.
    """

    def __init__(
        self,
        env_config: Union[SummarizationJudgeEnvConfig, DictConfig, None] = None,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        # Extract ground truth from reward_spec
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        
        ground_truth = extras["reward_spec"]["ground_truth"]
        self.user_intent = ground_truth.get("user_intent", {})
        self.original_document = ground_truth.get("original_document", "")

        # Setup config
        if env_config is None:
            env_config = SummarizationJudgeEnvConfig()
        
        if isinstance(env_config, DictConfig):
            self.model = env_config.get("model", "gpt-4o-mini")
            self.base_url = env_config.get("base_url", None)
            self.temperature = env_config.get("temperature", 0.0)
            self.max_tokens = env_config.get("max_tokens", 4096)
            api_key_env_var = env_config.get("api_key_env_var", "OPENAI_API_KEY")
        else:
            self.model = env_config.model
            self.base_url = env_config.base_url
            self.temperature = env_config.temperature
            self.max_tokens = env_config.max_tokens
            api_key_env_var = env_config.api_key_env_var

        # Set up OpenAI client
        api_key = os.getenv(api_key_env_var)
        if api_key is None:
            raise ValueError(f"`{api_key_env_var}` environment variable must be set")
        
        self.llm_client = OpenAI(base_url=self.base_url, api_key=api_key)

    def _format_user_intent(self) -> str:
        """Format user intent as a readable string."""
        if isinstance(self.user_intent, dict):
            parts = []
            for key, value in self.user_intent.items():
                parts.append(f"- **{key}**: {value}")
            return "\n".join(parts)
        return str(self.user_intent)

    def _build_grading_prompt(self, summary: str) -> str:
        """Build the user prompt for the grading API."""
        return f"""## User Intent
{self._format_user_intent()}

### Original Document ###
{self.original_document}

### Summary ###
{summary}"""

    def _parse_score(self, response_text: str) -> float:
        """Parse the aggregated score from the grader response."""
        try:
            # Try to parse as JSON
            # Handle potential markdown code blocks
            json_text = response_text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0]
            
            result = json.loads(json_text.strip())
            
            # Extract the aggregated score
            if "score" in result:
                score = float(result["score"])
                # Ensure score is in [0, 1] range
                return max(0.0, min(1.0, score))
            
            # Fallback: calculate from criteria scores
            if "criteria" in result:
                scores = [c.get("score", 3) for c in result["criteria"]]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    return avg_score / 5.0  # Normalize to 0-1
            
            print(f"Could not find score in response: {response_text[:200]}")
            return 0.0

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            # Fallback: try to extract score with regex
            match = re.search(r'"score"\s*:\s*([0-9.]+)', response_text)
            if match:
                return max(0.0, min(1.0, float(match.group(1))))
            
            print(f"Could not parse grader response: {response_text[:200]}")
            return 0.0

    def _get_reward(self, action: str) -> float:
        """Call the grading API to evaluate the summary."""
        user_prompt = self._build_grading_prompt(action)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": GRADING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            reply = response.choices[0].message.content.strip()
            return self._parse_score(reply)

        except Exception as e:
            print(f"Grading API error: {type(e).__name__}: {e}")
            return 0.0

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Evaluate the summary and return reward."""
        done = True  # Single-turn: always done after one step
        reward = self._get_reward(action)

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=done,
            metadata={
                "grader_model": self.model,
            },
        )
