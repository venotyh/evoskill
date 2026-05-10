"""Fitness evaluation — scores skills by running them against tasks."""

from __future__ import annotations

import json
import os

from .agent import SkillAgent
from .llm import LLMClient
from .skill import Skill
from .tasks import BUILTIN_TASKS, EvoTask


class FitnessEvaluator:
    """Evaluates skill fitness by running tasks and scoring with LLM-as-judge."""

    def __init__(self, model: str | None = None, task_subset: list[str] | None = None):
        self.model = model or os.environ.get("EVOSKILL_MODEL", "claude-sonnet-4-20250514")
        self.provider = os.environ.get("EVOSKILL_PROVIDER", "anthropic")
        self.tasks = self._select_tasks(task_subset)

    def _select_tasks(self, subset: list[str] | None) -> list[EvoTask]:
        if subset:
            lookup = {t.id: t for t in BUILTIN_TASKS}
            return [lookup[tid] for tid in subset if tid in lookup]
        return BUILTIN_TASKS

    def evaluate_skill(self, skill: Skill, max_tasks: int = 5) -> float:
        """Evaluate a skill on a sample of tasks and return composite fitness score."""
        import random
        sample = random.sample(self.tasks, min(max_tasks, len(self.tasks)))

        scores = []
        for task in sample:
            score = self._score_single_task(skill, task)
            scores.append(score)
            skill.record_fitness(score)

        return round(sum(scores) / len(scores), 2) if scores else 0.0

    def _score_single_task(self, skill: Skill, task: EvoTask) -> float:
        """Run a single task and score it with LLM-as-judge."""
        agent = SkillAgent(skill, model=self.model)
        result = agent.run(task.prompt)

        output = result.get("output", "")
        tool_calls = result.get("tool_calls", [])
        success = result.get("success", False)

        # Structural scoring (pre-judge)
        structural_score = 5.0
        if not success:
            structural_score -= 2.0
        if output and len(output) > 20:
            structural_score += 1.0
        expected = set(task.expected_tools)
        used = set(tool_calls)
        if expected and used:
            overlap = len(expected & used) / len(expected)
            structural_score += overlap * 2.0

        judge_score = self._judge_output(task, output, tool_calls)
        # Weight: 20% structural, 80% judge — judge quality is what matters
        final_score = round(structural_score * 0.2 + judge_score * 0.8, 1)
        return max(1.0, min(10.0, final_score))

    def _judge_output(self, task: EvoTask, output: str, tool_calls: list[str]) -> float:
        """Use an LLM to judge output quality with a discriminating rubric."""
        judge_prompt = (
            f"You are evaluating an AI agent's performance on a task.\n\n"
            f"TASK:\n{task.prompt}\n\n"
            f"AGENT OUTPUT:\n{output[:2000]}\n\n"
            f"TOOLS USED: {', '.join(tool_calls) if tool_calls else 'none'}\n"
            f"REFERENCE ANSWER HINT: {task.reference_answer_hint}\n\n"
            f"Score 1-10 using this STRICT rubric:\n"
            f"  1-3: Completely wrong, irrelevant, or failed to attempt the task\n"
            f"  4-6: Partially correct, missing key steps, or vague answer\n"
            f"  7-8: Mostly correct, reasonable but not excellent\n"
            f"  9-10: Excellent, complete, clear, efficient — could not be better\n\n"
            f"Do NOT give the same score to every response. Be critical and discriminating. "
            f"Compare: would a BETTER prompt have produced a BETTER output?\n\n"
            f"Reply with ONLY: {{\"score\": <int 1-10>, \"reason\": \"<one sentence>\"}}"
        )

        client = LLMClient(model=self.model, provider=self.provider)
        resp = client.chat(
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=256,
        )
        return self._parse_judge_response(resp.content or "")

    def _parse_judge_response(self, text: str) -> float:
        """Extract score from LLM judge response."""
        try:
            # Try to find JSON
            if "{" in text and "}" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
                score = float(data.get("score", 5))
                return max(1.0, min(10.0, score))
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: extract any number
        import re
        nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
        if nums:
            score = float(nums[0])
            return max(1.0, min(10.0, score))
        return 5.0


def quick_fitness(skill: Skill, num_tasks: int = 2) -> float:
    """Evaluate skill on a few tasks with LLM-as-judge scoring.

    Returns composite fitness score (1-10). Each task is judged by LLM.
    """
    evaluator = FitnessEvaluator()
    return evaluator.evaluate_skill(skill, max_tasks=num_tasks)
