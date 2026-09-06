"""KGK AI Task Planner — Decompose complex queries into sub-tasks.

The task planner analyzes a user query and determines whether it needs
to be broken into multiple sub-tasks. Each sub-task is then executed
by an agent, and results are combined into a final response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.logging_config import get_logger

logger = get_logger("agents.planner")

_PLANNER_SYSTEM_PROMPT = """You are a task planner for KGK AI. Analyze the user's request and determine if it needs to be broken into sub-tasks.

If the request is simple and can be answered directly, respond with:
PLAN: single
SUBTASKS: none

If the request requires multiple steps, respond with:
PLAN: multi
SUBTASKS:
1. <first sub-task description>
2. <second sub-task description>
...

Keep sub-task descriptions concise and actionable. Maximum 5 sub-tasks."""


@dataclass
class TaskPlan:
    """A plan for executing a task.

    Attributes:
        is_multi_step: Whether the task requires multiple sub-tasks.
        subtasks: List of sub-task descriptions.
    """

    is_multi_step: bool
    subtasks: list[str] = field(default_factory=list)


class TaskPlanner:
    """Decomposes complex queries into sub-tasks.

    Uses the model provider to analyze queries. Falls back to a
    single-step plan if no model is available.

    Attributes:
        model: Optional model provider for intelligent planning.
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        self.model = model

    def plan(self, query: str, context: str = "") -> TaskPlan:
        """Create an execution plan for a query.

        Args:
            query: The user's query.
            context: Additional context (conversation history).

        Returns:
            TaskPlan with subtask breakdown.
        """
        if self.model is None:
            return TaskPlan(is_multi_step=False, subtasks=[query])

        try:
            return self._plan_with_model(query, context)
        except Exception as e:
            logger.warning(f"Model planning failed, falling back to single-step: {e}")
            return TaskPlan(is_multi_step=False, subtasks=[query])

    def _plan_with_model(self, query: str, context: str) -> TaskPlan:
        """Use the model to create a plan.

        Args:
            query: User query.
            context: Additional context.

        Returns:
            TaskPlan parsed from model output.
        """
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
        ]

        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"})
        else:
            messages.append({"role": "user", "content": query})

        result = self.model.generate(messages)
        response = result.text.strip()

        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> TaskPlan:
        """Parse model response into a TaskPlan.

        Args:
            response: Model response text.

        Returns:
            TaskPlan instance.
        """
        is_multi = "multi" in response.lower()

        if not is_multi:
            return TaskPlan(is_multi_step=False)

        subtasks: list[str] = []
        in_subtasks = False

        for line in response.split("\n"):
            line = line.strip()

            if line.upper().startswith("SUBTASKS:"):
                in_subtasks = True
                rest = line[len("SUBTASKS:"):].strip()
                if rest and rest.lower() != "none":
                    subtasks.append(rest)
                continue

            if in_subtasks:
                if not line:
                    continue
                if line[0].isdigit() and "." in line[:3]:
                    desc = line.split(".", 1)[1].strip()
                    if desc:
                        subtasks.append(desc)
                elif line.startswith("-"):
                    desc = line.lstrip("-").strip()
                    if desc:
                        subtasks.append(desc)
                else:
                    break

        if not subtasks:
            return TaskPlan(is_multi_step=False)

        subtasks = subtasks[:5]

        logger.info(
            f"Task plan created: {len(subtasks)} subtasks",
            extra={"component": "agents.planner", "subtask_count": len(subtasks)},
        )

        return TaskPlan(is_multi_step=True, subtasks=subtasks)
