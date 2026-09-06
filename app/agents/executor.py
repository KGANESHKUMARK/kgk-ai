"""KGK AI Agent Executor — Orchestrates agent execution.

The executor manages the lifecycle of agent runs:
1. Plans the task (single or multi-step)
2. Executes each sub-task with the ReAct agent
3. Combines results into a final response
"""

from __future__ import annotations

from typing import Any, Optional

from app.agents.base import AgentResult, AgentTask, BaseAgent
from app.agents.planner import TaskPlanner
from app.agents.react import ReActAgent
from app.logging_config import get_logger

logger = get_logger("agents.executor")

_COMBINE_PROMPT = """You are KGK AI. You have completed multiple sub-tasks to answer a user's question. Combine the results into a single coherent response.

Original question: {query}

Sub-task results:
{results}

Provide a clear, unified response that addresses the original question. Do not mention sub-tasks or the planning process."""


class AgentExecutor:
    """Orchestrates agent execution with task planning.

    Attributes:
        agent: The agent to use for sub-task execution.
        planner: The task planner for query decomposition.
        model: Model provider for result combination.
    """

    def __init__(
        self,
        agent: Optional[BaseAgent] = None,
        planner: Optional[TaskPlanner] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.agent = agent or ReActAgent(model=model)
        self.planner = planner or TaskPlanner(model=model)

    def run(self, query: str, context: str = "") -> AgentResult:
        """Execute a query through the agent pipeline.

        Args:
            query: User's query string.
            context: Additional context (conversation history).

        Returns:
            AgentResult with the combined response.
        """
        logger.info(
            f"Agent executor started: query_len={len(query)}",
            extra={"component": "agents.executor", "query_length": len(query)},
        )

        plan = self.planner.plan(query, context)

        if not plan.is_multi_step:
            task = AgentTask(query=query, context=context)
            result = self.agent.execute(task)
            logger.info(
                f"Agent executor completed (single-step): success={result.success}",
                extra={"component": "agents.executor", "success": result.success},
            )
            return result

        return self._run_multi_step(query, context, plan.subtasks)

    def _run_multi_step(
        self,
        query: str,
        context: str,
        subtasks: list[str],
    ) -> AgentResult:
        """Execute a multi-step plan.

        Args:
            query: Original user query.
            context: Conversation context.
            subtasks: List of sub-task descriptions.

        Returns:
            Combined AgentResult.
        """
        all_tools_used: list[str] = []
        all_sources: list[str] = []
        subtask_results: list[str] = []
        accumulated_context = context

        for i, subtask_desc in enumerate(subtasks):
            logger.info(
                f"Executing subtask {i + 1}/{len(subtasks)}: {subtask_desc[:80]}",
                extra={"component": "agents.executor", "subtask_index": i + 1, "subtask_total": len(subtasks)},
            )

            task = AgentTask(
                query=subtask_desc,
                context=accumulated_context,
                metadata={"subtask_index": i, "original_query": query},
            )

            result = self.agent.execute(task)

            for tool in result.tools_used:
                if tool not in all_tools_used:
                    all_tools_used.append(tool)
            all_sources.extend(result.sources)

            if result.success:
                subtask_results.append(f"Sub-task {i + 1}: {subtask_desc}\nResult: {result.response}")
                accumulated_context += f"\n[Sub-task {i + 1} result]: {result.response}"
            else:
                subtask_results.append(f"Sub-task {i + 1}: {subtask_desc}\nResult: (failed - {result.error})")

        combined_response = self._combine_results(query, subtask_results)

        logger.info(
            f"Agent executor completed (multi-step): {len(subtasks)} subtasks, {len(all_tools_used)} tools",
            extra={
                "component": "agents.executor",
                "subtask_count": len(subtasks),
                "tools_used": all_tools_used,
            },
        )

        return AgentResult(
            success=True,
            response=combined_response,
            tools_used=all_tools_used,
            sources=all_sources,
        )

    def _combine_results(self, query: str, results: list[str]) -> str:
        """Combine sub-task results into a final response.

        Args:
            query: Original user query.
            results: List of sub-task result strings.

        Returns:
            Combined response string.
        """
        if len(results) == 1:
            return results[0].split("Result: ", 1)[-1] if "Result: " in results[0] else results[0]

        if self.model is None:
            return "\n\n".join(r.split("Result: ", 1)[-1] if "Result: " in r else r for r in results)

        try:
            results_text = "\n\n".join(results)
            prompt = _COMBINE_PROMPT.format(query=query, results=results_text)

            messages = [
                {"role": "system", "content": "You are KGK AI, an intelligent assistant."},
                {"role": "user", "content": prompt},
            ]

            result = self.model.generate(messages)
            return result.text.strip()
        except Exception as e:
            logger.warning(f"Result combination failed, using concatenation: {e}")
            return "\n\n".join(r.split("Result: ", 1)[-1] if "Result: " in r else r for r in results)
