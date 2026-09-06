"""KGK AI ReAct Agent — Reason-Act loop implementation.

The ReAct (Reason + Act) agent iteratively:
1. Reasons about the current state
2. Decides whether to use a tool or give a final answer
3. Executes the chosen tool if needed
4. Observes the result and loops back to step 1

This continues until the agent produces a final answer or hits the max steps.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from app.agents.base import AgentResult, AgentTask, BaseAgent
from app.logging_config import get_logger

logger = get_logger("agents.react")

_MAX_STEPS = 8

_REACT_SYSTEM_PROMPT = """You are KGK AI, an intelligent assistant that can use tools to help answer questions.

Available tools:
{tool_descriptions}

To use a tool, respond with EXACTLY this format:
THOUGHT: <your reasoning about what to do>
ACTION: <tool_name>
ACTION_INPUT: <arguments as key=value pairs separated by commas>

To give your final answer, respond with:
THOUGHT: <your reasoning>
ANSWER: <your final answer>

You must use the exact format above. Only use one tool per step.
If you already have enough information, provide your final answer with ANSWER."""


class ReActAgent(BaseAgent):
    """ReAct agent that reasons and acts in a loop.

    Uses the model to decide which tools to call, executes them,
    and iterates until a final answer is produced.

    Attributes:
        max_steps: Maximum number of reasoning steps before giving up.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tools: Optional[Any] = None,
        max_steps: int = _MAX_STEPS,
    ) -> None:
        super().__init__(
            name="react_agent",
            description="ReAct agent that reasons and uses tools to answer complex questions.",
            model=model,
            tools=tools,
        )
        self.max_steps = max_steps

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute a task using the ReAct loop.

        Args:
            task: The task to execute.

        Returns:
            AgentResult with the outcome.
        """
        if self.model is None:
            return AgentResult(
                success=False,
                error="No model provider available for ReAct agent.",
            )

        tool_descriptions = self._get_tool_descriptions()
        system_prompt = _REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        if task.context:
            messages.append({"role": "user", "content": f"Previous context:\n{task.context}"})

        messages.append({"role": "user", "content": task.query})

        tools_used: list[str] = []
        sources: list[str] = []

        for step in range(self.max_steps):
            logger.info(
                f"ReAct step {step + 1}/{self.max_steps}",
                extra={"component": "agents.react", "step": step + 1},
            )

            try:
                result = self.model.generate(messages)
                raw_response = result.text.strip()
            except Exception as e:
                logger.error(f"Model generation failed at step {step + 1}: {e}")
                return AgentResult(
                    success=False,
                    error=f"Model generation failed: {e}",
                    tools_used=tools_used,
                )

            assistant_msg = {"role": "assistant", "content": raw_response}
            messages.append(assistant_msg)

            parsed = self._parse_response(raw_response)

            if parsed["answer"] is not None:
                logger.info(
                    f"ReAct completed with final answer at step {step + 1}",
                    extra={"component": "agents.react", "step": step + 1, "tools_used": tools_used},
                )
                return AgentResult(
                    success=True,
                    response=parsed["answer"],
                    tools_used=tools_used,
                    sources=sources,
                )

            if parsed["action"] and parsed["action_input"] is not None:
                tool_name = parsed["action"]
                tool_input = parsed["action_input"]

                if self.tools is None:
                    observation = "Error: No tools available."
                else:
                    tool_result = self.tools.execute_tool(tool_name, **tool_input)
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)

                    if tool_result.success:
                        observation = tool_result.output or "(no output)"
                        if tool_result.metadata:
                            if "sources" in tool_result.metadata:
                                sources.extend(tool_result.metadata["sources"])
                    else:
                        observation = f"Error: {tool_result.error}"

                messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
                logger.info(
                    f"Tool '{tool_name}' executed",
                    extra={"component": "agents.react", "tool": tool_name},
                )
            else:
                messages.append({
                    "role": "user",
                    "content": "Please use the exact format. Respond with either ACTION or ANSWER.",
                })

        logger.warning(f"ReAct agent hit max steps ({self.max_steps}) without final answer")
        return AgentResult(
            success=False,
            response="I was unable to complete this task within the step limit. Please try rephrasing your request.",
            tools_used=tools_used,
            sources=sources,
            error="max_steps_exceeded",
        )

    def _get_tool_descriptions(self) -> str:
        """Build tool description string for the system prompt."""
        if self.tools is None:
            return "(no tools available)"

        tool_list = self.tools.list_tools()
        if not tool_list:
            return "(no tools available)"

        lines = []
        for info in tool_list:
            params = ", ".join(f"{k}: {v}" for k, v in info.parameters.items())
            lines.append(f"- {info.name}: {info.description} (params: {params})")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse the model's response for ACTION or ANSWER.

        Args:
            response: Raw model response text.

        Returns:
            Dict with keys: answer, action, action_input, thought.
        """
        result: dict[str, Any] = {
            "answer": None,
            "action": None,
            "action_input": None,
            "thought": None,
        }

        answer_match = re.search(r"ANSWER:\s*(.+)", response, re.DOTALL)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
            return result

        action_match = re.search(r"ACTION:\s*(\S+)", response)
        if action_match:
            result["action"] = action_match.group(1).strip()

            input_match = re.search(r"ACTION_INPUT:\s*(.+)", response, re.DOTALL)
            if input_match:
                result["action_input"] = self._parse_action_input(input_match.group(1).strip())

        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=\n[A-Z]|$)", response, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        return result

    def _parse_action_input(self, input_str: str) -> dict[str, Any]:
        """Parse action input string into kwargs dict.

        Supports key=value pairs separated by commas.
        Values are automatically typed (int, float, bool, str).

        Args:
            input_str: Raw action input string.

        Returns:
            Dict of parsed keyword arguments.
        """
        kwargs: dict[str, Any] = {}

        if not input_str:
            return kwargs

        parts = self._split_input(input_str)

        for part in parts:
            part = part.strip()
            if "=" not in part:
                if len(parts) == 1:
                    kwargs["expression"] = part
                    break
                continue

            key, _, value = part.partition("=")
            key = key.strip()
            value = value.strip()

            kwargs[key] = self._coerce_value(value)

        return kwargs

    def _split_input(self, input_str: str) -> list[str]:
        """Split action input by commas, respecting quotes."""
        parts = []
        current = []
        in_quotes = False
        quote_char = ""

        for char in input_str:
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
                current.append(char)
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = ""
                current.append(char)
            elif char == "," and not in_quotes:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)

        if current:
            parts.append("".join(current))

        return parts

    def _coerce_value(self, value: str) -> Any:
        """Convert a string value to the appropriate Python type.

        Args:
            value: String value to coerce.

        Returns:
            Python value (int, float, bool, str).
        """
        value = value.strip().strip("\"'")

        if value.lower() in ("true", "yes"):
            return True
        if value.lower() in ("false", "no"):
            return False
        if value.lower() in ("none", "null"):
            return None

        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            pass

        return value
