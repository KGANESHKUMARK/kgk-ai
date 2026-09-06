"""KGK AI Agent System — Abstract base agent.

All agents inherit from this base class. Agents use tools and the model
provider to accomplish tasks. The architecture is intentionally simple
for v1 — no complex multi-agent orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from app.models.base import BaseModelProvider, GenerationParams
from app.tools.registry import ToolRegistry


@dataclass
class AgentTask:
    """A task for an agent to execute.

    Attributes:
        query: The user's request or question.
        context: Additional context (e.g. conversation history).
        metadata: Extra task-specific metadata.
    """

    query: str
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result of an agent's execution.

    Attributes:
        success: Whether the agent completed the task successfully.
        response: The agent's response text.
        tools_used: List of tool names that were used.
        sources: List of sources cited (for RAG-backed agents).
        error: Error message if the task failed.
    """

    success: bool
    response: str = ""
    tools_used: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    error: str = ""


class BaseAgent(ABC):
    """Abstract base class for KGK AI agents.

    Agents receive a task, optionally use tools, and return a result.
    Each agent type implements its own execution logic.

    Attributes:
        name: Agent identifier.
        description: What this agent does.
        model: Reference to the model provider.
        tools: Reference to the tool registry.
    """

    def __init__(
        self,
        name: str,
        description: str,
        model: Optional[BaseModelProvider] = None,
        tools: Optional[ToolRegistry] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.model = model
        self.tools = tools

    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the given task.

        Args:
            task: The task to execute.

        Returns:
            AgentResult with the outcome.
        """
        ...

    def can_handle(self, task: AgentTask) -> bool:
        """Determine if this agent can handle the given task.

        Override in subclasses for task-specific routing logic.
        Default: True (any agent can attempt any task).

        Args:
            task: The task to evaluate.

        Returns:
            True if this agent can handle the task.
        """
        return True
