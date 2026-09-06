"""KGK AI Tool Registry — Extensible tool system.

Tools are registered with a name, description, and callable.
The registry enforces safe execution and provides introspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.logging_config import get_logger

logger = get_logger("tools")


@dataclass
class ToolResult:
    """Result of a tool execution.

    Attributes:
        success: Whether the tool executed successfully.
        output: The tool's output (string or structured data).
        error: Error message if execution failed.
        metadata: Additional tool-specific metadata.
    """

    success: bool
    output: Any = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolInfo:
    """Metadata describing a registered tool.

    Attributes:
        name: Unique tool name.
        description: Human-readable description of what the tool does.
        parameters: Description of expected parameters.
        safe_for_public: Whether the tool is safe for untrusted users.
    """

    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)
    safe_for_public: bool = True


class BaseTool(ABC):
    """Abstract base class for KGK AI tools.

    Implementations must provide:
        - info(): Return tool metadata.
        - execute(): Run the tool with given arguments.
    """

    @abstractmethod
    def info(self) -> ToolInfo:
        """Return tool metadata."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Tool-specific arguments.

        Returns:
            ToolResult with success status and output.
        """
        ...


class ToolRegistry:
    """Registry for managing available tools.

    Tools can be registered, removed, listed, and executed.
    Only tools marked safe_for_public are exposed to untrusted users.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool instance.

        Args:
            tool: BaseTool instance to register.
        """
        info = tool.info()
        self._tools[info.name] = tool
        logger.info(f"Registered tool: {info.name}")

    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name.

        Args:
            name: Name of the tool to remove.

        Returns:
            True if the tool was removed, False if not found.
        """
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Removed tool: {name}")
            return True
        return False

    def list_tools(self, public_only: bool = False) -> list[ToolInfo]:
        """List all registered tools.

        Args:
            public_only: If True, only return tools safe for public users.

        Returns:
            List of ToolInfo for each registered tool.
        """
        tools = []
        for tool in self._tools.values():
            info = tool.info()
            if not public_only or info.safe_for_public:
                tools.append(info)
        return tools

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def execute_tool(self, name: str, public_context: bool = False, **kwargs: Any) -> ToolResult:
        """Execute a tool by name.

        Args:
            name: Name of the tool to execute.
            public_context: If True, only allow safe_for_public tools.
            **kwargs: Arguments to pass to the tool.

        Returns:
            ToolResult with the execution outcome.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Tool '{name}' not found.")

        info = tool.info()
        if public_context and not info.safe_for_public:
            logger.warning(f"Blocked execution of non-public tool '{name}' in public context")
            return ToolResult(
                success=False,
                error=f"Tool '{name}' is not available in this context.",
            )

        try:
            logger.info(f"Executing tool: {name}", extra={"component": "tools", "tool": name})
            return tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}", extra={"component": "tools", "tool": name})
            return ToolResult(success=False, error=str(e))


tool_registry = ToolRegistry()
