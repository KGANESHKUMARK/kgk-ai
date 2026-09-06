"""KGK AI Tools package — tool registry and built-in tools."""

from app.tools.registry import BaseTool, ToolInfo, ToolResult, ToolRegistry, tool_registry
from app.tools.calculator import CalculatorTool
from app.tools.python_sandbox import PythonSandboxTool
from app.tools.doc_search import DocumentSearchTool


def register_default_tools() -> None:
    """Register all built-in tools in the global tool registry."""
    from app.config import get_settings

    settings = get_settings()

    tool_registry.register_tool(CalculatorTool())

    if settings.enable_python_tool:
        tool_registry.register_tool(PythonSandboxTool())

    if settings.enable_rag:
        tool_registry.register_tool(DocumentSearchTool())


__all__ = [
    "BaseTool",
    "ToolInfo",
    "ToolResult",
    "ToolRegistry",
    "tool_registry",
    "CalculatorTool",
    "PythonSandboxTool",
    "DocumentSearchTool",
    "register_default_tools",
]
