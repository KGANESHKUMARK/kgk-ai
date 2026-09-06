"""KGK AI Python Sandbox Tool — Safe Python code execution.

Uses RestrictedPython to compile and execute untrusted Python code
in a restricted namespace. Includes timeout protection and output capture.
"""

from __future__ import annotations

import threading
from typing import Any

from app.config import get_settings
from app.logging_config import get_logger
from app.tools.registry import BaseTool, ToolInfo, ToolResult

logger = get_logger("tools.python_sandbox")

_MAX_OUTPUT_LENGTH = 5000
_MAX_EXECUTION_TIME = 5  # seconds


class PythonSandboxTool(BaseTool):
    """Sandboxed Python code execution tool.

    Uses RestrictedPython to restrict dangerous operations (imports, attribute
    access to dunder methods, etc.). Executes with a timeout and captures stdout.

    Not safe for public/untrusted users — only available in trusted contexts.
    """

    def info(self) -> ToolInfo:
        settings = get_settings()
        return ToolInfo(
            name="python_sandbox",
            description="Execute Python code in a restricted sandbox. Supports basic operations, printing, and math. Not available for untrusted users.",
            parameters={
                "code": "Python code string to execute",
            },
            safe_for_public=False,
        )

    def execute(self, code: str = "", **kwargs: Any) -> ToolResult:
        """Execute Python code in a restricted sandbox.

        Args:
            code: Python code string to execute.

        Returns:
            ToolResult with captured stdout or error.
        """
        if not code or not code.strip():
            return ToolResult(success=False, error="No code provided.")

        settings = get_settings()
        timeout = settings.python_tool_timeout

        try:
            from RestrictedPython import compile_restricted
            from RestrictedPython.Guards import safe_builtins, safer_getattr
        except ImportError:
            return ToolResult(
                success=False,
                error="RestrictedPython not installed. Install with: pip install RestrictedPython",
            )

        # Compile with restrictions
        try:
            byte_code = compile_restricted(code, filename="<sandbox>", mode="exec")
        except SyntaxError as e:
            return ToolResult(success=False, error=f"Syntax error: {e}")

        # Check for compilation errors
        if byte_code is None:
            return ToolResult(success=False, error="Code compilation failed (restricted).")

        # Prepare restricted globals
        # RestrictedPython transforms print() into _print_() calls
        # which need a collector object with _call_print and __str__
        class _PrintCollector:
            def __init__(self):
                self.txt = []
            def write(self, text):
                self.txt.append(text)
            def _call_print(self, *args, **kwargs):
                kwargs["file"] = self
                print(*args, **kwargs)
            def __str__(self):
                return "".join(self.txt)

        def _print_(_getattr):
            return _PrintCollector()

        # _getiter_ is required for loops and iterations
        def _getiter_(obj):
            return iter(obj)

        # _getitem_ is required for subscript access
        def _getitem_(obj, key):
            return obj[key]

        restricted_globals = {
            "__builtins__": {
                **safe_builtins,
                "range": range,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "type": type,
                "isinstance": isinstance,
                "True": True,
                "False": False,
                "None": None,
            },
            "_getattr_": safer_getattr,
            "_write_": lambda obj: obj,
            "_inplacevar_": self._inplacevar,
            "_apply_": lambda func, *args, **kwargs: func(*args, **kwargs),
            "_print_": _print_,
            "_getiter_": _getiter_,
            "_getitem_": _getitem_,
        }

        # Execute with timeout and output capture
        result_holder: dict[str, Any] = {"error": None, "done": False}

        def _run() -> None:
            try:
                exec(byte_code, restricted_globals)
            except Exception as e:
                result_holder["error"] = str(e)
            finally:
                result_holder["done"] = True

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if not result_holder["done"]:
            return ToolResult(
                success=False,
                error=f"Execution timed out after {timeout} seconds.",
            )

        if result_holder["error"]:
            return ToolResult(success=False, error=f"Runtime error: {result_holder['error']}")

        # RestrictedPython collects printed output in the _print collector
        print_collector = restricted_globals.get("_print")
        if print_collector is not None:
            output = str(print_collector).strip()
        else:
            output = str(restricted_globals.get("printed", "")).strip()
        if len(output) > _MAX_OUTPUT_LENGTH:
            output = output[:_MAX_OUTPUT_LENGTH] + "\n... (output truncated)"

        logger.info(
            f"Python sandbox executed: {len(code)} chars, {len(output)} output",
            extra={"component": "tools.python_sandbox", "code_length": len(code), "output_length": len(output)},
        )

        return ToolResult(
            success=True,
            output=output if output else "(no output)",
            metadata={"code_length": len(code), "timeout": timeout},
        )

    def _inplacevar(self, op, x, y):
        """Handle in-place operations safely (+=, -=, etc.)."""
        import operator
        ops = {
            "+=": operator.iadd,
            "-=": operator.isub,
            "*=": operator.imul,
            "/=": operator.itruediv,
            "//=": operator.ifloordiv,
            "%=": operator.imod,
            "**=": operator.ipow,
        }
        if isinstance(op, str):
            func = ops.get(op)
            if func is None:
                raise ValueError(f"Unsupported in-place operator: {op}")
            return func(x, y)
        return op(x, y)
