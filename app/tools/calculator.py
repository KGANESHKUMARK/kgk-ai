"""KGK AI Calculator Tool — Safe mathematical expression evaluation.

Evaluates arithmetic and mathematical expressions using Python's ast module
with a restricted node set. No imports, no function calls (except safe builtins
like abs, round, min, max, sum), no attribute access.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from app.logging_config import get_logger
from app.tools.registry import BaseTool, ToolInfo, ToolResult

logger = get_logger("tools.calculator")

_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_SAFE_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

_SAFE_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "degrees": math.degrees,
    "radians": math.radians,
}

_MAX_RESULT_LENGTH = 1000


class CalculatorTool(BaseTool):
    """Safe calculator tool for mathematical expressions.

    Supports: +, -, *, /, //, %, **, parentheses, and common math functions.
    Blocks: imports, attribute access, function definitions, arbitrary calls.
    """

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="calculator",
            description="Evaluate mathematical expressions safely. Supports arithmetic, trigonometry, logarithms, and constants (pi, e).",
            parameters={
                "expression": "Mathematical expression string (e.g. '2 + 3 * 4', 'sqrt(16)', 'sin(pi/2)')",
            },
            safe_for_public=True,
        )

    def execute(self, expression: str = "", **kwargs: Any) -> ToolResult:
        """Evaluate a mathematical expression.

        Args:
            expression: Mathematical expression string.

        Returns:
            ToolResult with the computed value or error.
        """
        if not expression or not expression.strip():
            return ToolResult(success=False, error="No expression provided.")

        expression = expression.strip()

        try:
            tree = ast.parse(expression, mode="eval")
            result = self._eval_node(tree.body)
            result_str = str(result)
            if len(result_str) > _MAX_RESULT_LENGTH:
                result_str = result_str[:_MAX_RESULT_LENGTH] + "..."

            logger.info(
                f"Calculator: {expression} = {result_str}",
                extra={"component": "tools.calculator", "expression": expression},
            )

            return ToolResult(
                success=True,
                output=result_str,
                metadata={"expression": expression, "result_type": type(result).__name__},
            )
        except ZeroDivisionError:
            return ToolResult(success=False, error="Division by zero.")
        except Exception as e:
            return ToolResult(success=False, error=f"Invalid expression: {e}")

    def _eval_node(self, node: ast.AST) -> Any:
        """Recursively evaluate an AST node safely.

        Args:
            node: AST node to evaluate.

        Returns:
            Computed value.

        Raises:
            ValueError: If the node type is not allowed.
        """
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex)):
                return node.value
            raise ValueError(f"Constant type not allowed: {type(node.value)}")

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_func = _SAFE_BINOPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
            return op_func(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_func = _SAFE_UNARYOPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unary operator not allowed: {type(node.op).__name__}")
            return op_func(operand)

        elif isinstance(node, ast.Name):
            if node.id in _SAFE_CONSTANTS:
                return _SAFE_CONSTANTS[node.id]
            raise ValueError(f"Unknown variable: {node.id}")

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed.")
            func_name = node.func.id
            if func_name not in _SAFE_FUNCS:
                raise ValueError(f"Function not allowed: {func_name}")
            args = [self._eval_node(arg) for arg in node.args]
            return _SAFE_FUNCS[func_name](*args)

        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt) for elt in node.elts)
        elif isinstance(node, ast.List):
            return [self._eval_node(elt) for elt in node.elts]

        else:
            raise ValueError(f"Expression element not allowed: {type(node).__name__}")
