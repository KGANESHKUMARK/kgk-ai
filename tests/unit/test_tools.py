"""Unit tests for KGK AI tools.

Tests cover:
- CalculatorTool: arithmetic, functions, constants, error handling, safety
- PythonSandboxTool: basic execution, output capture, timeout, safety
- DocumentSearchTool: search with mocked RAG, empty results, not initialized
- ToolRegistry: registration of default tools
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.tools.registry import ToolRegistry, ToolResult, ToolInfo, tool_registry
from app.tools.calculator import CalculatorTool
from app.tools.python_sandbox import PythonSandboxTool
from app.tools.doc_search import DocumentSearchTool


class TestCalculatorTool:
    """Tests for CalculatorTool."""

    def test_basic_arithmetic(self):
        tool = CalculatorTool()
        result = tool.execute(expression="2 + 3")
        assert result.success
        assert result.output == "5"

    def test_order_of_operations(self):
        tool = CalculatorTool()
        result = tool.execute(expression="2 + 3 * 4")
        assert result.success
        assert result.output == "14"

    def test_parentheses(self):
        tool = CalculatorTool()
        result = tool.execute(expression="(2 + 3) * 4")
        assert result.success
        assert result.output == "20"

    def test_division(self):
        tool = CalculatorTool()
        result = tool.execute(expression="10 / 4")
        assert result.success
        assert result.output == "2.5"

    def test_floor_division(self):
        tool = CalculatorTool()
        result = tool.execute(expression="10 // 3")
        assert result.success
        assert result.output == "3"

    def test_modulo(self):
        tool = CalculatorTool()
        result = tool.execute(expression="10 % 3")
        assert result.success
        assert result.output == "1"

    def test_power(self):
        tool = CalculatorTool()
        result = tool.execute(expression="2 ** 10")
        assert result.success
        assert result.output == "1024"

    def test_negative_numbers(self):
        tool = CalculatorTool()
        result = tool.execute(expression="-5 + 3")
        assert result.success
        assert result.output == "-2"

    def test_sqrt_function(self):
        tool = CalculatorTool()
        result = tool.execute(expression="sqrt(16)")
        assert result.success
        assert float(result.output) == 4.0

    def test_trig_functions(self):
        tool = CalculatorTool()
        result = tool.execute(expression="sin(pi / 2)")
        assert result.success
        assert float(result.output) == 1.0

    def test_constants(self):
        tool = CalculatorTool()
        result = tool.execute(expression="pi")
        assert result.success
        assert float(result.output) == 3.141592653589793

    def test_nested_functions(self):
        tool = CalculatorTool()
        result = tool.execute(expression="sqrt(abs(-16))")
        assert result.success
        assert float(result.output) == 4.0

    def test_min_max(self):
        tool = CalculatorTool()
        result = tool.execute(expression="max(1, 2, 3)")
        assert result.success
        assert result.output == "3"

    def test_division_by_zero(self):
        tool = CalculatorTool()
        result = tool.execute(expression="1 / 0")
        assert not result.success
        assert "Division by zero" in result.error

    def test_empty_expression(self):
        tool = CalculatorTool()
        result = tool.execute(expression="")
        assert not result.success
        assert "No expression" in result.error

    def test_invalid_expression(self):
        tool = CalculatorTool()
        result = tool.execute(expression="2 + + +")
        assert not result.success
        assert "Invalid" in result.error

    def test_blocks_imports(self):
        """Imports are not allowed."""
        tool = CalculatorTool()
        result = tool.execute(expression="__import__('os')")
        assert not result.success

    def test_blocks_attribute_access(self):
        """Attribute access is not allowed."""
        tool = CalculatorTool()
        result = tool.execute(expression="(1).__class__")
        assert not result.success

    def test_blocks_variable_assignment(self):
        """Variable assignment is not allowed in eval mode."""
        tool = CalculatorTool()
        result = tool.execute(expression="x = 5")
        assert not result.success

    def test_info(self):
        tool = CalculatorTool()
        info = tool.info()
        assert info.name == "calculator"
        assert info.safe_for_public is True
        assert "expression" in info.parameters

    def test_metadata(self):
        tool = CalculatorTool()
        result = tool.execute(expression="2 + 2")
        assert result.metadata["expression"] == "2 + 2"
        assert "result_type" in result.metadata


class TestPythonSandboxTool:
    """Tests for PythonSandboxTool."""

    def test_basic_execution(self):
        tool = PythonSandboxTool()
        result = tool.execute(code="print('Hello, KGK!')")
        assert result.success
        assert "Hello, KGK!" in result.output

    def test_arithmetic(self):
        tool = PythonSandboxTool()
        result = tool.execute(code="x = 5\ny = 10\nprint(x + y)")
        assert result.success
        assert "15" in result.output

    def test_loops(self):
        tool = PythonSandboxTool()
        result = tool.execute(code="total = 0\nfor i in range(5):\n    total += i\nprint(total)")
        assert result.success
        assert "10" in result.output

    def test_list_operations(self):
        tool = PythonSandboxTool()
        result = tool.execute(code="nums = [3, 1, 4, 1, 5]\nprint(sorted(nums))")
        assert result.success
        assert "[1, 1, 3, 4, 5]" in result.output

    def test_empty_code(self):
        tool = PythonSandboxTool()
        result = tool.execute(code="")
        assert not result.success
        assert "No code" in result.error

    def test_no_output(self):
        tool = PythonSandboxTool()
        result = tool.execute(code="x = 5")
        assert result.success
        assert "no output" in result.output.lower()

    def test_runtime_error(self):
        tool = PythonSandboxTool()
        result = tool.execute(code="print(undefined_var)")
        assert not result.success
        assert "Runtime error" in result.error

    def test_info(self):
        tool = PythonSandboxTool()
        info = tool.info()
        assert info.name == "python_sandbox"
        assert info.safe_for_public is False
        assert "code" in info.parameters

    def test_blocks_imports(self):
        """Imports should be blocked by RestrictedPython."""
        tool = PythonSandboxTool()
        result = tool.execute(code="import os\nos.system('echo hacked')")
        assert not result.success

    def test_timeout(self):
        """Infinite loop should timeout."""
        tool = PythonSandboxTool()
        result = tool.execute(code="while True:\n    pass")
        assert not result.success
        assert "timed out" in result.error.lower()


class TestDocumentSearchTool:
    """Tests for DocumentSearchTool."""

    def test_info(self):
        tool = DocumentSearchTool()
        info = tool.info()
        assert info.name == "document_search"
        assert info.safe_for_public is True
        assert "query" in info.parameters

    def test_empty_query(self):
        tool = DocumentSearchTool()
        result = tool.execute(query="")
        assert not result.success
        assert "No search query" in result.error

    def test_not_initialized(self):
        """Search when RAG is not ready returns helpful error."""
        tool = DocumentSearchTool(pipeline=MagicMock())
        tool._pipeline.is_ready.return_value = False

        result = tool.execute(query="test query")
        assert not result.success
        assert "not initialized" in result.error

    def test_successful_search(self):
        """Search with mocked RAG returns formatted results."""
        from app.rag.pipeline import RAGResponse
        from app.rag.vector_store import DocumentChunk

        mock_pipeline = MagicMock()
        mock_pipeline.is_ready.return_value = True
        mock_pipeline.retrieve.return_value = RAGResponse(
            context="Test content",
            sources=["doc1.txt:doc1_0000"],
            chunks=[DocumentChunk(
                content="Important KGK information",
                filename="doc1.txt",
                chunk_id="doc1_0000",
            )],
            scores=[0.95],
        )

        tool = DocumentSearchTool(pipeline=mock_pipeline)
        result = tool.execute(query="KGK information")

        assert result.success
        assert "Important KGK information" in result.output
        assert "doc1.txt" in result.output
        assert result.metadata["results"] == 1

    def test_no_results(self):
        """Search with no results returns graceful message."""
        from app.rag.pipeline import RAGResponse

        mock_pipeline = MagicMock()
        mock_pipeline.is_ready.return_value = True
        mock_pipeline.retrieve.return_value = RAGResponse()

        tool = DocumentSearchTool(pipeline=mock_pipeline)
        result = tool.execute(query="nonexistent topic")

        assert result.success
        assert "No relevant documents" in result.output

    def test_search_error(self):
        """Search handles RAG errors gracefully."""
        mock_pipeline = MagicMock()
        mock_pipeline.is_ready.return_value = True
        mock_pipeline.retrieve.side_effect = Exception("RAG error")

        tool = DocumentSearchTool(pipeline=mock_pipeline)
        result = tool.execute(query="test")

        assert not result.success
        assert "Search failed" in result.error


class TestToolRegistryIntegration:
    """Tests for tool registry integration with built-in tools."""

    def test_register_calculator(self):
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())

        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "calculator"

    def test_register_multiple_tools(self):
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        registry.register_tool(DocumentSearchTool(pipeline=MagicMock()))

        tools = registry.list_tools()
        assert len(tools) == 2

    def test_execute_calculator_via_registry(self):
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())

        result = registry.execute_tool("calculator", expression="2 + 2")
        assert result.success
        assert result.output == "4"

    def test_public_context_blocks_sandbox(self):
        registry = ToolRegistry()
        registry.register_tool(PythonSandboxTool())

        result = registry.execute_tool("python_sandbox", public_context=True, code="print('hi')")
        assert not result.success
        assert "not available" in result.error

    def test_public_context_allows_calculator(self):
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())

        result = registry.execute_tool("calculator", public_context=True, expression="1 + 1")
        assert result.success

    def test_nonexistent_tool(self):
        registry = ToolRegistry()
        result = registry.execute_tool("nonexistent")
        assert not result.success
        assert "not found" in result.error
