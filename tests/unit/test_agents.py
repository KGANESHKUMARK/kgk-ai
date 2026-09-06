"""Unit tests for KGK AI agent components.

Tests cover:
- ReActAgent: response parsing, tool execution, multi-step loop, edge cases
- TaskPlanner: single/multi-step planning, parsing, fallback
- AgentExecutor: single-step, multi-step, result combination, error handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.agents.base import AgentResult, AgentTask, BaseAgent
from app.agents.react import ReActAgent
from app.agents.planner import TaskPlanner, TaskPlan
from app.agents.executor import AgentExecutor
from app.tools.registry import ToolRegistry, ToolResult


class TestReActAgentParsing:
    """Tests for ReActAgent response parsing."""

    def setup_method(self):
        self.agent = ReActAgent(model=MagicMock(), tools=MagicMock())

    def test_parse_answer(self):
        response = "THOUGHT: I have enough info.\nANSWER: The result is 42."
        parsed = self.agent._parse_response(response)
        assert parsed["answer"] == "The result is 42."
        assert parsed["action"] is None

    def test_parse_action(self):
        response = "THOUGHT: I need to calculate.\nACTION: calculator\nACTION_INPUT: expression=2+3*4"
        parsed = self.agent._parse_response(response)
        assert parsed["answer"] is None
        assert parsed["action"] == "calculator"
        assert parsed["action_input"] == {"expression": "2+3*4"}

    def test_parse_action_with_int_input(self):
        response = "THOUGHT: Search docs.\nACTION: document_search\nACTION_INPUT: query=test, top_k=3"
        parsed = self.agent._parse_response(response)
        assert parsed["action"] == "document_search"
        assert parsed["action_input"]["query"] == "test"
        assert parsed["action_input"]["top_k"] == 3
        assert isinstance(parsed["action_input"]["top_k"], int)

    def test_parse_action_with_float_input(self):
        response = "THOUGHT: Calculate.\nACTION: calculator\nACTION_INPUT: expression=3.14"
        parsed = self.agent._parse_response(response)
        assert parsed["action_input"]["expression"] == 3.14
        assert isinstance(parsed["action_input"]["expression"], float)

    def test_parse_action_with_bool_input(self):
        response = "THOUGHT: Search.\nACTION: search\nACTION_INPUT: verbose=true, cached=false"
        parsed = self.agent._parse_response(response)
        assert parsed["action_input"]["verbose"] is True
        assert parsed["action_input"]["cached"] is False

    def test_parse_thought(self):
        response = "THOUGHT: I should calculate first.\nACTION: calculator\nACTION_INPUT: expression=1+1"
        parsed = self.agent._parse_response(response)
        assert "calculate" in parsed["thought"]

    def test_parse_no_action_no_answer(self):
        response = "I'm not sure what to do."
        parsed = self.agent._parse_response(response)
        assert parsed["answer"] is None
        assert parsed["action"] is None

    def test_parse_action_input_with_quotes(self):
        response = 'THOUGHT: Search.\nACTION: search\nACTION_INPUT: query="hello, world"'
        parsed = self.agent._parse_response(response)
        assert parsed["action_input"]["query"] == "hello, world"

    def test_parse_single_expression_without_key(self):
        response = "THOUGHT: Calculate.\nACTION: calculator\nACTION_INPUT: 2 + 3 * 4"
        parsed = self.agent._parse_response(response)
        assert parsed["action_input"] == {"expression": "2 + 3 * 4"}


class TestReActAgentExecution:
    """Tests for ReActAgent execution loop."""

    def test_no_model_returns_error(self):
        agent = ReActAgent(model=None, tools=None)
        task = AgentTask(query="test")
        result = agent.execute(task)
        assert result.success is False
        assert "No model" in result.error

    def test_immediate_answer(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(
            text="THOUGHT: I know this.\nANSWER: Paris is the capital of France."
        )
        agent = ReActAgent(model=mock_model, tools=None)
        result = agent.execute(AgentTask(query="What is the capital of France?"))

        assert result.success is True
        assert result.response == "Paris is the capital of France."
        assert result.tools_used == []

    def test_tool_then_answer(self):
        mock_model = MagicMock()
        mock_model.generate.side_effect = [
            MagicMock(text="THOUGHT: Need to calculate.\nACTION: calculator\nACTION_INPUT: expression=2+3"),
            MagicMock(text="THOUGHT: Got the result.\nANSWER: The answer is 5."),
        ]

        mock_tools = MagicMock()
        mock_tools.list_tools.return_value = []
        mock_tools.execute_tool.return_value = ToolResult(success=True, output="5")

        agent = ReActAgent(model=mock_model, tools=mock_tools)
        result = agent.execute(AgentTask(query="What is 2+3?"))

        assert result.success is True
        assert result.response == "The answer is 5."
        assert result.tools_used == ["calculator"]

    def test_max_steps_exceeded(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(
            text="THOUGHT: I need more info.\nACTION: search\nACTION_INPUT: query=test"
        )

        mock_tools = MagicMock()
        mock_tools.list_tools.return_value = []
        mock_tools.execute_tool.return_value = ToolResult(success=True, output="result")

        agent = ReActAgent(model=mock_model, tools=mock_tools, max_steps=3)
        result = agent.execute(AgentTask(query="complex query"))

        assert result.success is False
        assert result.error == "max_steps_exceeded"
        assert result.tools_used == ["search"]

    def test_tool_error_continues_loop(self):
        mock_model = MagicMock()
        mock_model.generate.side_effect = [
            MagicMock(text="THOUGHT: Try tool.\nACTION: bad_tool\nACTION_INPUT: x=1"),
            MagicMock(text="THOUGHT: Tool failed, answer directly.\nANSWER: I'll answer without the tool."),
        ]

        mock_tools = MagicMock()
        mock_tools.list_tools.return_value = []
        mock_tools.execute_tool.return_value = ToolResult(success=False, error="Tool not found")

        agent = ReActAgent(model=mock_model, tools=mock_tools)
        result = agent.execute(AgentTask(query="test"))

        assert result.success is True
        assert result.response == "I'll answer without the tool."

    def test_model_generation_error(self):
        mock_model = MagicMock()
        mock_model.generate.side_effect = Exception("Model crashed")

        agent = ReActAgent(model=mock_model, tools=None)
        result = agent.execute(AgentTask(query="test"))

        assert result.success is False
        assert "Model generation failed" in result.error

    def test_sources_collected_from_tool_metadata(self):
        mock_model = MagicMock()
        mock_model.generate.side_effect = [
            MagicMock(text="THOUGHT: Search docs.\nACTION: document_search\nACTION_INPUT: query=test"),
            MagicMock(text="THOUGHT: Found info.\nANSWER: Here is the answer."),
        ]

        mock_tools = MagicMock()
        mock_tools.list_tools.return_value = []
        mock_tools.execute_tool.return_value = ToolResult(
            success=True,
            output="Found relevant docs",
            metadata={"sources": ["doc1.pdf", "doc2.pdf"]},
        )

        agent = ReActAgent(model=mock_model, tools=mock_tools)
        result = agent.execute(AgentTask(query="search for info"))

        assert result.success is True
        assert "doc1.pdf" in result.sources
        assert "doc2.pdf" in result.sources

    def test_get_tool_descriptions_no_tools(self):
        agent = ReActAgent(model=None, tools=None)
        desc = agent._get_tool_descriptions()
        assert "no tools" in desc

    def test_get_tool_descriptions_with_tools(self):
        mock_tools = MagicMock()
        from app.tools.registry import ToolInfo
        mock_tools.list_tools.return_value = [
            ToolInfo(name="calculator", description="Math tool", parameters={"expression": "expr"}),
        ]
        agent = ReActAgent(model=None, tools=mock_tools)
        desc = agent._get_tool_descriptions()
        assert "calculator" in desc
        assert "Math tool" in desc


class TestTaskPlanner:
    """Tests for TaskPlanner."""

    def test_no_model_returns_single_step(self):
        planner = TaskPlanner(model=None)
        plan = planner.plan("What is 2+2?")
        assert plan.is_multi_step is False
        assert len(plan.subtasks) == 1
        assert plan.subtasks[0] == "What is 2+2?"

    def test_model_single_step(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(
            text="PLAN: single\nSUBTASKS: none"
        )
        planner = TaskPlanner(model=mock_model)
        plan = planner.plan("What is the capital of France?")
        assert plan.is_multi_step is False
        assert len(plan.subtasks) == 0

    def test_model_multi_step(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(
            text="PLAN: multi\nSUBTASKS:\n1. Calculate the total revenue\n2. Find the growth rate\n3. Compare with last year"
        )
        planner = TaskPlanner(model=mock_model)
        plan = planner.plan("Analyze revenue growth")
        assert plan.is_multi_step is True
        assert len(plan.subtasks) == 3
        assert "Calculate the total revenue" in plan.subtasks[0]

    def test_model_error_falls_back(self):
        mock_model = MagicMock()
        mock_model.generate.side_effect = Exception("Model error")
        planner = TaskPlanner(model=mock_model)
        plan = planner.plan("test query")
        assert plan.is_multi_step is False
        assert plan.subtasks == ["test query"]

    def test_multi_step_with_dash_format(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(
            text="PLAN: multi\nSUBTASKS:\n- First task\n- Second task"
        )
        planner = TaskPlanner(model=mock_model)
        plan = planner.plan("complex query")
        assert plan.is_multi_step is True
        assert len(plan.subtasks) == 2

    def test_max_five_subtasks(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(
            text="PLAN: multi\nSUBTASKS:\n1. Task one\n2. Task two\n3. Task three\n4. Task four\n5. Task five\n6. Task six"
        )
        planner = TaskPlanner(model=mock_model)
        plan = planner.plan("very complex query")
        assert len(plan.subtasks) == 5

    def test_empty_subtasks_returns_single(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(
            text="PLAN: multi\nSUBTASKS:\nnone"
        )
        planner = TaskPlanner(model=mock_model)
        plan = planner.plan("query")
        assert plan.is_multi_step is False


class TestAgentExecutor:
    """Tests for AgentExecutor."""

    def test_single_step_execution(self):
        mock_agent = MagicMock()
        mock_agent.execute.return_value = AgentResult(
            success=True,
            response="Direct answer",
            tools_used=["calculator"],
        )
        mock_planner = MagicMock()
        mock_planner.plan.return_value = TaskPlan(is_multi_step=False)

        executor = AgentExecutor(agent=mock_agent, planner=mock_planner, model=None)
        result = executor.run("What is 2+2?")

        assert result.success is True
        assert result.response == "Direct answer"
        assert result.tools_used == ["calculator"]
        mock_agent.execute.assert_called_once()

    def test_multi_step_execution(self):
        mock_agent = MagicMock()
        mock_agent.execute.side_effect = [
            AgentResult(success=True, response="Result 1", tools_used=["calculator"]),
            AgentResult(success=True, response="Result 2", tools_used=["document_search"]),
        ]
        mock_planner = MagicMock()
        mock_planner.plan.return_value = TaskPlan(
            is_multi_step=True,
            subtasks=["Calculate something", "Search for info"],
        )
        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(text="Combined final answer")

        executor = AgentExecutor(agent=mock_agent, planner=mock_planner, model=mock_model)
        result = executor.run("Complex query")

        assert result.success is True
        assert result.response == "Combined final answer"
        assert "calculator" in result.tools_used
        assert "document_search" in result.tools_used
        assert mock_agent.execute.call_count == 2

    def test_multi_step_without_model_concatenates(self):
        mock_agent = MagicMock()
        mock_agent.execute.side_effect = [
            AgentResult(success=True, response="Result 1"),
            AgentResult(success=True, response="Result 2"),
        ]
        mock_planner = MagicMock()
        mock_planner.plan.return_value = TaskPlan(
            is_multi_step=True,
            subtasks=["Task 1", "Task 2"],
        )

        executor = AgentExecutor(agent=mock_agent, planner=mock_planner, model=None)
        result = executor.run("Complex query")

        assert result.success is True
        assert "Result 1" in result.response
        assert "Result 2" in result.response

    def test_multi_step_with_failed_subtask(self):
        mock_agent = MagicMock()
        mock_agent.execute.side_effect = [
            AgentResult(success=False, error="tool_error", response=""),
            AgentResult(success=True, response="Result 2"),
        ]
        mock_planner = MagicMock()
        mock_planner.plan.return_value = TaskPlan(
            is_multi_step=True,
            subtasks=["Task 1", "Task 2"],
        )
        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(text="Combined answer with partial info")

        executor = AgentExecutor(agent=mock_agent, planner=mock_planner, model=mock_model)
        result = executor.run("Complex query")

        assert result.success is True
        assert result.response == "Combined answer with partial info"

    def test_default_agent_created(self):
        executor = AgentExecutor(model=None)
        assert executor.agent is not None
        assert isinstance(executor.agent, ReActAgent)
        assert executor.planner is not None
        assert isinstance(executor.planner, TaskPlanner)

    def test_sources_collected_across_subtasks(self):
        mock_agent = MagicMock()
        mock_agent.execute.side_effect = [
            AgentResult(success=True, response="R1", sources=["doc1.pdf"]),
            AgentResult(success=True, response="R2", sources=["doc2.pdf", "doc3.pdf"]),
        ]
        mock_planner = MagicMock()
        mock_planner.plan.return_value = TaskPlan(
            is_multi_step=True,
            subtasks=["Task 1", "Task 2"],
        )
        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(text="Combined")

        executor = AgentExecutor(agent=mock_agent, planner=mock_planner, model=mock_model)
        result = executor.run("Complex query")

        assert "doc1.pdf" in result.sources
        assert "doc2.pdf" in result.sources
        assert "doc3.pdf" in result.sources

    def test_single_result_returns_directly(self):
        mock_agent = MagicMock()
        mock_agent.execute.return_value = AgentResult(
            success=True,
            response="Single result",
        )
        mock_planner = MagicMock()
        mock_planner.plan.return_value = TaskPlan(
            is_multi_step=True,
            subtasks=["Only task"],
        )

        executor = AgentExecutor(agent=mock_agent, planner=mock_planner, model=None)
        result = executor.run("query")

        assert result.success is True
        assert "Single result" in result.response
