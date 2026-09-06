"""Unit tests for KGK AI deployment configuration.

Tests cover:
- Dockerfile: structure, multi-stage build, non-root user, health check
- .dockerignore: excludes sensitive files
- app.py: mode selection (ui/api), logging setup
- .env.example: all config variables present
- CI/CD: workflow file exists and has required jobs
- README: HF Space frontmatter
"""

from __future__ import annotations

import os
import re

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_file(filename: str) -> str:
    path = os.path.join(PROJECT_ROOT, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class TestDockerfile:
    """Tests for Dockerfile configuration."""

    def test_dockerfile_exists(self):
        assert os.path.exists(os.path.join(PROJECT_ROOT, "Dockerfile"))

    def test_multi_stage_build(self):
        content = _read_file("Dockerfile")
        assert "AS builder" in content
        assert "FROM python:3.12-slim" in content

    def test_non_root_user(self):
        content = _read_file("Dockerfile")
        assert "useradd" in content
        assert "USER kgk" in content

    def test_health_check(self):
        content = _read_file("Dockerfile")
        assert "HEALTHCHECK" in content

    def test_exposes_port(self):
        content = _read_file("Dockerfile")
        assert "EXPOSE 7860" in content

    def test_python_unbuffered(self):
        content = _read_file("Dockerfile")
        assert "PYTHONUNBUFFERED=1" in content

    def test_requirements_installed(self):
        content = _read_file("Dockerfile")
        assert "requirements.txt" in content
        assert "pip install" in content

    def test_libgomp_for_faiss(self):
        content = _read_file("Dockerfile")
        assert "libgomp1" in content


class TestDockerIgnore:
    """Tests for .dockerignore."""

    def test_dockerignore_exists(self):
        assert os.path.exists(os.path.join(PROJECT_ROOT, ".dockerignore"))

    def test_ignores_env_files(self):
        content = _read_file(".dockerignore")
        assert ".env" in content

    def test_ignores_pycache(self):
        content = _read_file(".dockerignore")
        assert "__pycache__" in content

    def test_ignores_git(self):
        content = _read_file(".dockerignore")
        assert ".git/" in content

    def test_ignores_model_files(self):
        content = _read_file(".dockerignore")
        assert "*.safetensors" in content
        assert "*.bin" in content

    def test_ignores_venv(self):
        content = _read_file(".dockerignore")
        assert "venv/" in content

    def test_ignores_test_artifacts(self):
        content = _read_file(".dockerignore")
        assert ".pytest_cache" in content
        assert ".coverage" in content


class TestAppEntryPoint:
    """Tests for app.py Hugging Face entry point."""

    def test_app_py_exists(self):
        assert os.path.exists(os.path.join(PROJECT_ROOT, "app.py"))

    def test_supports_ui_mode(self):
        content = _read_file("app.py")
        assert "run_ui" in content

    def test_supports_api_mode(self):
        content = _read_file("app.py")
        assert "run_api" in content
        assert "KGK_MODE" in content

    def test_sets_up_logging(self):
        content = _read_file("app.py")
        assert "setup_logging" in content

    def test_main_function(self):
        content = _read_file("app.py")
        assert "def main()" in content
        assert '__main__' in content


class TestEnvExample:
    """Tests for .env.example completeness."""

    def test_env_example_exists(self):
        assert os.path.exists(os.path.join(PROJECT_ROOT, ".env.example"))

    def test_has_model_config(self):
        content = _read_file(".env.example")
        assert "MODEL_NAME=" in content
        assert "MODEL_NAME_FALLBACK=" in content

    def test_has_generation_params(self):
        content = _read_file(".env.example")
        assert "MAX_NEW_TOKENS=" in content
        assert "TEMPERATURE=" in content
        assert "TOP_P=" in content

    def test_has_quantization(self):
        content = _read_file(".env.example")
        assert "QUANTIZATION=" in content

    def test_has_rag_config(self):
        content = _read_file(".env.example")
        assert "ENABLE_RAG=" in content
        assert "CHUNK_SIZE=" in content
        assert "TOP_K_RETRIEVAL=" in content

    def test_has_memory_config(self):
        content = _read_file(".env.example")
        assert "ENABLE_MEMORY=" in content
        assert "MEMORY_DIR=" in content
        assert "CONTEXT_WINDOW_TOKENS=" in content
        assert "ENABLE_SUMMARIZATION=" in content

    def test_has_tools_config(self):
        content = _read_file(".env.example")
        assert "ENABLE_TOOLS=" in content
        assert "ENABLE_PYTHON_TOOL=" in content

    def test_has_agents_config(self):
        content = _read_file(".env.example")
        assert "ENABLE_AGENTS=" in content

    def test_has_api_config(self):
        content = _read_file(".env.example")
        assert "API_HOST=" in content
        assert "API_PORT=" in content
        assert "ENABLE_API_AUTH=" in content
        assert "API_KEY=" in content

    def test_has_cors_config(self):
        content = _read_file(".env.example")
        assert "ENABLE_CORS=" in content
        assert "CORS_ORIGINS=" in content

    def test_has_kgk_mode(self):
        content = _read_file(".env.example")
        assert "KGK_MODE=" in content

    def test_has_hf_token(self):
        content = _read_file(".env.example")
        assert "HF_TOKEN" in content

    def test_has_security_config(self):
        content = _read_file(".env.example")
        assert "MAX_INPUT_LENGTH=" in content
        assert "MAX_CONCURRENT_REQUESTS=" in content


class TestCICD:
    """Tests for GitHub Actions CI/CD pipeline."""

    def test_workflow_file_exists(self):
        path = os.path.join(PROJECT_ROOT, ".github", "workflows", "ci.yml")
        assert os.path.exists(path)

    def test_has_lint_job(self):
        content = _read_file(".github/workflows/ci.yml")
        assert "lint" in content.lower()
        assert "ruff" in content.lower()

    def test_has_test_jobs(self):
        content = _read_file(".github/workflows/ci.yml")
        assert "test-core" in content
        assert "test-model" in content
        assert "test-rag" in content

    def test_has_docker_build_job(self):
        content = _read_file(".github/workflows/ci.yml")
        assert "docker-build" in content

    def test_has_hf_deploy_job(self):
        content = _read_file(".github/workflows/ci.yml")
        assert "deploy-hf" in content
        assert "HF_TOKEN" in content

    def test_uses_python_312(self):
        content = _read_file(".github/workflows/ci.yml")
        assert "3.12" in content

    def test_triggers_on_push_and_pr(self):
        content = _read_file(".github/workflows/ci.yml")
        assert "on:" in content
        assert "push:" in content
        assert "pull_request:" in content


class TestReadme:
    """Tests for README.md Hugging Face frontmatter."""

    def test_readme_exists(self):
        assert os.path.exists(os.path.join(PROJECT_ROOT, "README.md"))

    def test_has_hf_frontmatter(self):
        content = _read_file("README.md")
        assert content.startswith("---")
        assert "title: KGK AI" in content
        assert "sdk: gradio" in content
        assert "app_file: app.py" in content

    def test_has_emoji(self):
        content = _read_file("README.md")
        assert "emoji:" in content

    def test_has_license_section(self):
        content = _read_file("README.md")
        assert "Apache License 2.0" in content

    def test_roadmark_shows_completed_phases(self):
        content = _read_file("README.md")
        assert "[x] Model inference" in content
        assert "[x] RAG pipeline" in content
        assert "[x] Memory" in content
        assert "[x] Tools" in content
        assert "[x] Agents" in content
        assert "[x] API" in content
        assert "[x] Testing suite" in content
        assert "[x] Hugging Face deployment" in content


class TestGitignore:
    """Tests for .gitignore completeness."""

    def test_gitignore_exists(self):
        assert os.path.exists(os.path.join(PROJECT_ROOT, ".gitignore"))

    def test_ignores_env(self):
        content = _read_file(".gitignore")
        assert ".env" in content

    def test_ignores_pycache(self):
        content = _read_file(".gitignore")
        assert "__pycache__" in content

    def test_ignores_model_files(self):
        content = _read_file(".gitignore")
        assert "*.safetensors" in content
        assert "*.bin" in content

    def test_ignores_venv(self):
        content = _read_file(".gitignore")
        assert "venv/" in content

    def test_ignores_data(self):
        content = _read_file(".gitignore")
        assert "data/" in content

    def test_ignores_logs(self):
        content = _read_file(".gitignore")
        assert "logs/" in content
