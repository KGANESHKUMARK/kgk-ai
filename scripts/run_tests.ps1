# KGK AI Test Runner
# On Windows, torch and faiss can crash when loaded in the same process.
# This script runs tests in separate batches to avoid the conflict.

Write-Host "=== KGK AI Test Suite ===" -ForegroundColor Cyan
Write-Host ""

$failed = $false

# Batch 1: Non-torch, non-faiss tests
Write-Host "[1/4] Running core tests (config, logging, interfaces, chat, UI, tools)..." -ForegroundColor Yellow
python -m pytest tests/unit/test_config.py tests/unit/test_logging.py tests/unit/test_base_interfaces.py tests/unit/test_chat_controller.py tests/unit/test_ui.py tests/unit/test_tools.py -q --tb=short
if ($LASTEXITCODE -ne 0) { $failed = $true }

# Batch 2: Torch tests (model layer)
Write-Host ""
Write-Host "[2/4] Running model layer tests (requires torch)..." -ForegroundColor Yellow
python -m pytest tests/unit/test_model_layer.py -q --tb=short
if ($LASTEXITCODE -ne 0) { $failed = $true }

# Batch 3: FAISS tests (RAG)
Write-Host ""
Write-Host "[3/4] Running RAG tests (requires faiss)..." -ForegroundColor Yellow
python -m pytest tests/unit/test_rag.py -q --tb=short
if ($LASTEXITCODE -ne 0) { $failed = $true }

Write-Host ""
Write-Host "[4/4] Summary" -ForegroundColor Yellow
if ($failed) {
    Write-Host "=== SOME TESTS FAILED ===" -ForegroundColor Red
    exit 1
} else {
    Write-Host "=== ALL TESTS PASSED ===" -ForegroundColor Green
    exit 0
}
