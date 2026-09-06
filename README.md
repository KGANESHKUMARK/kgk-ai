---
title: KGK AI
emoji: 🤖
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
---

# KGK AI

**AI Intelligence by KGK**

KGK AI is a modular, open-source-friendly AI platform built under the KGK brand. It starts with an open foundation model and is enhanced with KGK personality, knowledge base, RAG, memory, tools, and agent capabilities.

> **Note:** KGK AI is built on open-source foundation models. It does not claim to have created a foundation model from scratch. The runtime model is configurable and replaceable.

---

## Architecture

```
                     KGK AI
                        |
               KGK Application Layer
                        |
              KGK AI Controller
                        |
         +--------------+--------------+
         |              |              |
    Model Layer     RAG Layer     Memory Layer
    (BaseProvider)  (Pipeline)    (Manager)
         |              |              |
         +--------------+--------------+
                |
          Tool System (Registry)
                |
          Agent System (Base + Pluggable)
                |
           KGK API (FastAPI)
                |
          Gradio UI (Hugging Face Space)
                |
          ZeroGPU Deployment
```

### Key Design Principles

- **Model-agnostic**: `BaseModelProvider` abstracts all model interactions. The rest of the system never imports a specific model library directly.
- **Modular**: Each component (RAG, memory, tools, agents) is independent and replaceable.
- **Configurable**: All settings via environment variables (`.env`).
- **Graceful degradation**: Every component fails gracefully with user-friendly messages.
- **Secure**: No secrets in code, sandboxed tools, input validation.

---

## Features

- **KGK Chat**: Conversational AI with KGK personality
- **KGK Personality**: Intelligent, practical, honest, friendly system prompt
- **KGK Knowledge Base**: Document storage for RAG
- **KGK RAG**: Retrieval-augmented generation (TXT, MD, PDF, JSON)
- **KGK Memory**: Short-term (conversation) and long-term (persistent) memory
- **KGK Tools**: Calculator, sandboxed Python, document retrieval
- **KGK Agents**: Extensible agent architecture (researcher, coder, data science)
- **KGK Vision**: Image processing capability (future)
- **KGK Model Layer**: Pluggable model providers (local, API, future)
- **KGK API**: FastAPI-based REST API
- **KGK Web UI**: Gradio interface with streaming, source display, settings
- **Hugging Face Deployment**: ZeroGPU Space compatible

---

## Installation

### Prerequisites

- Python 3.10+ (3.12 recommended for Hugging Face ZeroGPU)
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/kgk-ai.git
cd kgk-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements-dev.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

---

## Local Development

### Running the UI

```bash
python -m app.main
# Opens Gradio UI at http://localhost:7860
```

### Running the API

```bash
python -m app.main --api
# FastAPI server at http://localhost:7860
# API docs at http://localhost:7860/docs
```

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
ruff check app/
black app/
```

---

## Environment Variables

All configuration is via environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` | Hugging Face model ID |
| `MODEL_NAME_FALLBACK` | `Qwen/Qwen3-0.6B` | Fallback for low-resource |
| `MAX_NEW_TOKENS` | `1024` | Max tokens to generate |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `TOP_P` | `0.9` | Nucleus sampling |
| `QUANTIZATION` | `4bit` | Quantization mode (none/4bit/8bit) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `ENABLE_RAG` | `true` | Enable RAG pipeline |
| `ENABLE_MEMORY` | `true` | Enable memory system |
| `CONTEXT_WINDOW_TOKENS` | `4096` | Max tokens for context window |
| `ENABLE_SUMMARIZATION` | `true` | Summarize old messages |
| `ENABLE_TOOLS` | `true` | Enable tool system |
| `ENABLE_AGENTS` | `false` | Enable agent system |
| `ENABLE_API_AUTH` | `false` | Enable API key authentication |
| `API_KEY` | (unset) | API key (required when auth enabled) |
| `ENABLE_CORS` | `true` | Enable CORS middleware |
| `KGK_MODE` | `ui` | Deployment mode (ui/api) |
| `CHUNK_SIZE` | `512` | RAG chunk size (tokens) |
| `TOP_K_RETRIEVAL` | `5` | RAG retrieval count |
| `HF_TOKEN` | (unset) | Hugging Face token (for private models) |
| `MAX_INPUT_LENGTH` | `10000` | Max input characters (security) |

---

## RAG Usage

### Ingesting Documents

Place documents in `knowledge/documents/` (supports TXT, MD, PDF, JSON), then run:

```bash
python scripts/ingest.py
```

### How RAG Works

1. Documents are loaded from `knowledge/documents/`
2. Text is extracted and cleaned
3. Text is chunked (default: 512 tokens, 50 overlap)
4. Chunks are embedded using the embedding model
5. Embeddings are stored in FAISS vector store
6. On query, relevant chunks are retrieved and provided as context
7. The LLM generates an answer with source citations

### Source Display

Every RAG response includes source citations:
- Source file name
- Chunk ID
- Document ID
- Timestamp

---

## Memory

### Short-Term Memory

- Current conversation history
- In-memory storage
- Cleared when conversation ends

### Long-Term Memory

- Persistent across sessions (file-based for v1)
- Only stores explicitly useful information
- Does not store sensitive personal information by default

### Memory API

```python
from app.memory.manager import MemoryManager

manager.save_memory("important fact", conversation_id="user123", long_term=True)
items = manager.retrieve_memory("user123", include_long_term=True)
manager.delete_memory("user123")
manager.clear_memory()
```

---

## Tools

### Available Tools

| Tool | Description | Safe for Public |
|---|---|---|
| `calculator` | Mathematical expression evaluation | Yes |
| `python_tool` | Sandboxed Python execution (RestrictedPython) | Yes |
| `search` | Document retrieval from knowledge base | Yes |

### Tool API

```python
from app.tools.registry import tool_registry

# List tools
tools = tool_registry.list_tools(public_only=True)

# Execute a tool
result = tool_registry.execute_tool("calculator", expression="2 + 2", public_context=True)
```

### Security

- Python tool uses RestrictedPython (no file access, no imports, no network)
- Execution time limited (default: 5 seconds)
- No arbitrary shell execution for public users

---

## Agents

### Available Agents

| Agent | Description |
|---|---|
| `Planner` | Task classification and agent routing |
| `ResearchAgent` | Research using RAG and web |
| `CodingAgent` | Code generation and execution |
| `DataScienceAgent` | Data analysis and visualization |

### Agent Flow

```
User → KGK Controller → Task classification → Agent selection
→ Tool usage → Result synthesis → KGK response
```

---

## Testing

### Test Structure

```
tests/
├── unit/          # Unit tests per module
├── integration/   # Cross-component integration tests
└── evaluation/    # Quality evaluation suite
```

### Running Tests

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# With coverage
pytest tests/ --cov=app --cov-report=html
```

### Evaluation Suite

```bash
python scripts/evaluate.py
```

Tests: basic conversation, KGK personality, RAG retrieval, hallucination behavior, tool execution, memory, streaming, error handling, long/empty prompts, unsupported requests, model unavailable, missing env vars.

---

## Hugging Face Deployment

### Prerequisites

- Hugging Face account (free or PRO)
- For ZeroGPU: account in good standing (verified email, >30 days old)

### Steps

1. Create a new Space on Hugging Face (SDK: Gradio, Hardware: ZeroGPU)
2. Push this repository to the Space's Git repo
3. Set environment variables in Space Settings → Variables and secrets
4. The Space will automatically build from `app.py` and `requirements.txt`

### Hugging Face Space README

The Space README should contain this YAML frontmatter:

```yaml
---
title: KGK AI
emoji: 🤖
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
---
```

### Free/Low-Cost Limitations

| Limitation | Impact |
|---|---|
| 5 min/day GPU quota (free visitors) | Heavy users will hit quota |
| Cold start delay (10-30s) | First request after idle |
| Space sleeps when idle | Needs wake-up request |
| No persistent disk | Vector store rebuilt on restart |
| Gradio-only on ZeroGPU | Cannot use vLLM/SGLang |

### Mitigation

- 4-bit quantization reduces per-request GPU time
- Vector store rebuilds from `knowledge/` on startup
- Model is configurable — larger models can be used with dedicated GPU

---

## Model Configuration

### Current Model

| Property | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Parameters | 4.0B |
| License | Apache 2.0 |
| Context | 262,144 tokens |
| VRAM (4-bit) | ~3 GB |
| VRAM (fp16) | ~8 GB |

### Fallback Model (CPU)

| Property | Value |
|---|---|
| Model | `Qwen/Qwen3-0.6B` |
| Parameters | 0.6B |
| License | Apache 2.0 |
| Context | 32,768 tokens |

### Embedding Model

| Property | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| License | Apache 2.0 |
| Size | ~90 MB |

### License Compliance

All chosen models are Apache 2.0 licensed, allowing commercial use, modification, and redistribution with attribution. No MAU (Monthly Active User) restrictions apply.

---

## Fine-Tuning

Fine-tuning is optional and should only be attempted after the baseline system works.

See `training/README.md` for:
- Dataset preparation
- Training commands (LoRA/QLoRA)
- Evaluation
- Model export
- Hugging Face upload

> **Important:** Fine-tuning does not automatically give the model new factual knowledge. Use RAG for frequently changing knowledge.

---

## Roadmap

### v1 (Current)

- [x] Architecture & repository setup
- [x] Model inference (transformers, streaming, quantization)
- [x] KGK chat with personality
- [x] Gradio UI with streaming
- [x] RAG pipeline (FAISS, document ingestion, source citations)
- [x] Memory (short-term + long-term, context window, summarization)
- [x] Tools (calculator, Python sandbox, document search)
- [x] Agents (ReAct loop, task planner, agent executor)
- [x] API (FastAPI REST, WebSocket streaming, API key auth, CORS)
- [x] Testing suite (290+ unit tests)
- [x] Hugging Face deployment (ZeroGPU, Docker, CI/CD)
- [ ] Fine-tuning pipeline
- [ ] Production hardening

### v2 (Future)

- [ ] Better RAG (re-ranking, hybrid search)
- [ ] Semantic memory search
- [ ] Agent router (multi-agent)
- [ ] Research agent
- [ ] Coding agent
- [ ] Data science agent
- [ ] Vision support
- [ ] Evaluation benchmark
- [ ] Fine-tuning dataset
- [ ] LoRA/QLoRA pipeline
- [ ] KGK fine-tuned model

### v3+ (Future)

- [ ] Dedicated GPU infrastructure
- [ ] Cloud inference
- [ ] Distributed vector databases
- [ ] Authentication & user accounts
- [ ] Mobile app
- [ ] Billing

---

## License

Apache License 2.0 — See [LICENSE](LICENSE) for details.

Copyright 2026 KGK (Siddharitha Technologies Private Limited)

---

## Credits

KGK AI is built on open-source foundation models and libraries:

- [Qwen3](https://huggingface.co/Qwen) — Foundation model (Apache 2.0)
- [Sentence Transformers](https://huggingface.co/sentence-transformers) — Embedding model (Apache 2.0)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — Model inference
- [FAISS](https://github.com/facebookresearch/faiss) — Vector search
- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [Gradio](https://gradio.app/) — UI framework
