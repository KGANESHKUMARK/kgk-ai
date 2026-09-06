"""KGK AI Gradio UI — Main interface.

Provides a chat interface with streaming, source citations, model info,
and settings display. Designed for both local development and Hugging Face
ZeroGPU Spaces deployment.
"""

from __future__ import annotations

from typing import Any, Generator, Optional

from app.config import get_settings
from app.logging_config import get_logger, setup_logging
from app.chat.session import get_session_manager
from app.chat.controller import KGKChatController
from app.models.registry import registry as model_registry

logger = get_logger("ui")


def _get_model_info_text() -> str:
    """Build a markdown string with current model information.

    Returns:
        Markdown-formatted model info string.
    """
    settings = get_settings()
    lines = [
        "### Model Configuration",
        f"- **Model**: `{settings.model_name}`",
        f"- **Fallback**: `{settings.model_name_fallback}`",
        f"- **Quantization**: `{settings.quantization}`",
        f"- **Device**: `{settings.device}`",
        f"- **Embedding**: `{settings.embedding_model}`",
        f"- **Max Tokens**: {settings.max_new_tokens}",
        f"- **Temperature**: {settings.temperature}",
    ]

    info = model_registry.get_info()
    if info is not None:
        lines.append("")
        lines.append("### Active Model Status")
        lines.append(f"- **Loaded**: {'Yes' if info.loaded else 'No'}")
        lines.append(f"- **Provider**: `{info.provider}`")
        lines.append(f"- **Context Length**: {info.context_length:,} tokens")

    lines.append("")
    lines.append("### Features")
    lines.append(f"- **RAG**: {'Enabled' if settings.enable_rag else 'Disabled'}")
    lines.append(f"- **Memory**: {'Enabled' if settings.enable_memory else 'Disabled'}")
    lines.append(f"- **Tools**: {'Enabled' if settings.enable_tools else 'Disabled'}")
    lines.append(f"- **Agents**: {'Enabled' if settings.enable_agents else 'Disabled'}")

    if settings.is_hf_space:
        lines.append("")
        lines.append("### Environment")
        lines.append("- **Hugging Face Space**: Yes")
        lines.append(f"- **ZeroGPU**: {'Yes' if settings.is_zero_gpu else 'No'}")

    return "\n".join(lines)


def _get_about_text() -> str:
    """Build the about/description markdown.

    Returns:
        Markdown-formatted about string.
    """
    return """\
# KGK AI

**AI Intelligence by KGK**

KGK AI is built on open-source foundation models and enhanced with KGK's own
knowledge base, RAG, memory, tools, and agent capabilities.

## How to use

- Type your question in the chat box below
- KGK AI will respond with streaming text
- Sources (if any) are shown with each response

## Capabilities

- **Conversational AI** with KGK personality
- **Knowledge retrieval** (RAG) from uploaded documents
- **Memory** across conversation turns
- **Tools** (calculator, Python sandbox, document search)

## About

KGK AI is an open-source-friendly platform by **KGK (Siddharitha Technologies
Private Limited)**. It uses open foundation models and does not claim to have
created its foundation model from scratch.
"""


def chat_stream(
    message: str,
    history: list[dict[str, str]],
    conversation_id: str = "default",
) -> Generator[str, None, None]:
    """Stream a chat response for Gradio's ChatInterface.

    Args:
        message: User's message text.
        history: Gradio conversation history (list of message dicts).
        conversation_id: Conversation identifier.

    Yields:
        Accumulated response text (for Gradio streaming display).
    """
    session_mgr = get_session_manager()
    accumulated = ""

    try:
        for chunk in session_mgr.stream_message(
            message=message,
            conversation_id=conversation_id,
        ):
            accumulated += chunk
            yield accumulated
    except Exception as e:
        logger.error(f"UI streaming error: {e}")
        yield accumulated + f"\n\n[Error: {str(e)}]"


def chat_respond(
    message: str,
    history: list[dict[str, str]],
    conversation_id: str = "default",
) -> str:
    """Non-streaming chat response for Gradio.

    Args:
        message: User's message text.
        history: Gradio conversation history.
        conversation_id: Conversation identifier.

    Returns:
        Response text.
    """
    session_mgr = get_session_manager()
    result = session_mgr.send_message(
        message=message,
        conversation_id=conversation_id,
    )
    return result.text


def clear_conversation(conversation_id: str = "default") -> str:
    """Clear a conversation and return confirmation.

    Args:
        conversation_id: Conversation identifier.

    Returns:
        Confirmation message.
    """
    session_mgr = get_session_manager()
    cleared = session_mgr.clear_session(conversation_id)
    if cleared:
        return "Conversation cleared."
    return "No conversation found to clear."


def _gradio_version() -> tuple[int, int]:
    """Get Gradio major.minor version."""
    import gradio as gr
    parts = gr.__version__.split(".")
    return (int(parts[0]), int(parts[1]))


def create_ui():
    """Create and configure the Gradio UI.

    Returns:
        Configured Gradio Blocks application.
    """
    import gradio as gr

    settings = get_settings()
    ver = _gradio_version()

    # Gradio 5.x: theme/css on Blocks(); Gradio 6.x: on launch()
    blocks_kwargs: dict[str, Any] = {"title": settings.ui_title}
    if ver < (6, 0):
        blocks_kwargs["theme"] = gr.themes.Soft()
        blocks_kwargs["css"] = (
            ".kgk-header { text-align: center; margin-bottom: 1rem; }"
            " .kgk-footer { text-align: center; margin-top: 2rem; color: #888; font-size: 0.85rem; }"
        )

    with gr.Blocks(**blocks_kwargs) as demo:
        # Header
        gr.Markdown(
            f"# {settings.ui_title}\n\n**{settings.ui_subtitle}**",
            elem_classes="kgk-header",
        )

        with gr.Tabs():
            # Tab 1: Chat
            with gr.Tab("Chat"):
                conversation_id_state = gr.State("default")

                chatbot = gr.Chatbot(
                    label="KGK AI Chat",
                    height=500,
                    show_label=False,
                    avatar_images=(None, "🤖"),
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Ask KGK AI anything...",
                        lines=2,
                        scale=8,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear Conversation", variant="stop", size="sm")
                    conv_id_input = gr.Textbox(
                        label="Conversation ID",
                        value="default",
                        scale=2,
                        interactive=True,
                    )

                # Streaming chat handler
                def _stream_handler(message: str, history: list, conv_id: str):
                    """Handle streaming chat with Gradio."""
                    if not message.strip():
                        yield history, ""
                        return

                    history = history + [{"role": "user", "content": message}]
                    response = ""
                    for chunk in chat_stream(message, history, conv_id):
                        response = chunk
                        yield history + [{"role": "assistant", "content": response}], ""

                send_btn.click(
                    _stream_handler,
                    inputs=[msg_input, chatbot, conv_id_input],
                    outputs=[chatbot, msg_input],
                )

                msg_input.submit(
                    _stream_handler,
                    inputs=[msg_input, chatbot, conv_id_input],
                    outputs=[chatbot, msg_input],
                )

                def _clear_handler(conv_id: str):
                    """Handle clear conversation."""
                    msg = clear_conversation(conv_id)
                    return [], msg

                clear_btn.click(
                    _clear_handler,
                    inputs=[conv_id_input],
                    outputs=[chatbot, msg_input],
                )

            # Tab 2: Model Info
            with gr.Tab("Model & Settings"):
                info_display = gr.Markdown(_get_model_info_text())
                refresh_btn = gr.Button("Refresh Info", size="sm")
                refresh_btn.click(
                    lambda: _get_model_info_text(),
                    outputs=[info_display],
                )

            # Tab 3: About
            with gr.Tab("About"):
                gr.Markdown(_get_about_text())

        # Footer
        gr.Markdown(
            "KGK AI v0.1.0 | [Siddharitha Technologies](https://kgk.ai) | Built on open-source models",
            elem_classes="kgk-footer",
        )

    return demo


def launch_ui(
    host: Optional[str] = None,
    port: Optional[int] = None,
    share: bool = False,
) -> None:
    """Create and launch the KGK AI Gradio UI.

    Args:
        host: Server host (defaults to settings).
        port: Server port (defaults to settings).
        share: Whether to create a public link.
    """
    setup_logging()
    settings = get_settings()
    host = host or settings.api_host
    port = port or settings.api_port

    logger.info(
        f"Launching KGK AI Gradio UI on {host}:{port}",
        extra={"component": "ui", "host": host, "port": port},
    )

    import gradio as gr
    ver = _gradio_version()

    demo = create_ui()

    launch_kwargs: dict[str, Any] = {
        "server_name": host,
        "server_port": port,
        "share": share,
        "show_error": True,
    }
    # Gradio 6.x: theme/css moved to launch()
    if ver >= (6, 0):
        launch_kwargs["theme"] = gr.themes.Soft()
        launch_kwargs["css"] = (
            ".kgk-header { text-align: center; margin-bottom: 1rem; }"
            " .kgk-footer { text-align: center; margin-top: 2rem; color: #888; font-size: 0.85rem; }"
        )

    demo.launch(**launch_kwargs)
