from __future__ import annotations

import os
import importlib.util
from pathlib import Path
from typing import Any

import gradio as gr
from openai import OpenAI

_MESSAGES_PATH = Path(__file__).with_name("2_multimodal_messages.py")
_MESSAGES_SPEC = importlib.util.spec_from_file_location("multimodal_messages", _MESSAGES_PATH)
if _MESSAGES_SPEC is None or _MESSAGES_SPEC.loader is None:
    raise ImportError(f"Could not load {_MESSAGES_PATH}")
_MESSAGES_MODULE = importlib.util.module_from_spec(_MESSAGES_SPEC)
_MESSAGES_SPEC.loader.exec_module(_MESSAGES_MODULE)
build_multimodal_messages = _MESSAGES_MODULE.build_multimodal_messages


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
APP_TITLE = "Basic Multimodal Chat"
WEB_SEARCH_TOOL = {
    "type": "openrouter:web_search",
    "parameters": {
        "max_results": 5,
        "max_total_results": 10,
        "search_context_size": "medium",
    },
}
RECENT_IMAGE_FILES: list[Any] = []


def add_recent_image_context(message: dict[str, Any]) -> dict[str, Any]:
    """Reuse the latest uploaded image when the next turn has no file."""
    global RECENT_IMAGE_FILES

    files = message.get("files") or []
    if files:
        RECENT_IMAGE_FILES = files
        return message

    if RECENT_IMAGE_FILES:
        return {**message, "files": RECENT_IMAGE_FILES}

    return message


def stream_basic_chat(
    message: dict[str, Any],
    history: list[dict[str, Any]],
    api_key: str,
):
    """Stream a multimodal response into Gradio."""
    api_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        yield "Add your OpenRouter API key first."
        return

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    message = add_recent_image_context(message)

    # TODO 1: call client.chat.completions.create with:
    # model=DEFAULT_MODEL
    # messages=build_multimodal_messages(history, message)
    # stream=True
    # extra_body={"provider": {"data_collection": "deny"}}
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=build_multimodal_messages(history, message),
        tools=[WEB_SEARCH_TOOL],
        stream=True,
        extra_body={"provider": {"data_collection": "deny"}},
    )

    answer = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            # TODO 2: add delta to answer and yield the growing answer.
            answer += delta
            yield answer


def build_demo() -> gr.ChatInterface:
    """Create the basic Gradio app."""
    # TODO 3: return gr.ChatInterface with:
    # fn=stream_basic_chat
    # type="messages"
    # multimodal=True
    # title=APP_TITLE
    # textbox=gr.MultimodalTextbox(file_types=["image"])
    # additional_inputs=[gr.Textbox(label="OpenRouter API Key", type="password")]
    return gr.ChatInterface(
        fn=stream_basic_chat,
        multimodal=True,
        title=APP_TITLE,
        textbox=gr.MultimodalTextbox(file_types=["image"]),
        additional_inputs=[gr.Textbox(label="OpenRouter API Key", type="password")],
    )


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
