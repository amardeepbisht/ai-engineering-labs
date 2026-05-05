from __future__ import annotations

import os
import importlib.util
import subprocess
import tempfile
import uuid
from datetime import date
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
get_uploaded_file_path = _MESSAGES_MODULE.get_uploaded_file_path


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
MODEL_CHOICES = [
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3.5-haiku",
]
TTS_ENGINE_CHOICES = [
    "macOS say",
    "Edge TTS",
    "Piper TTS",
]
CONVERSION_LANGUAGE_CHOICES = ["English", "Hindi"]
TTS_VOICES_BY_ENGINE_LANGUAGE = {
    "macOS say": {
        "English": ["Default", "Samantha", "Alex", "Victoria", "Daniel", "Karen", "Rishi"],
        "Hindi": ["Lekha", "Rishi"],
    },
    "Edge TTS": {
        "English": ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural", "en-US-JennyNeural", "en-US-GuyNeural"],
        "Hindi": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
    },
    "Piper TTS": {
        "English": ["en_US-lessac-medium", "en_US-amy-medium", "en_GB-alba-medium"],
        "Hindi": ["No Hindi Piper voice configured"],
    },
}
APP_TITLE = "OmniAI Studio"
WEB_SEARCH_TOOL = {
    "type": "openrouter:web_search",
    "parameters": {
        "engine": "exa",
        "max_results": 5,
        "max_total_results": 10,
        "search_context_size": "medium",
    },
}
DATETIME_TOOL = {
    "type": "openrouter:datetime",
    "parameters": {"timezone": "Asia/Kolkata"},
}
RECENCY_KEYWORDS = {
    "today",
    "latest",
    "current",
    "recent",
    "new",
    "news",
    "this week",
    "last week",
    "last one week",
    "past week",
    "last 7 days",
    "2026",
}
RECENT_IMAGE_FILES: list[Any] = []
LAST_ASSISTANT_TEXT = ""


def message_text(message: dict[str, Any] | None) -> str:
    """Return the text part of a Gradio multimodal message."""
    return ((message or {}).get("text") or "").strip()


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


def needs_fresh_web_search(message: dict[str, Any]) -> bool:
    """Use web search fallback for clearly time-sensitive questions."""
    text = message_text(message).lower()
    return any(keyword in text for keyword in RECENCY_KEYWORDS)


def build_grounded_messages(
    history: list[dict[str, Any]],
    message: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add temporal guidance before the OpenRouter chat messages."""
    system_message = {
        "role": "system",
        "content": (
            f"Today's date is {date.today().isoformat()}. "
            "For questions about latest, recent, current, today, this week, last week, "
            "or dated facts, use web search before answering. Include concrete dates "
            "from the sources and do not answer from memory for time-sensitive questions."
        ),
    }
    return [system_message, *build_multimodal_messages(history, message)]


def add_user_message(
    history: list[dict[str, Any]] | None,
    message: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], gr.MultimodalTextbox, dict[str, Any], int]:
    """Show the user's text/files immediately in the chat UI."""
    history = history or []
    message = message or {"text": "", "files": []}
    text = message_text(message)
    files = message.get("files") or []
    added_count = 0

    for file_value in files:
        file_path = get_uploaded_file_path(file_value)
        if file_path:
            history.append({"role": "user", "content": {"path": file_path}})
            added_count += 1

    if text:
        history.append({"role": "user", "content": text})
        added_count += 1

    if not added_count:
        history.append({"role": "user", "content": "Please send text or upload an image."})
        added_count = 1

    return history, gr.MultimodalTextbox(value=None, interactive=False), message, added_count


def stream_chatbot_response(
    history: list[dict[str, Any]],
    message: dict[str, Any],
    added_count: int,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
):
    """Stream the assistant response into the custom chatbot."""
    global LAST_ASSISTANT_TEXT

    history = history or []
    prior_history = history[:-added_count] if added_count else history
    api_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        history.append({"role": "assistant", "content": "Add your OpenRouter API key first."})
        yield history
        return

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    message = add_recent_image_context(message)
    extra_body: dict[str, Any] = {"provider": {"data_collection": "deny"}}
    if needs_fresh_web_search(message):
        extra_body["plugins"] = [{"id": "web", "max_results": 5}]

    response = client.chat.completions.create(
        model=(model or DEFAULT_MODEL).strip(),
        messages=build_grounded_messages(prior_history, message),
        temperature=temperature,
        max_tokens=max_tokens,
        tools=[WEB_SEARCH_TOOL, DATETIME_TOOL],
        stream=True,
        extra_body=extra_body,
    )

    answer = ""
    history.append({"role": "assistant", "content": ""})
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            answer += delta
            history[-1]["content"] = answer
            yield history

    if answer.strip():
        LAST_ASSISTANT_TEXT = answer.strip()


def macos_say_audio(text: str, voice: str, speed: int) -> tuple[str, str]:
    """Create an audio file using the macOS say command."""
    audio_path = Path(tempfile.gettempdir()) / f"assistant_audio_{uuid.uuid4().hex}.aiff"
    command = ["say", "-r", str(speed), "-o", str(audio_path)]
    if voice and voice != "Default":
        command.extend(["-v", voice])
    command.append(text)

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise gr.Error("The built-in macOS 'say' command was not found.") from exc
    except subprocess.CalledProcessError as exc:
        raise gr.Error(f"Audio generation failed: {exc.stderr or exc}") from exc

    return str(audio_path), str(audio_path)


def edge_tts_audio(text: str, voice: str, speed: int) -> tuple[str, str]:
    """Create an MP3 file using Edge TTS if installed."""
    audio_path = Path(tempfile.gettempdir()) / f"assistant_audio_{uuid.uuid4().hex}.mp3"
    rate = speed - 100
    rate_text = f"+{rate}%" if rate >= 0 else f"{rate}%"

    try:
        import edge_tts
    except ModuleNotFoundError as exc:
        raise gr.Error("Install Edge TTS with: pip install edge-tts") from exc

    async def save_audio() -> None:
        communicate = edge_tts.Communicate(text, voice, rate=rate_text)
        await communicate.save(str(audio_path))

    import asyncio

    asyncio.run(save_audio())
    return str(audio_path), str(audio_path)


def piper_tts_audio(text: str, voice: str, speed: int) -> tuple[str, str]:
    """Create a WAV file using Piper TTS if installed."""
    if voice.startswith("No Hindi"):
        raise gr.Error("Piper Hindi is not configured. Use Edge TTS for Hindi.")

    audio_path = Path(tempfile.gettempdir()) / f"assistant_audio_{uuid.uuid4().hex}.wav"
    length_scale = max(0.5, min(2.0, 100 / max(speed, 1)))

    try:
        subprocess.run(
            [
                "piper",
                "--model",
                voice,
                "--output_file",
                str(audio_path),
                "--length_scale",
                str(length_scale),
            ],
            input=text,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise gr.Error("Install Piper with: pip install piper-tts") from exc
    except subprocess.CalledProcessError as exc:
        raise gr.Error(f"Piper audio generation failed: {exc.stderr or exc}") from exc

    return str(audio_path), str(audio_path)


def tts_voice_choices(language: str, engine: str) -> list[str]:
    """Return voice choices for the selected language and engine."""
    return TTS_VOICES_BY_ENGINE_LANGUAGE.get(engine, {}).get(language, ["Default"])


def update_tts_voice_choices(language: str, engine: str):
    """Refresh the voice dropdown when language or engine changes."""
    choices = tts_voice_choices(language, engine)
    return gr.Dropdown(choices=choices, value=choices[0])


def synthesize_audio(
    text: str,
    language: str,
    engine: str,
    voice: str,
    speed: int,
) -> tuple[str, str]:
    """Create an audio file with the selected free TTS engine."""
    if engine == "Edge TTS":
        return edge_tts_audio(text, voice, speed)
    if engine == "Piper TTS":
        return piper_tts_audio(text, voice, speed)
    return macos_say_audio(text, voice, speed)


def last_answer_to_audio(
    language: str,
    engine: str,
    voice: str,
    speed: int,
) -> tuple[str | None, str | None]:
    """Create an audio file from the latest assistant response."""
    if not LAST_ASSISTANT_TEXT:
        raise gr.Error("Ask the assistant something first, then generate audio.")

    return synthesize_audio(LAST_ASSISTANT_TEXT, language, engine, voice, speed)


def custom_text_to_audio(
    text: str,
    language: str,
    engine: str,
    voice: str,
    speed: int,
) -> tuple[str | None, str | None]:
    """Create an audio file from user-provided text."""
    clean_text = (text or "").strip()
    if not clean_text:
        raise gr.Error("Type or paste some text first.")

    return synthesize_audio(clean_text, language, engine, voice, speed)


def build_demo() -> gr.Blocks:
    """Create the Gradio app with chat, audio tools, and bottom advanced options."""
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}")

        chatbot = gr.Chatbot(label=APP_TITLE)
        chat_input = gr.MultimodalTextbox(
            file_types=["image"],
            placeholder="Ask anything, or upload an image...",
            show_label=False,
        )
        pending_message = gr.State({})
        added_message_count = gr.State(0)

        with gr.Row():
            conversion_language_input = gr.Dropdown(
                choices=CONVERSION_LANGUAGE_CHOICES,
                value="English",
                label="Conversion Language",
            )
            tts_engine_input = gr.Dropdown(
                choices=TTS_ENGINE_CHOICES,
                value="macOS say",
                label="Audio Engine",
            )
            audio_speed_input = gr.Slider(
                minimum=60,
                maximum=240,
                value=175,
                step=5,
                label="Voice Speed",
            )
        tts_voice_input = gr.Dropdown(
            choices=tts_voice_choices("English", "macOS say"),
            value=tts_voice_choices("English", "macOS say")[0],
            label="Voice",
        )

        audio_button = gr.Button("Generate Audio from Last Answer")
        audio_text = gr.Textbox(
            label="Text to Convert",
            lines=4,
            placeholder="Write or paste English or Hindi text here.",
        )
        custom_audio_button = gr.Button("Generate Audio")
        audio_output = gr.Audio(label="Generated Audio", type="filepath")
        audio_download = gr.File(label="Download Audio")

        with gr.Accordion("Advanced Options", open=False):
            api_key_input = gr.Textbox(label="OpenRouter API Key", type="password")
            model_input = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=DEFAULT_MODEL,
                label="Model",
            )
            temperature_input = gr.Slider(
                minimum=0,
                maximum=1.5,
                value=0.7,
                step=0.1,
                label="Temperature",
            )
            max_tokens_input = gr.Slider(
                minimum=64,
                maximum=2048,
                value=512,
                step=64,
                label="Max Tokens",
            )

        chat_msg = chat_input.submit(
            fn=add_user_message,
            inputs=[chatbot, chat_input],
            outputs=[chatbot, chat_input, pending_message, added_message_count],
            queue=False,
        )
        bot_msg = chat_msg.then(
            fn=stream_chatbot_response,
            inputs=[
                chatbot,
                pending_message,
                added_message_count,
                api_key_input,
                model_input,
                temperature_input,
                max_tokens_input,
            ],
            outputs=chatbot,
        )
        bot_msg.then(
            fn=lambda: gr.MultimodalTextbox(interactive=True),
            inputs=None,
            outputs=chat_input,
        )

        audio_button.click(
            fn=last_answer_to_audio,
            inputs=[
                conversion_language_input,
                tts_engine_input,
                tts_voice_input,
                audio_speed_input,
            ],
            outputs=[audio_output, audio_download],
        )
        custom_audio_button.click(
            fn=custom_text_to_audio,
            inputs=[
                audio_text,
                conversion_language_input,
                tts_engine_input,
                tts_voice_input,
                audio_speed_input,
            ],
            outputs=[audio_output, audio_download],
        )
        conversion_language_input.change(
            fn=update_tts_voice_choices,
            inputs=[conversion_language_input, tts_engine_input],
            outputs=tts_voice_input,
        )
        tts_engine_input.change(
            fn=update_tts_voice_choices,
            inputs=[conversion_language_input, tts_engine_input],
            outputs=tts_voice_input,
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
