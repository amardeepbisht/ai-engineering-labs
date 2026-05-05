from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any


def image_file_to_data_url(file_path: str) -> str:
    """Convert an image file path into a base64 data URL."""
    path = Path(file_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("Please provide a supported image file.")

    # TODO 1: read the image bytes.
    image_bytes = path.read_bytes()

    # TODO 2: base64 encode the bytes and decode to text.
    encoded_text = base64.b64encode(image_bytes).decode("utf-8")

    # TODO 3: return f"data:{mime_type};base64,{encoded_text}".
    return f"data:{mime_type};base64,{encoded_text}"


def get_uploaded_file_path(file_value: Any) -> str | None:
    """Normalize common Gradio file values into a path string."""
    # TODO 4: if file_value is a string, return it.
    if isinstance(file_value, str):
        return file_value

    # TODO 5: if file_value is a dict, return file_value["path"] or file_value["name"].
    if isinstance(file_value, dict):
        return file_value.get("path") or file_value.get("name")

    # TODO 6: otherwise try file_value.path or file_value.name.
    return getattr(file_value, "path", None) or getattr(file_value, "name", None)


def build_user_content(message: dict[str, Any]) -> str | list[dict[str, Any]]:
    """Convert Gradio multimodal input into OpenRouter user content."""
    text = (message.get("text") or "").strip()
    files = message.get("files") or []

    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})

    for file_value in files:
        file_path = get_uploaded_file_path(file_value)
        if file_path:
            # TODO 7: append an image_url content block.
            # Shape:
            # {"type": "image_url", "image_url": {"url": image_file_to_data_url(file_path)}}
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_file_to_data_url(file_path)},
                }
            )

    if not content:
        return "Please send text or upload an image."
    if len(content) == 1 and content[0]["type"] == "text":
        return content[0]["text"]
    if not text:
        content.insert(0, {"type": "text", "text": "Please analyze this image."})
    return content


def text_from_history_content(content: Any) -> str | None:
    """Extract text from common Gradio history content shapes."""
    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_parts = [
            item.get("text", "").strip()
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        text = "\n".join(part for part in text_parts if part)
        return text or None

    return None


def build_history_content(content: Any, include_images: bool = False) -> str | list[dict[str, Any]] | None:
    """Convert Gradio history content back into OpenRouter content."""
    text = text_from_history_content(content)
    if not include_images:
        return text

    content_blocks: list[dict[str, Any]] = []
    if text:
        content_blocks.append({"type": "text", "text": text})

    if isinstance(content, dict) and "files" in content:
        user_content = build_user_content(content)
        if isinstance(user_content, str):
            return user_content
        return user_content

    values = content if isinstance(content, list) else [content]
    for value in values:
        if isinstance(value, dict) and value.get("type") == "image_url":
            content_blocks.append(value)
            continue

        file_path = get_uploaded_file_path(value)
        if file_path:
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_file_to_data_url(file_path)},
                }
            )

    if not content_blocks:
        return None
    if len(content_blocks) == 1 and content_blocks[0]["type"] == "text":
        return content_blocks[0]["text"]
    if content_blocks[0]["type"] != "text":
        content_blocks.insert(0, {"type": "text", "text": "Previously uploaded image."})
    return content_blocks


def build_multimodal_messages(
    history: list[dict[str, Any]],
    current_message: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build messages for a vision-capable OpenRouter chat model."""
    messages: list[dict[str, Any]] = []

    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_content = build_history_content(item[0], include_images=True)
            assistant_content = build_history_content(item[1])
            if user_content:
                messages.append({"role": "user", "content": user_content})
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            continue

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = build_history_content(item.get("content"), include_images=role == "user")
        if role in {"user", "assistant"} and content:
            # TODO 8: append prior text messages.
            messages.append({"role": role, "content": content})

    # TODO 9: append the latest user message using build_user_content(current_message).
    messages.append({"role": "user", "content": build_user_content(current_message)})
    return messages
