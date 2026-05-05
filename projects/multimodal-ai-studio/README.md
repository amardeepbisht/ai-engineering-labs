# OmniAI Studio

OmniAI Studio is a Gradio-based multimodal AI assistant built with OpenRouter. It supports text chat, image understanding, chat memory, web-search-assisted answers for recent topics, and text-to-audio generation.

## Features

- Text chat through OpenRouter-compatible models
- Image upload and multimodal question answering
- Follow-up questions with chat history and latest-image context
- Web search support for current or recent questions
- Model, temperature, and max-token controls
- Text-to-audio from the last assistant answer
- Text-to-audio from custom English or Hindi text
- Audio engines: macOS `say`, Edge TTS, and Piper TTS

## Files

```text
1_openrouter.py              Basic OpenRouter text request helpers
2_multimodal_messages.py     Text/image message construction helpers
3a_basic_gradio_chat.py      Basic multimodal Gradio chat app
3b_advanced_gradio_chat.py   OmniAI Studio app with web search and audio tools
requirements.txt             Python dependencies
```

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your_openrouter_key_here"
```

You can also paste the API key into the app's `Advanced Options` section.

## Run

Basic app:

```bash
python 3a_basic_gradio_chat.py
```

Advanced app:

```bash
python 3b_advanced_gradio_chat.py
```

Open the local Gradio URL shown in the terminal.

## Audio Notes

`macOS say` works on macOS without extra setup.

For Edge TTS:

```bash
pip install edge-tts
```

Recommended Hindi setup:

```text
Conversion Language: Hindi
Audio Engine: Edge TTS
Voice: hi-IN-SwaraNeural
```

For Piper TTS:

```bash
pip install piper-tts
```

Piper currently has English voices configured in this app. For Hindi audio, use Edge TTS.

## Notes

The files are named with numeric prefixes to match the assignment sequence. Because Python modules cannot be imported normally when they start with a number, the Gradio apps load `2_multimodal_messages.py` from the same folder using `importlib`.

Do not commit local virtual environments, cache folders, or `.env` files.
