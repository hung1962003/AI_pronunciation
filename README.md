# AI Pronunciation Trainer

An AI-powered web app for scoring English pronunciation. The learner reads a reference sentence, the app records it, recognizes the speech, compares IPA phoneme by phoneme, and returns a score, per-letter correctness, and corrective feedback.

![Main screen](images/MainScreen.jpg)

## Features

- Pronunciation scoring at **phoneme** and **letter** level
- Supports **American English** (`en`) and **British English** (`en-gb`)
- Per-word timestamps with good / medium / bad classification
- Detailed AI feedback for mispronounced words (Groq – Llama 3.3 70B)
- Text-to-Speech playback of the reference sentence (gTTS or OpenAI TTS)

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI, Uvicorn, Gunicorn |
| ASR + word timestamps | Whisper (`openai/whisper-base`, Transformers) |
| Audio-to-IPA recognition | Hugging Face Space `lgtitony/doan` (gradio_client) |
| Reference IPA | phonemizer / eSpeak NG, eng_to_ipa |
| Sequence alignment | Needleman–Wunsch / Hirschberg (`sequence_align`), DTW |
| AI feedback | Groq API |
| TTS | gTTS, OpenAI `gpt-4o-mini-tts` |
| Frontend | HTML / CSS / JavaScript (Jinja2 template) |

## Project Structure

```
├── webApp.py                  # FastAPI app and endpoints
├── lambdaSpeechToScore.py     # Audio processing → scoring
├── lambdaGetSample.py         # Returns sample sentence + IPA
├── lambdaTTS.py               # TTS via gTTS
├── lambdaTTSOpenAI.py         # TTS via OpenAI
├── pronunciationTrainer.py    # Core scoring pipeline
├── function.py                # IPA, alignment, phoneme comparison, AI feedback
├── whisper_wrapper.py         # Whisper ASR wrapper
├── models.py / AIModels.py    # ASR / TTS model loading
├── RuleBasedModels.py         # Text → IPA (US / UK)
├── WordMatching.py, WordMetrics.py
├── databases1/                # Sample sentence datasets (en, de)
├── templates/main.html        # UI
├── static/                    # CSS, JS, feedback sounds
└── gunicorn_config.py
```

## Installation

### Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) (available in `PATH`)
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng)

### Setup

```bash
git clone https://github.com/hung1962003/AI_pronunciation.git
cd AI_pronunciation
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
HF_API_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key      # optional, for OpenAI TTS
```

> **Note:** `function.py` hardcodes the eSpeak library path for Windows
> (`C:\Program Files\eSpeak NG\libespeak-ng.dll`). On Linux/macOS, update it, e.g.
> `/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1`.

## Running

Development:

```bash
uvicorn webApp:app --host 0.0.0.0 --port 8000
# or
python webApp.py
```

Production:

```bash
gunicorn webApp:app -c gunicorn_config.py
```

Open: http://127.0.0.1:8000/

See [RUN.md](RUN.md) for details.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/getSample` | Get sample sentence and IPA |
| POST | `/getAudioFromText` | TTS via gTTS |
| POST | `/getOpenAIAudioFromText` | TTS via OpenAI |
| POST | `/GetAccuracyFromRecordedAudio` | Score pronunciation |

Example scoring request:

```json
{
  "title": "This is a sample sentence",
  "base64Audio": "data:audio/ogg;base64,AAA...",
  "language": "en"
}
```

The response includes `pronunciation_accuracy`, `ipa_transcript`, `is_letter_correct_all_words`, `start_time`, `end_time`, `pair_accuracy_category`, `AIFeedback`, and more.

Full documentation: [API_GetAccuracyFromRecordedAudio.md](API_GetAccuracyFromRecordedAudio.md).

## Processing Pipeline

1. Client sends OGG audio (base64) plus the reference sentence
2. FFmpeg converts it to 16 kHz mono WAV
3. The HF Space returns the spoken IPA; Whisper provides the transcript and word timestamps
4. Reference IPA is generated from the sentence and aligned with the recorded IPA
5. Phoneme comparison produces per-word and per-letter scores
6. A Groq LLM generates feedback for mispronounced words

## Author

**Nguyen Quang Hung** – [@hung1962003](https://github.com/hung1962003)
