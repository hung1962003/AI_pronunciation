import base64
import json
import os
from typing import Optional, Tuple

from openai import OpenAI


DEFAULT_OPENAI_TTS_MODEL = os.getenv('OPENAI_TTS_MODEL', 'gpt-4o-mini-tts')
DEFAULT_VOICE = os.getenv('OPENAI_TTS_VOICE', 'verse')
DEFAULT_AUDIO_FORMAT = os.getenv('OPENAI_TTS_FORMAT', 'mp3')
SUPPORTED_FORMATS = {
    'mp3': 'audio/mpeg',
    'wav': 'audio/wav',
    'ogg': 'audio/ogg',
}

_openai_client: Optional[OpenAI] = None


def lambda_handler(event, context):

    try:
        body = json.loads(event.get('body', '{}'))
    except Exception:
        body = {}

    text_string = body.get('value') or body.get('title')
    voice = body.get('voice') or DEFAULT_VOICE
    requested_format = (body.get('audioFormat') or DEFAULT_AUDIO_FORMAT).lower()

    if text_string is None:
        return _build_error_response(
            400,
            "Missing 'value' field in request body"
        )

    if not os.getenv('OPENAI_API_KEY'):
        return _build_error_response(
            500,
            "OPENAI_API_KEY is not configured on the server"
        )

    try:
        audio_bytes, mime_type = _synthesize_with_openai(
            text=text_string,
            voice=voice,
            audio_format=requested_format
        )
    except Exception as exc:
        return _build_error_response(500, str(exc))

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps(
            {
                "wavBase64": base64.b64encode(audio_bytes).decode('utf-8'),
                "mimeType": mime_type,
                "provider": "openai",
                "voice": voice,
            },
        )
    }


def _synthesize_with_openai(text: str, voice: str, audio_format: str) -> Tuple[bytes, str]:
    client = _get_openai_client()

    format_safe = audio_format if audio_format in SUPPORTED_FORMATS else 'mp3'
    mime_type = SUPPORTED_FORMATS[format_safe]

    response = client.audio.speech.create(
        model=DEFAULT_OPENAI_TTS_MODEL,
        voice=voice,
        input=text,
        response_format=format_safe,
    )

    audio_bytes = response.read()
    return audio_bytes, mime_type


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _build_error_response(status_code: int, message: str) -> dict:
    return {
        'statusCode': status_code,
        'headers': {
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps({"error": message})
    }

