
import json
import os
import base64

from gtts import gTTS

import utilsFileIO


def lambda_handler(event, context):

    # Parse body from API Gateway / Flask-style event
    try:
        body = json.loads(event.get('body', '{}'))
    except Exception:
        body = {}

    # Hỗ trợ cả key 'value' (TTS) và 'title' (cho thống nhất với các API khác)
    text_string = body.get('value') or body.get('title')

    if text_string is None:
        # Không có text để đọc, trả về lỗi rõ ràng thay vì raise exception
        return {
            'statusCode': 400,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps(
                {
                    "error": "Missing 'value' field in request body"
                },
            )
        }
    # Generate TTS audio using gTTS (Google Text-to-Speech)
    # Lưu ý: cần kết nối Internet để gTTS hoạt động
    random_file_name = utilsFileIO.generateRandomString(20) + ".mp3"

    tts = gTTS(text=text_string, lang="en")
    tts.save(random_file_name)

    with open(random_file_name, "rb") as f:
        audio_byte_array = f.read()

    os.remove(random_file_name)


    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps(
            {
                "wavBase64": str(base64.b64encode(audio_byte_array))[2:-1],
                "mimeType": "audio/mpeg",
            },
        )
    }
