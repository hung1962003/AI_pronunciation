from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
import webbrowser
import os
import json

import lambdaTTS
import lambdaTTSOpenAI
import lambdaSpeechToScore
import lambdaGetSample

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

rootPath = ''


@app.get(rootPath + '/', response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})


@app.post(rootPath + '/getAudioFromText')
async def getAudioFromText(request: Request):
    body = await request.json()
    event = {'body': json.dumps(body)}
    lambda_response = lambdaTTS.lambda_handler(event, [])
    # Parse body từ JSON string thành object để FastAPI trả về đúng format
    if isinstance(lambda_response, dict) and 'body' in lambda_response:
        try:
            lambda_response['body'] = json.loads(lambda_response['body'])
        except:
            pass
    return lambda_response


@app.post(rootPath + '/getOpenAIAudioFromText')
async def getOpenAIAudioFromText(request: Request):
    body = await request.json()
    event = {'body': json.dumps(body)}
    lambda_response = lambdaTTSOpenAI.lambda_handler(event, [])
    if isinstance(lambda_response, dict) and 'body' in lambda_response:
        try:
            lambda_response['body'] = json.loads(lambda_response['body'])
        except Exception:
            pass
    return lambda_response


@app.post(rootPath + '/getSample')
async def getNext(request: Request):
    body = await request.json()
    event = {'body': json.dumps(body)}
    lambda_response = lambdaGetSample.lambda_handler(event, [])
    # Parse JSON string từ lambda_handler thành dict
    if isinstance(lambda_response, str):
        try:
            return json.loads(lambda_response)
        except:
            return lambda_response
    return lambda_response


@app.post(rootPath + '/GetAccuracyFromRecordedAudio')
async def GetAccuracyFromRecordedAudio(request: Request):
    try:
        body = await request.json()
        event = {'body': json.dumps(body)}
        lambda_correct_output = lambdaSpeechToScore.lambda_handler(event, [])
        
        # Parse JSON string từ lambda_handler thành dict
        if isinstance(lambda_correct_output, str):
            try:
                return json.loads(lambda_correct_output)
            except json.JSONDecodeError:
                # Nếu không parse được, có thể là error response
                return lambda_correct_output
        elif isinstance(lambda_correct_output, dict) and 'body' in lambda_correct_output:
            # Nếu là dict với 'body' (format API Gateway), parse body
            try:
                if isinstance(lambda_correct_output['body'], str):
                    lambda_correct_output['body'] = json.loads(lambda_correct_output['body'])
            except:
                pass
            return lambda_correct_output
        
        return lambda_correct_output
    except Exception as e:
        print('Error in GetAccuracyFromRecordedAudio: ', str(e))
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Credentials': "true",
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': ''
        }


if __name__ == "__main__":
    language = 'en'
    print(f"Current directory: {os.getcwd()}")
    webbrowser.open_new('http://127.0.0.1:8000/')
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
