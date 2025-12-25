import torch
import torch.nn as nn
import pickle
from ModelInterfaces import IASRModel
from AIModels import NeuralASR 

def getASRModel(language: str,use_whisper:bool=True) -> IASRModel:

    if use_whisper:
        from whisper_wrapper import WhisperASRModel
        return WhisperASRModel()
    
    
    if language == 'en':
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language='en',
                                               device=torch.device('cpu'))
        model.eval()
        return NeuralASR(model, decoder)
    elif language == 'en-gb':
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language='en',
                                               device=torch.device('cpu'))
        model.eval()
        return NeuralASR(model, decoder)
    
    else:
        raise ValueError('Language not implemented')


def getTTSModel(language: str) -> nn.Module:

    
    if language == 'en':
        speaker = 'lj_16khz'  # 16 kHz
        model_obj = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                   model='silero_tts',
                                   language=language,
                                   speaker=speaker)
    elif language == 'en-gb':
        speaker = 'thorsten_v2'  # 16 kHz
        model_obj = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                   model='silero_tts',
                                   language=language,
                                   speaker=speaker)        
    else:
        raise ValueError('Language not implemented')

    # torch.hub.load for silero_tts thường trả về tuple (model, example_text, sample_rate, speakers, ...)
    # Đảm bảo chỉ trả lại chính object model có method `apply_tts`
    if isinstance(model_obj, tuple):
        model = model_obj[0]
    else:
        model = model_obj

    return model

