import ModelInterfaces
import torch
import numpy as np
import eng_to_ipa
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from phonemizer.punctuation import Punctuation


def get_phonem_converter(language: str):
    if language == 'en':
        phonem_converter = EngPhonemConverter()
    elif language == 'en-gb':
        phonem_converter = EngPhonemConverterUK()
    else:
        raise ValueError('Language not implemented')

    return phonem_converter

# class EpitranPhonemConverter(ModelInterfaces.ITextToPhonemModel):
#     word_locations_in_samples = None
#     audio_transcript = None

#     def __init__(self, epitran_model) -> None:
#         super().__init__()
#         self.epitran_model = epitran_model

#     def convertToPhonem(self, sentence: str) -> str:
#         phonem_representation = self.epitran_model.transliterate(sentence)
#         return phonem_representation


class EngPhonemConverter(ModelInterfaces.ITextToPhonemModel):

    def __init__(self,) -> None:
        super().__init__()

    def convertToPhonem(self, sentence: str) -> str:
        print(0)
        phonem_representation = eng_to_ipa.convert(sentence)
        print("phonem_representation  " + phonem_representation)
        phonem_representation = phonem_representation.replace('*','')
        return phonem_representation
class EngPhonemConverterUK(ModelInterfaces.ITextToPhonemModel):
    def __init__(self):
        # Khởi tạo backend tiếng Anh (UK)
        self.backend = EspeakBackend('en-gb')
        self.separator = Separator(phone='', word=' ')
        self.punct = Punctuation(';:,.!"?()')

    def convertToPhonem(self, sentence: str) -> str:
        # Loại bỏ dấu câu
        clean_text = self.punct.remove(sentence)
        # Chuyển sang IPA giọng Anh
        phonemes = self.backend.phonemize(
            [clean_text],
            separator=self.separator,
            strip=True
        )[0]
        return phonemes