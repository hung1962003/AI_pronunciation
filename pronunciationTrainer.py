
import torch
import numpy as np
import models as mo
import WordMetrics
import WordMatching as wm

import ModelInterfaces as mi
import AIModels
import RuleBasedModels
from string import punctuation
import time

import function
def getTrainer(language: str):

    asr_model = mo.getASRModel(language,use_whisper=True)
    
    
    if language == 'en':
        phonem_converter = RuleBasedModels.EngPhonemConverter()
    elif language == 'en-gb':
        phonem_converter = RuleBasedModels.EngPhonemConverterUK()
    else:
        raise ValueError('Language not implemented')

    trainer = PronunciationTrainer(
        asr_model, phonem_converter)

    return trainer


class PronunciationTrainer:
    current_transcript: str
    current_ipa: str
    current_recorded_audio: torch.Tensor
    current_recorded_transcript: str
    current_recorded_word_locations: list
    current_recorded_intonations: torch.tensor
    current_words_pronunciation_accuracy = []
    categories_thresholds = np.array([80, 60, 59])
    sampling_rate = 16000

    def __init__(self, asr_model: mi.IASRModel, word_to_ipa_coverter: mi.ITextToPhonemModel) -> None:
        self.asr_model = asr_model
        self.ipa_converter = word_to_ipa_coverter

    def getTranscriptAndWordsLocations(self, audio_length_in_samples: int):

        audio_transcript = self.asr_model.getTranscript()
        word_locations_in_samples = self.asr_model.getWordLocations()

        fade_duration_in_samples = 0.05*self.sampling_rate
        word_locations_in_samples = [(int(np.maximum(0, word['start_ts']-fade_duration_in_samples)), int(np.minimum(
            audio_length_in_samples-1, word['end_ts']+fade_duration_in_samples))) for word in word_locations_in_samples]

        return audio_transcript, word_locations_in_samples

    def getWordsRelativeIntonation(self, Audio: torch.tensor, word_locations: list):
        intonations = torch.zeros((len(word_locations), 1))
        intonation_fade_samples = 0.3*self.sampling_rate
        print(intonations.shape)
        for word in range(len(word_locations)):
            intonation_start = int(np.maximum(
                0, word_locations[word][0]-intonation_fade_samples))
            intonation_end = int(np.minimum(
                Audio.shape[1]-1, word_locations[word][1]+intonation_fade_samples))
            intonations[word] = torch.sqrt(torch.mean(
                Audio[0][intonation_start:intonation_end]**2))

        intonations = intonations/torch.mean(intonations)
        return intonations

    ##################### ASR Functions ###########################

    def processAudioForGivenText(self, recordedAudio:str ,recordedAudio1:torch.Tensor = None, real_text=None, language='en'):

        start = time.time()
        # recording_transcript, recording_ipa, word_locations = self.getAudioTranscript(
        #     recordedAudio)
         
        recording_ipa = function.get_phonemes(recordedAudio)
        print("Debug - recording_ipa:", recording_ipa)
        print("Debug - real_text:", real_text)
        recording_transcript = function.ipa_to_english(recording_ipa, real_text)
        print("Debug - recording_transcript after ipa_to_english:", recording_transcript)
        word_locations= self.getAudioTranscript(recordedAudio1)
        print ("Debug - word_locations:", word_locations)
        print(1)
        print("recording_transcript: " + recording_transcript)
        print("recording_ipa: " + recording_ipa)
        print("word_locations: " + str(word_locations))
        lambda_ipa_converter = {}
        lambda_ipa_converter[language] =RuleBasedModels.get_phonem_converter(language)
        reference_ipa = lambda_ipa_converter[language].convertToPhonem(real_text)
        print("reference_ipa: "+ str(reference_ipa))
        # real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices = self.matchSampleAndRecordedWords(
        #     real_text, recording_transcript)
        # print('Time for matching transcripts: ', str(time.time()-start))
        #mapped_words_indices = self.matchSampleAndRecordedWords(real_text, recording_transcript)
        #print("mapped_words_indices: " + str(mapped_words_indices))
        # Handle case where recording_transcript is empty or very short
        if not recording_transcript or len(recording_transcript.strip()) <= 2:
            print("Warning: Very short or empty recording transcript")
            # Create fallback alignment
            real_words = real_text.split()
            real_and_transcribed_words = [(word, '-') for word in real_words]
        else:        
            real_and_transcribed_words = function.align_real_and_transcribed(real_text, recording_transcript)  
        real_and_transcribed_words_ipa = function.getComparationPhonemes(reference_ipa, recording_ipa)
        print("real_and_transcribed_words: " + str(real_and_transcribed_words))
        print("real_and_transcribed_words_ipa: "+ str(real_and_transcribed_words_ipa))
        # print("mapped_words_indices: " +str(mapped_words_indices))
        # start_time, end_time = self.getWordLocationsFromRecordInSeconds(word_locations, mapped_words_indices) 
        start_time, end_time = 0, 0
        pronunciation_accuracy, current_words_pronunciation_accuracy = self.getPronunciationAccuracy(
            real_and_transcribed_words_ipa)  # _ipa
        print("pronunciation_accuracy: " + str(pronunciation_accuracy))
        print("current_words_pronunciation_accuracy: " + str(current_words_pronunciation_accuracy))
        pronunciation_categories = self.getWordsPronunciationCategory(
            current_words_pronunciation_accuracy)
        print("pronunciation_categories: " + str(pronunciation_categories))
        AIFeedback = function.aiFeedback(real_text, real_and_transcribed_words_ipa)
        #print("AIFeedback: " + str(AIFeedback))
        result = {'recording_transcript': recording_transcript,
                  'real_and_transcribed_words': real_and_transcribed_words,
                  'recording_ipa': recording_ipa, 'start_time': start_time, 'end_time': end_time,
                  'real_and_transcribed_words_ipa': real_and_transcribed_words_ipa, #'pronunciation_accuracy': pronunciation_accuracy,
                  'pronunciation_categories': pronunciation_categories,
                  'real_text': real_text,
                  'AIFeedback': AIFeedback}
        print("pronunciation result: " + str(result))
        return result

    def getAudioTranscript(self, recordedAudio: torch.Tensor = None):
        current_recorded_audio = recordedAudio

        current_recorded_audio = self.preprocessAudio(
            current_recorded_audio)

        self.asr_model.processAudio(current_recorded_audio)

        current_recorded_transcript, current_recorded_word_locations = self.getTranscriptAndWordsLocations(
            current_recorded_audio.shape[1])
        # current_recorded_ipa = self.ipa_converter.convertToPhonem(
        #     current_recorded_transcript)
        return  current_recorded_word_locations
        # return current_recorded_transcript, current_recorded_ipa, current_recorded_word_locations

    def getWordLocationsFromRecordInSeconds(self, word_locations, mapped_words_indices) -> list:
        start_time = []
        end_time = []
        for word_idx in range(len(mapped_words_indices)):
            start_time.append(float(word_locations[mapped_words_indices[word_idx]]
                                    [0])/self.sampling_rate)
            end_time.append(float(word_locations[mapped_words_indices[word_idx]]
                                  [1])/self.sampling_rate)
        return ' '.join([str(time) for time in start_time]), ' '.join([str(time) for time in end_time])

    ##################### END ASR Functions ###########################

    ##################### Evaluation Functions ###########################
    def matchSampleAndRecordedWords(self, real_text, recorded_transcript):
        words_estimated = recorded_transcript.split()

        if real_text is None:
            words_real = self.current_transcript[0].split()
        else:
            print(2)
            words_real = real_text.split()

        mapped_words, mapped_words_indices = wm.get_best_mapped_words(
            words_estimated, words_real)

        # real_and_transcribed_words = []
        # real_and_transcribed_words_ipa = []
        # for word_idx in range(len(words_real)):
        #     if word_idx >= len(mapped_words)-1:
        #         mapped_words.append('-')
        #     real_and_transcribed_words.append(
        #         (words_real[word_idx],    mapped_words[word_idx]))
        #     real_and_transcribed_words_ipa.append((self.ipa_converter.convertToPhonem(words_real[word_idx]),
        #                                            self.ipa_converter.convertToPhonem(mapped_words[word_idx])))
        # return real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices
        return mapped_words_indices
    def getPronunciationAccuracy(self, real_and_transcribed_words_ipa) -> float:
        """
        Tính độ chính xác phát âm dựa trên so sánh phoneme (không phải ký tự).
        Sử dụng compare_ipa_pairs để so sánh chính xác từng phoneme.
        """
        total_correct_phonemes = 0.
        number_of_phonemes = 0.
        current_words_pronunciation_accuracy = []
        
        # Handle case where input might be empty or malformed
        if not real_and_transcribed_words_ipa:
            return 0.0, []
        
        # Sử dụng compare_ipa_pairs để so sánh phoneme chính xác
        comparison_results = function.compare_ipa_pairs(
            real_and_transcribed_words_ipa, 
            strict_syllable_match=False,
            return_as_string=False
        )
            
        for idx, pair in enumerate(real_and_transcribed_words_ipa):
            # Ensure pair is a tuple/list with at least 2 elements
            if not isinstance(pair, (tuple, list)) or len(pair) < 2:
                continue
                
            real_phoneme = str(pair[0]) if pair[0] is not None else ""
            transcribed_phoneme = str(pair[1]) if pair[1] is not None else ""
            
            # Skip if either is empty
            if not real_phoneme or not transcribed_phoneme:
                continue
            
            # Lấy kết quả so sánh phoneme cho từ này
            if idx < len(comparison_results):
                phoneme_comparison = comparison_results[idx]  # Chuỗi '1'/'0' cho mỗi phoneme
            else:
                # Fallback: tách thành phoneme và so sánh
                real_phonemes = function.split_ipa_into_phonemes(real_phoneme)
                transcribed_phonemes = function.split_ipa_into_phonemes(transcribed_phoneme)
                # So sánh đơn giản: đếm số phoneme giống nhau ở cùng vị trí
                min_len = min(len(real_phonemes), len(transcribed_phonemes))
                phoneme_comparison = ''.join(['1' if i < min_len and real_phonemes[i] == transcribed_phonemes[i] else '0' 
                                             for i in range(len(real_phonemes))])
            
            # Đếm số phoneme đúng và tổng số phoneme
            number_of_phonemes_in_word = len(phoneme_comparison)
            number_of_correct_phonemes_in_word = phoneme_comparison.count('1')
            number_of_incorrect_phonemes_in_word = number_of_phonemes_in_word - number_of_correct_phonemes_in_word
            
            # Cập nhật tổng
            total_correct_phonemes += number_of_correct_phonemes_in_word
            number_of_phonemes += number_of_phonemes_in_word

            # Tính độ chính xác cho từ này (không bao giờ âm - đảm bảo >= 0)
            if number_of_phonemes_in_word > 0:
                word_accuracy = max(0.0, float(number_of_correct_phonemes_in_word) / number_of_phonemes_in_word * 100)
                current_words_pronunciation_accuracy.append(word_accuracy)

        # Tính tổng độ chính xác (không bao giờ âm - đảm bảo >= 0)
        if number_of_phonemes == 0:
            return 0.0, current_words_pronunciation_accuracy
            
        percentage_of_correct_pronunciations = max(0.0, (
            total_correct_phonemes / number_of_phonemes * 100))

        return np.round(percentage_of_correct_pronunciations), current_words_pronunciation_accuracy

    def removePunctuation(self, word: str) -> str:
        return ''.join([char for char in word if char not in punctuation])

    def getWordsPronunciationCategory(self, accuracies) -> list:
          # accuracies có thể là:
        # - list[float]
        # - tuple(overall_accuracy, list[float])
        # - scalar float

        if isinstance(accuracies, tuple) and len(accuracies) == 2:
            _, accuracies = accuracies

        if np.isscalar(accuracies):
            accuracies = [float(accuracies)]
        categories = []

        for accuracy in accuracies:
            categories.append(
                self.getPronunciationCategoryFromAccuracy(accuracy))

        return categories

    def getPronunciationCategoryFromAccuracy(self, accuracy) -> int:
         # đảm bảo accuracy là scalar
        if not np.isscalar(accuracy):
            accuracy = float(np.mean(accuracy))
        else:
            accuracy = float(accuracy)
        return np.argmin(abs(self.categories_thresholds-accuracy))

    def preprocessAudio(self, audio: torch.tensor) -> torch.tensor:
        audio = audio-torch.mean(audio)
        audio = audio/torch.max(torch.abs(audio))
        return audio
