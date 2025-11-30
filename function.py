
import dotenv

import json
import os

import wave
import base64
import requests
import io
import re

import phonemizer
from phonemizer.punctuation import Punctuation
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from phonemizer.backend.espeak.wrapper import EspeakWrapper




import soundfile as sf
import ast
import random
import difflib
from gradio_client import Client, handle_file

import numpy as np
import torch
from typing import Union

from sequence_align.pairwise import hirschberg, needleman_wunsch
from phonemizer import phonemize
from groq import Groq
dotenv.load_dotenv()

# HF INFERENCE API
API_TOKEN = os.environ.get("HF_API_TOKEN") #https://huggingface.co/settings/profile
headers = {"Authorization": f"Bearer {API_TOKEN}"}
client1 = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
PHONEME_API_URL = "https://api-inference.huggingface.co/models/mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme" # "https://api-inference.huggingface.co/facebook/wav2vec2-xlsr-53-phon-cv-ft"
STT_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"


client = Client("lgtitony/doan")  # Nếu Space private: thêm hf_token="hf_xxx"
EspeakWrapper.set_library('C:\Program Files\eSpeak NG\libespeak-ng.dll')
# def generate_reference_phoneme(reference_text, language='en'):
#     text = Punctuation(';:,.!"?()').remove(reference_text)
#     ref_words = [w.lower() for w in text.strip().split(' ') if w]

#     if language == 'en':
#         backend = EspeakBackend('en-us')
#     else:
#         backend = EspeakBackend('en-gb')

#     separator = Separator(phone='', word=None)
#     lexicon = []
#     for word in ref_words:
#         phoneme = backend.phonemize([word], separator=separator, strip=True)[0]
#         lexicon.append((word, phoneme))

#     reference_phoneme = ' '.join([phon for _, phon in lexicon])

#     return reference_phoneme
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    else:
        return obj
def getComparationPhonemes(reference_phoneme, recorded_phoneme):
    print("Reference Phoneme: ", reference_phoneme)
    print("Recorded Phoneme: ", recorded_phoneme)
    reference_phoneme = reference_phoneme.replace("ˈ", "").replace("ˌ", "")
    seq_a = reference_phoneme
    seq_b = list(recorded_phoneme.replace(' ',''))

    # recorded_phoneme['text']
    aligned_seq_a, aligned_seq_b = needleman_wunsch( # smith_waterman(
        seq_a,
        seq_b,
        match_score=1.0,
        mismatch_score=-1.0,
        indel_score=-1.0,
        gap="_",
    )

    aligned_reference_seq = ''.join(seq_a)
    aligned_recorded_seq = ''.join(aligned_seq_b)
    # recorded_sequence = "aɪ_hoːp_ðeɪ_hɛv_maɪ_fiːv__rədbrænd_aɪl_biː_bæk_su_n__tʊ_pliːz_w_iːdfoː__miː_"
    ref_start_positions = find_word_start_positions(''.join(aligned_reference_seq))

    # split recorded based on the reference start positions
    rec_split_words = split_recorded_sequence(''.join(aligned_recorded_seq), ref_start_positions)
    rec_split_words = [re.sub('( |\\_)$','',w) for w in rec_split_words]

    # split ref based on the reference start positions
    ref_split_words = split_recorded_sequence(''.join(aligned_reference_seq), ref_start_positions)
    ref_split_words = [re.sub('(\\_| )$','',w) for w in ref_split_words]

    # print('Reference Text: ',reference_text)
    # print('(word, reference_phoneme, recorded_phoneme)',list(zip(ref_words, ref_split_words, rec_split_words)))
    #word_comparision_list = list(zip(ref_words, ref_split_words, rec_split_words))
    word_comparision_list = list(zip(ref_split_words, rec_split_words))
    word_comparision_list
    return word_comparision_list
def ipa_to_english(ipa_text: str, english_words_list=None):
    # Handle case where english_words_list is a string
    if isinstance(english_words_list, str):
        english_words_list = english_words_list.split()
    
    # Handle case where english_words_list is None or empty
    if not english_words_list:
        return ""
    
    # Handle case where ipa_text is very short (single phoneme)
    if len(ipa_text.strip()) <= 2:
        print(f"Warning: Very short IPA text received: '{ipa_text}'")
        # Return the first word from english_words_list as fallback
        return english_words_list[0] if english_words_list else ""
    
    # Sử dụng separator hợp lệ
    separator = Separator(phone='|', word=' ')  # khác nhau giữa âm vị và từ
    phonemizer = EspeakBackend(language='en-us')

    ipa_map = {}
    for word in english_words_list:
        try:
            ipa = phonemizer.phonemize([word], separator=separator, strip=True)[0]
            ipa = ipa.replace('|', ' ')  # chuyển về khoảng trắng giữa âm vị để dễ so sánh
            ipa_map[word] = ipa
        except Exception as e:
            print(f"Error phonemizing word '{word}': {e}")
            continue

    ipa_tokens = ipa_text.strip().split()
    result_words = []

    for token in ipa_tokens:
        ipa_values = list(ipa_map.values())
        closest = difflib.get_close_matches(token, ipa_values, n=1, cutoff=0.75)
        if closest:
            for k, v in ipa_map.items():
                if v == closest[0]:
                    result_words.append(k)
                    break
        else:
            # If no close match found, try with lower cutoff
            closest = difflib.get_close_matches(token, ipa_values, n=1, cutoff=0.5)
            if closest:
                for k, v in ipa_map.items():
                    if v == closest[0]:
                        result_words.append(k)
                        break

    return " ".join(result_words)
def find_word_start_positions(reference_sequence):
    # Split the sequence into words based on spaces
    words = reference_sequence.split()
    # Initialize a list to store the start positions
    start_positions = []
    # Initialize the current position
    current_position = 0
    # Iterate over the words
    for word in words:
        # Add the current position to the start positions list
        start_positions.append(current_position)
        # Increment the current position by the length of the word plus 1 (for the space)
        current_position += len(word) + 1
    return start_positions
def split_recorded_sequence(recorded_sequence, start_positions):
    # Initialize a list to store the split words
    split_words = []
    # Iterate over the start positions
    for i in range(len(start_positions)):
        # Get the start position
        start = start_positions[i]
        # If it's the last word, get the end position as the length of the sequence
        if i == len(start_positions) - 1:
            end = len(recorded_sequence)
        # Otherwise, get the end position as the start position of the next word
        else:
            end = start_positions[i + 1]
        # Extract the word from the recorded sequence
        word = recorded_sequence[start:end]
        # Add the word to the list
        split_words.append(word)
    return split_words
def get_phonemes(filepath):
    print("⏳ Đang gửi file âm thanh đến Hugging Face Space...")

    try:
        # Gọi hàm predict (Gradio sẽ tự upload file qua handle_file)
        result = client.predict(
            audio_file=handle_file(filepath),  # đúng tên input của Space (audio_file)
            api_name="/predict"                # đúng endpoint /predict theo View API
        )
        
        # Handle different result formats
        if isinstance(result, str):
            try:
                data = ast.literal_eval(result)
                text = data.get("text", "")
            except:
                text = result
        elif isinstance(result, dict):
            text = result.get("text", "")
        elif isinstance(result, list):
            text = result[0] if result else ""
        else:
            text = str(result)
        
        print("✅ Nhận kết quả thành công!")
        print("📦 Raw result:", text)
        
        return text
        
    except Exception as e:
        print(f"Error getting phonemes: {e}")
        return ""
def align_real_and_transcribed(reference_sequence: str, recorded_sequence: str):
    ref_words = reference_sequence.split()
    rec_words = recorded_sequence.split()

    # Tạo đối tượng so khớp
    matcher = difflib.SequenceMatcher(None, ref_words, rec_words)
    result = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i in range(i2 - i1):
                result.append((ref_words[i1 + i], rec_words[j1 + i]))
        elif tag == 'replace':
            # thay thế => hai bên khác nhau
            for i in range(max(i2 - i1, j2 - j1)):
                ref_word = ref_words[i1 + i] if i1 + i < len(ref_words) else '-'
                rec_word = rec_words[j1 + i] if j1 + i < len(rec_words) else '-'
                result.append((ref_word, rec_word))
        elif tag == 'delete':
            # bị thiếu bên record
            for i in range(i2 - i1):
                result.append((ref_words[i1 + i], '-'))
        elif tag == 'insert':
            # thêm dư bên record
            for i in range(j2 - j1):
                result.append(('-', rec_words[j1 + i]))

    return result    
def aiFeedback(reference_text, word_comparision_list):
    system_message = """You are an expert dialect/accent coach for american spoken english. you will provide valuable feedback to improve my american accent. For ease of understanding, I would prefer you give suggestions for mipronunciation using google pronunciation respelling.
    provide following Overall Impression, Specific Feedback, Google Pronunciation Respelling Suggestions, additional tips"""
    chat_completion = client1.chat.completions.create(
        messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": f"""Reference Text:  {reference_text}
        ( reference_phoneme, recorded_phoneme) 
        {word_comparision_list}""",
                }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    feedback = chat_completion.choices[0].message.content
    return feedback




def split_ipa_into_phonemes(ipa_string):
    """
    Tách IPA thành các phoneme hoàn chỉnh, gồm diphthong, phụ âm ghép, và nguyên âm dài.
    """
    diphthongs = ['aɪ', 'aʊ', 'eɪ', 'oɪ', 'oʊ', 'ɔɪ', 'ɪə', 'ʊə', 'eə']
    consonant_clusters = ['tʃ', 'dʒ']
    special_consonants = ['ʃ', 'ʒ', 'ŋ', 'θ', 'ð']
    
    ipa_clean = ipa_string.replace('_', '').replace(' ', '')
    if not ipa_clean:
        return []
    
    phonemes = []
    i = 0
    
    while i < len(ipa_clean):
        # 2 ký tự ghép (phụ âm hoặc nguyên âm đôi)
        if i + 1 < len(ipa_clean):
            two = ipa_clean[i:i+2]
            if two in diphthongs or two in consonant_clusters:
                phonemes.append(two)
                i += 2
                continue
        
        # Nguyên âm dài (vd: ɜː, iː, uː, ɔː)
        if i + 1 < len(ipa_clean) and ipa_clean[i+1] == 'ː':
            phonemes.append(ipa_clean[i:i+2])
            i += 2
            continue
        
        # Phụ âm đặc biệt
        if ipa_clean[i] in special_consonants:
            phonemes.append(ipa_clean[i])
            i += 1
            continue
        
        # Còn lại
        phonemes.append(ipa_clean[i])
        i += 1
    
    return phonemes

def split_ipa_into_syllables(ipa_string):
    """
    Tách IPA thành các âm tiết (syllables).
    Một âm tiết thường có: phụ âm đầu (optional) + nguyên âm (required) + phụ âm cuối (optional)
    """
    # Định nghĩa các nguyên âm IPA (vowels)
    vowels = set('aæɑɒeəɛiɪoɔuʊʌɚː')
    
    # Loại bỏ underscore và spaces để tách
    ipa_clean = ipa_string.replace('_', '').replace(' ', '')
    if not ipa_clean:
        return []
    
    syllables = []
    current_syllable = []
    found_vowel = False
    
    i = 0
    while i < len(ipa_clean):
        char = ipa_clean[i]
        current_syllable.append(char)
        
        # Kiểm tra xem ký tự hiện tại có phải là nguyên âm không
        if char in vowels:
            found_vowel = True
        
        # Nếu đã tìm thấy nguyên âm và gặp phụ âm mới (không phải nguyên âm)
        # thì có thể là bắt đầu âm tiết mới
        if found_vowel and char not in vowels:
            # Kiểm tra xem có nguyên âm phía sau không
            # Nếu có, thì ký tự hiện tại thuộc âm tiết tiếp theo
            has_vowel_ahead = False
            for j in range(i + 1, len(ipa_clean)):
                if ipa_clean[j] in vowels:
                    has_vowel_ahead = True
                    break
                elif ipa_clean[j] not in vowels:
                    # Nếu gặp nhiều phụ âm liên tiếp, dừng tìm
                    break
            
            if has_vowel_ahead:
                # Ký tự hiện tại thuộc âm tiết tiếp theo
                current_syllable.pop()  # Bỏ ký tự hiện tại
                syllables.append(''.join(current_syllable))
                current_syllable = [char]
                found_vowel = False
        
        i += 1
    
    # Thêm âm tiết cuối cùng
    if current_syllable:
        syllables.append(''.join(current_syllable))
    print("syllables: ", syllables)
    return syllables if syllables else [ipa_clean]

def segment_word_into_graphemes(word: str):
	"""
	Chia từ tiếng Anh thành các 'grapheme' (cụm chữ) theo bộ quy tắc khóa cứng.
	Ưu tiên khớp dài nhất (longest-match) để gom các cụm như: tion, sion, io, ch, sh, th, ph, qu, ng, ee, oo, ea, ai, oi, oy, ay, au, aw, oa, ow, ew, ue, igh, tch, dge, ear, eer, ure...
	"""
	if not word:
		return []
	lowered = word.lower()
	# Bộ grapheme nhiều chữ (khóa cứng 1 IPA hoặc vai trò đơn vị)
	locked_multi = [
		"tion","sion","tian","cian","ture","sure",
		"tch","dge","igh",
		"ch","sh","th","ph","ng","qu",
		"ee","oo","ea","ai","oi","oy","ay","au","aw","oa","ow","ew","ue","ie","ei",
		"io",  # ví dụ trong pronunciation → /ə/
		"ear","eer","ure"
	]
	# Sắp xếp giảm dần độ dài để ưu tiên khớp dài nhất
	locked_multi = sorted(locked_multi, key=len, reverse=True)
	graphemes = []
	i = 0
	while i < len(lowered):
		matched = None
		for pat in locked_multi:
			if lowered.startswith(pat, i):
				matched = word[i:i+len(pat)]
				break
		if matched:
			graphemes.append(matched)
			i += len(matched)
		else:
			graphemes.append(word[i])
			i += 1
	print("graphemes: ", graphemes)
	return graphemes

GRAPHEME_EXPECTED_PHONEMES = {
	"tion": 3,
	"sion": 3,
	"tian": 3,
	"cian": 3,
	"ture": 3,
	"sure": 3   ,
}

GRAPHEME_LETTER_TO_PHONEME_MAP = {
	"tion": [[0], [1], [1], [2]],   # t → ʃ, i/o → ə, n → n
	"sion": [[0], [1], [1], [2]],
	"tian": [[0], [1], [1], [2]],
	"cian": [[0], [1], [1], [2]],
	"ture": [[0], [1], [2], [2]],   # tu → tʃər or similar
	"sure": [[0], [1], [2], [2]]
}

def grapheme_expected_phoneme_count(grapheme: str) -> int:
	"""
	Ước lượng số IPA phoneme cho 1 grapheme.
	Mặc định 1. Các ngoại lệ phổ biến: 'x' → /ks/ (2), 'qu' → /kw/ (2).
	"""
	g = grapheme.lower()
	if g in GRAPHEME_EXPECTED_PHONEMES:
		return GRAPHEME_EXPECTED_PHONEMES[g]
	if g == "x":
		return 2
	if g == "qu":
		return 2
	return 1

def map_graphemes_to_phoneme_indices(graphemes, ipa_phonemes_count: int):
	"""
	Phân bổ index phoneme cho từng grapheme theo thứ tự trái→phải dựa trên số lượng ước lượng.
	Nếu tổng ước lượng != số phoneme thật, sẽ điều chỉnh nhẹ để khớp tổng:
	- Nếu thiếu: dồn phần thiếu vào grapheme cuối.
	- Nếu thừa: cắt bớt ở grapheme cuối cùng nhưng vẫn tối thiểu 1 phoneme cho grapheme nếu tổng cho phép.
	Trả về list các tuple (start_idx, end_idx_exclusive) cho mỗi grapheme.
	"""
	est_counts = [max(1, grapheme_expected_phoneme_count(g)) for g in graphemes]
	total_est = sum(est_counts)
	# Điều chỉnh để tổng = ipa_phonemes_count
	if total_est < ipa_phonemes_count and len(est_counts) > 0:
		est_counts[-1] += (ipa_phonemes_count - total_est)
	elif total_est > ipa_phonemes_count and len(est_counts) > 0:
		over = total_est - ipa_phonemes_count
		# Giảm ở cuối trước, đảm bảo >=1 nếu có thể
		for i in range(len(est_counts)-1, -1, -1):
			if over == 0:
				break
			can_reduce = est_counts[i] - 1
			if can_reduce > 0:
				reduce_by = min(can_reduce, over)
				est_counts[i] -= reduce_by
				over -= reduce_by
		# Nếu vẫn còn over (trường hợp grapheme ít hơn phoneme), sẽ cắt cứng ở cuối
		if sum(est_counts) > ipa_phonemes_count:
			est_counts[-1] -= (sum(est_counts) - ipa_phonemes_count)
	# Xây mapping theo tích lũy
	# QUAN TRỌNG: Đảm bảo grapheme cuối luôn được gán phoneme cuối
	mapping = []
	cursor = 0
	remaining_graphemes = len(est_counts)
	
	for idx, cnt in enumerate(est_counts):
		start = cursor
		remaining_graphemes -= 1
		
		# Nếu là grapheme cuối, đảm bảo nó được gán phoneme cuối
		if remaining_graphemes == 0:
			# Grapheme cuối luôn được gán từ cursor đến cuối
			end = ipa_phonemes_count
		else:
			# Các grapheme khác: phân bổ bình thường
			# Nhưng phải để lại ít nhất 1 phoneme cho grapheme cuối
			max_available = ipa_phonemes_count - 1  # Để lại 1 cho grapheme cuối
			if max_available < 0:
				max_available = 0
			end = min(cursor + max(0, cnt), max_available)
			# Nếu không có phoneme mới, share với grapheme trước (nếu có)
			if end <= start:
				if cursor > 0:
					# Share phoneme với grapheme trước
					end = cursor
					start = cursor - 1
				elif cursor < max_available:
					end = cursor + 1
				else:
					# Không có phoneme nào, sẽ được gán sau
					end = cursor
		
		mapping.append((start, end))
		cursor = end
		# Nếu đã hết phoneme (trừ grapheme cuối), các grapheme còn lại share phoneme cuối
		if cursor >= ipa_phonemes_count - 1 and remaining_graphemes > 0:
			# Các grapheme còn lại sẽ share phoneme cuối với grapheme cuối
			for _ in range(remaining_graphemes - 1):
				mapping.append((max(0, ipa_phonemes_count - 1), ipa_phonemes_count))
			break
	
	# Đảm bảo grapheme cuối luôn được gán phoneme cuối (nếu có phoneme)
	if mapping and ipa_phonemes_count > 0:
		last_start, last_end = mapping[-1]
		# Grapheme cuối phải được gán ít nhất 1 phoneme (phoneme cuối)
		if last_end <= last_start or last_end < ipa_phonemes_count:
			mapping[-1] = (max(0, ipa_phonemes_count - 1), ipa_phonemes_count)
	
	return mapping

def compare_ipa_pairs(real_and_transcribed_words_ipa, strict_syllable_match: bool = False, return_as_string: bool = False, real_words = None):
    """
    So sánh IPA và trả về kết quả.
    
    Args:
        real_and_transcribed_words_ipa: List of tuples (real_ipa, recorded_ipa)
        strict_syllable_match: Nếu True, chỉ match trong cùng syllable
        return_as_string: Nếu True, trả về string thay vì list
        real_words: List hoặc string các từ tiếng Anh tương ứng (để map theo số chữ cái)
                   - Nếu là string: sẽ tách thành list các từ
                   - Nếu là list: giữ nguyên
    """
    # Xử lý real_words nếu là string
    if isinstance(real_words, str):
        real_words = real_words.split()
    def ipa_compare(real_ipa, recorded_ipa, real_word=None):
        # Tách theo phoneme (nhận diện diphthong) thay vì từng ký tự
        ipa_units_real = split_ipa_into_phonemes(real_ipa)
        
        # Parse recorded IPA, keeping underscores for alignment tracking
        recorded_list = []
        for char in recorded_ipa:
            if char == '_':
                recorded_list.append('_')  # Keep underscore to mark gap
            elif char not in [' ', '\t']:  # Skip spaces
                recorded_list.append(char)
        
        # Tách recorded IPA thành phoneme (bỏ underscore khi tách phoneme)
        recorded_clean = recorded_ipa.replace('_', '').replace(' ', '')
        recorded_phonemes_clean = split_ipa_into_phonemes(recorded_clean)
        
        # Tách theo âm tiết trước để so sánh
        real_syllables = split_ipa_into_syllables(real_ipa)
        recorded_syllables = split_ipa_into_syllables(recorded_ipa)
        
        # Tạo ipa_units_recorded từ recorded_phonemes_clean với underscore
        # Map từ character list sang phoneme list
        ipa_units_recorded = []
        recorded_phoneme_idx = 0
        
        i = 0
        while i < len(recorded_list):
            if recorded_list[i] == '_':
                ipa_units_recorded.append('_')
                i += 1
            else:
                # Tìm phoneme chứa ký tự này
                if recorded_phoneme_idx < len(recorded_phonemes_clean):
                    phoneme = recorded_phonemes_clean[recorded_phoneme_idx]
                    # Kiểm tra xem ký tự hiện tại có phải là ký tự đầu của phoneme này không
                    if recorded_list[i] == phoneme[0]:
                        ipa_units_recorded.append(phoneme)
                        # Bỏ qua các ký tự còn lại của phoneme này (trừ underscore)
                        chars_consumed = 0
                        for j in range(i + 1, min(i + len(phoneme), len(recorded_list))):
                            if recorded_list[j] == '_':
                                break
                            if j < len(phoneme) and recorded_list[j] == phoneme[j - i]:
                                chars_consumed += 1
                            else:
                                break
                        i += chars_consumed + 1
                        recorded_phoneme_idx += 1
                    else:
                        # Ký tự không khớp với phoneme hiện tại, thử tìm phoneme khác
                        # Hoặc đơn giản là một ký tự đơn
                        found_phoneme = False
                        for ph_idx, ph in enumerate(recorded_phonemes_clean):
                            if recorded_list[i] == ph[0] and ph_idx >= recorded_phoneme_idx:
                                # Nếu có phoneme khác bắt đầu bằng ký tự này
                                ipa_units_recorded.append(ph)
                                chars_consumed = 0
                                for j in range(i + 1, min(i + len(ph), len(recorded_list))):
                                    if recorded_list[j] == '_':
                                        break
                                    if j - i < len(ph) and recorded_list[j] == ph[j - i]:
                                        chars_consumed += 1
                                    else:
                                        break
                                i += chars_consumed + 1
                                recorded_phoneme_idx = ph_idx + 1
                                found_phoneme = True
                                break
                        
                        if not found_phoneme:
                            # Ký tự đơn, không thuộc phoneme nào
                            ipa_units_recorded.append(recorded_list[i])
                            i += 1
                else:
                    # Không còn phoneme nào, thêm ký tự đơn
                    ipa_units_recorded.append(recorded_list[i])
                    i += 1
        
        # Tạo mapping: mỗi phoneme trong real IPA thuộc âm tiết nào
        # Cần map từ phoneme list sang syllable
        phoneme_to_syllable = {}
        phoneme_idx = 0
        for syl_idx, syllable in enumerate(real_syllables):
            # Tách syllable thành phoneme để đếm
            syl_phonemes = split_ipa_into_phonemes(syllable)
            for i in range(len(syl_phonemes)):
                if phoneme_idx < len(ipa_units_real):
                    phoneme_to_syllable[phoneme_idx] = syl_idx
                    phoneme_idx += 1
        
        # So sánh từng âm tiết với recorded
        syllable_results = {}
        recorded_syl_used = [False] * len(recorded_syllables)
        
        for syl_idx, real_syllable in enumerate(real_syllables):
            # Tách real syllable thành phoneme
            real_syl_phonemes = split_ipa_into_phonemes(real_syllable)
            
            # Ưu tiên match theo thứ tự: syllable 1 với syllable 1, syllable 2 với syllable 2, ...
            # Chỉ khi không match ở đúng vị trí mới tìm ở vị trí khác
            best_match_idx = None
            best_match_score = 0
            
            # Đầu tiên, thử match với recorded syllable ở cùng vị trí
            if syl_idx < len(recorded_syllables) and not recorded_syl_used[syl_idx]:
                recorded_syllable = recorded_syllables[syl_idx]
                recorded_syl_phonemes = split_ipa_into_phonemes(recorded_syllable)
                
                # So sánh âm tiết: đếm số phoneme giống nhau ở đúng vị trí
                match_count = 0
                min_len = min(len(real_syl_phonemes), len(recorded_syl_phonemes))
                for i in range(min_len):
                    if real_syl_phonemes[i] == recorded_syl_phonemes[i]:
                        match_count += 1
                
                if match_count > 0:
                    best_match_score = match_count
                    best_match_idx = syl_idx
            
            # Nếu không match ở đúng vị trí, tìm ở vị trí khác
            if best_match_idx is None:
                for rec_syl_idx, recorded_syllable in enumerate(recorded_syllables):
                    if recorded_syl_used[rec_syl_idx]:
                        continue
                    
                    # Tách recorded syllable thành phoneme
                    recorded_syl_phonemes = split_ipa_into_phonemes(recorded_syllable)
                    
                    # So sánh âm tiết: đếm số phoneme giống nhau (theo thứ tự)
                    match_count = 0
                    min_len = min(len(real_syl_phonemes), len(recorded_syl_phonemes))
                    for i in range(min_len):
                        if real_syl_phonemes[i] == recorded_syl_phonemes[i]:
                            match_count += 1
                    
                    if match_count > best_match_score:
                        best_match_score = match_count
                        best_match_idx = rec_syl_idx
            
            # Nếu tìm thấy match tốt, đánh dấu đã dùng
            if best_match_idx is not None and best_match_score > 0:
                syllable_results[syl_idx] = best_match_idx
                recorded_syl_used[best_match_idx] = True
        
        # So sánh từng phoneme: nếu âm tiết match, so sánh phoneme trong âm tiết đó
        result = []
        recorded_phoneme_idx = 0
        
        for real_idx, real_phoneme in enumerate(ipa_units_real):
            syl_idx = phoneme_to_syllable.get(real_idx, -1)
            found_match = False
            
            # Nếu âm tiết này có match trong recorded, so sánh phoneme cụ thể
            if syl_idx in syllable_results:
                rec_syl_idx = syllable_results[syl_idx]
                real_syllable = real_syllables[syl_idx]
                recorded_syllable = recorded_syllables[rec_syl_idx]
                
                # Tách thành phoneme để so sánh
                real_syl_phonemes = split_ipa_into_phonemes(real_syllable)
                recorded_syl_phonemes = split_ipa_into_phonemes(recorded_syllable)
                
                # Tìm vị trí của phoneme này trong âm tiết real
                real_syl_start = sum(len(split_ipa_into_phonemes(real_syllables[i])) for i in range(syl_idx))
                offset_in_syl = real_idx - real_syl_start
                
                # So sánh với phoneme tương ứng trong recorded syllable
                # CHỈ match ở đúng vị trí offset, không tìm từ vị trí sau
                # Điều này đảm bảo thứ tự chính xác: mỗi phoneme phải match ở đúng vị trí
                if offset_in_syl < len(recorded_syl_phonemes):
                    recorded_phoneme = recorded_syl_phonemes[offset_in_syl]
                    if recorded_phoneme == real_phoneme:
                        result.append('1')
                        found_match = True
                        # Cập nhật recorded_phoneme_idx để tránh match lại phoneme đã dùng
                        # Tìm vị trí của phoneme này trong ipa_units_recorded (bỏ qua underscore)
                        # Tính vị trí bắt đầu của recorded syllable trong ipa_units_recorded (không tính underscore)
                        recorded_syl_start_in_units = 0
                        for prev_syl_idx in range(rec_syl_idx):
                            prev_recorded_syl = recorded_syllables[prev_syl_idx]
                            prev_recorded_syl_phonemes = split_ipa_into_phonemes(prev_recorded_syl)
                            recorded_syl_start_in_units += len(prev_recorded_syl_phonemes)
                        recorded_phoneme_pos_in_units = recorded_syl_start_in_units + offset_in_syl
                        # Tìm vị trí thực tế trong ipa_units_recorded (bỏ qua underscore)
                        actual_pos = 0
                        non_underscore_count = 0
                        for idx, unit in enumerate(ipa_units_recorded):
                            if unit != '_':
                                if non_underscore_count == recorded_phoneme_pos_in_units:
                                    actual_pos = idx
                                    break
                                non_underscore_count += 1
                        if actual_pos < len(ipa_units_recorded):
                            recorded_phoneme_idx = actual_pos + 1
                    # Nếu không match ở đúng vị trí, không tìm ở vị trí khác
                    # Điều này đảm bảo: nếu phoneme sai ở vị trí đó, phải là '0'
                    # Cập nhật recorded_phoneme_idx để bỏ qua phoneme ở vị trí offset
                    if not found_match:
                        recorded_syl_start_in_units = 0
                        for prev_syl_idx in range(rec_syl_idx):
                            prev_recorded_syl = recorded_syllables[prev_syl_idx]
                            prev_recorded_syl_phonemes = split_ipa_into_phonemes(prev_recorded_syl)
                            recorded_syl_start_in_units += len(prev_recorded_syl_phonemes)
                        recorded_phoneme_pos_in_units = recorded_syl_start_in_units + offset_in_syl
                        # Tìm vị trí thực tế trong ipa_units_recorded (bỏ qua underscore)
                        actual_pos = 0
                        non_underscore_count = 0
                        for idx, unit in enumerate(ipa_units_recorded):
                            if unit != '_':
                                if non_underscore_count == recorded_phoneme_pos_in_units:
                                    actual_pos = idx
                                    break
                                non_underscore_count += 1
                        if actual_pos < len(ipa_units_recorded):
                            # Chỉ cập nhật nếu recorded_phoneme_idx chưa vượt quá vị trí này
                            if recorded_phoneme_idx <= actual_pos:
                                recorded_phoneme_idx = actual_pos + 1
            
            # KHÔNG cho phép fallback match từ vị trí xa
            # Nếu không match trong syllable, phải là '0'
            # Điều này đảm bảo tính chính xác: mỗi phoneme chỉ match trong syllable tương ứng
            if not found_match:
                result.append('0')
                # Khi không match trong syllable, vẫn cần cập nhật recorded_phoneme_idx
                # để bỏ qua phoneme tương ứng trong recorded và tiếp tục với phoneme tiếp theo
                if syl_idx in syllable_results:
                    rec_syl_idx = syllable_results[syl_idx]
                    recorded_syllable = recorded_syllables[rec_syl_idx]
                    recorded_syl_phonemes = split_ipa_into_phonemes(recorded_syllable)
                    real_syl_start = sum(len(split_ipa_into_phonemes(real_syllables[i])) for i in range(syl_idx))
                    offset_in_syl = real_idx - real_syl_start
                    # Nếu offset hợp lệ và chưa vượt quá recorded syllable, tăng recorded_phoneme_idx
                    # để bỏ qua phoneme ở vị trí offset trong recorded syllable
                    if offset_in_syl < len(recorded_syl_phonemes):
                        recorded_syl_start_in_units = 0
                        for prev_syl_idx in range(rec_syl_idx):
                            prev_recorded_syl = recorded_syllables[prev_syl_idx]
                            prev_recorded_syl_phonemes = split_ipa_into_phonemes(prev_recorded_syl)
                            recorded_syl_start_in_units += len(prev_recorded_syl_phonemes)
                        recorded_phoneme_pos_in_units = recorded_syl_start_in_units + offset_in_syl
                        # Tìm vị trí thực tế trong ipa_units_recorded (bỏ qua underscore)
                        actual_pos = 0
                        non_underscore_count = 0
                        for idx, unit in enumerate(ipa_units_recorded):
                            if unit != '_':
                                if non_underscore_count == recorded_phoneme_pos_in_units:
                                    actual_pos = idx
                                    break
                                non_underscore_count += 1
                        if actual_pos < len(ipa_units_recorded):
                            # Luôn tăng để bỏ qua phoneme này trong recorded
                            recorded_phoneme_idx = actual_pos + 1
		
        # Nếu có real_word, map từ phoneme sang chữ cái theo grapheme
        if real_word:
            phoneme_result = list(result)        # list các '1'/'0' cho mỗi phoneme chuẩn
            ipa_phoneme_count = len(ipa_units_real)
            graphemes = segment_word_into_graphemes(real_word)
            # Map grapheme → dải index phoneme
            ranges = map_graphemes_to_phoneme_indices(graphemes, ipa_phoneme_count)
            # Nếu mapping bất hợp lệ về kích thước, fallback heuristic cũ
            if len(ranges) != len(graphemes):
                # Fallback: giữ nguyên logic cũ nếu có trục trặc
                letter_result = []
                if len(real_word) == len(phoneme_result):
                    letter_result = phoneme_result
                elif len(real_word) > len(phoneme_result):
                    if len(phoneme_result) == 1:
                        letter_result = [phoneme_result[0]] * len(real_word)
                    else:
                        letter_result.append(phoneme_result[0])
                        middle_letters = len(real_word) - 2
                        middle_phonemes = len(phoneme_result) - 2
                        if middle_letters > 0 and middle_phonemes > 0:
                            for i in range(1, len(real_word) - 1):
                                letter_pos_in_middle = i - 1
                                ph_idx = 1 + int(letter_pos_in_middle * middle_phonemes / middle_letters)
                                if ph_idx >= len(phoneme_result) - 1:
                                    ph_idx = len(phoneme_result) - 2
                                letter_result.append(phoneme_result[ph_idx])
                        elif middle_letters > 0:
                            for i in range(1, len(real_word) - 1):
                                if i == len(real_word) - 2:
                                    letter_result.append(phoneme_result[-1])
                                else:
                                    letter_result.append('0')
                        letter_result.append(phoneme_result[-1])
                else:
                    ratio = len(phoneme_result) / len(real_word)
                    for i in range(len(real_word)):
                        start_idx = int(i * ratio)
                        end_idx = int((i + 1) * ratio)
                        if start_idx < len(phoneme_result):
                            if any(phoneme_result[j] == '1' for j in range(start_idx, min(end_idx, len(phoneme_result)))):
                                letter_result.append('1')
                            else:
                                letter_result.append('0')
                        else:
                            letter_result.append('0')
                return ''.join(letter_result)
            # Tạo kết quả theo quy tắc:
            # - Grapheme 1 chữ, 1 phoneme → dùng trực tiếp
            # - Grapheme nhiều chữ, 1 phoneme → nhân kết quả ra số chữ (vd 'io'→ một IPA, sai → '00')
            # - Grapheme 1 chữ, nhiều phoneme → tất cả phải đúng mới là '1', nếu có 1 sai → '0'
            # - Grapheme nhiều chữ, nhiều phoneme → tất cả phoneme của grapheme phải đúng; nhân ra theo số chữ
            letter_result = []
            for g, (start, end) in zip(graphemes, ranges):
                assigned = phoneme_result[start:end] if start < end else []
                g_lower = g.lower()
                if g_lower in GRAPHEME_LETTER_TO_PHONEME_MAP and assigned:
                    per_letter_idx = GRAPHEME_LETTER_TO_PHONEME_MAP[g_lower]
                    # Điều chỉnh nếu mapping không khớp độ dài
                    if len(per_letter_idx) != len(g):
                        per_letter_idx = per_letter_idx[:len(g)]
                        if len(per_letter_idx) < len(g):
                            per_letter_idx.extend([[len(assigned)-1]] * (len(g) - len(per_letter_idx)))
                    for idxs in per_letter_idx:
                        # idxs là list index phoneme tương ứng với chữ
                        if not idxs:
                            letter_result.append('0')
                            continue
                        values = []
                        for ix in idxs:
                            if 0 <= ix < len(assigned):
                                values.append(assigned[ix])
                        letter_result.append('1' if values and all(v == '1' for v in values) else '0')
                else:
                    # Nếu không có mapping cụ thể
                    if not assigned:
                        g_scores = ['0'] * len(g)
                    elif len(assigned) == 1:
                        g_scores = [assigned[0]] * len(g)
                    else:
                        # Phân bố đều phoneme cho chữ
                        g_scores = []
                        total_phonemes = len(assigned)
                        for letter_idx in range(len(g)):
                            start_idx = (letter_idx * total_phonemes) // len(g)
                            end_idx = ((letter_idx + 1) * total_phonemes) // len(g)
                            if end_idx <= start_idx:
                                end_idx = start_idx + 1 if start_idx < total_phonemes else total_phonemes
                            segment = assigned[start_idx:end_idx]
                            g_scores.append('1' if segment and all(x == '1' for x in segment) else '0')
                    letter_result.extend(g_scores)
            # Nếu vì điều chỉnh mapping khiến số chữ ≠ độ dài thật của từ (hiếm), cắt/đệm cho khớp
            if len(letter_result) > len(real_word):
                letter_result = letter_result[:len(real_word)]
            elif len(letter_result) < len(real_word):
                letter_result.extend(['0'] * (len(real_word) - len(letter_result)))
            return ''.join(letter_result)

        return ''.join(result)

    results = []
    for idx, (real, recorded) in enumerate(real_and_transcribed_words_ipa):
        # Lấy từ tương ứng từ real_words
        real_word = None
        if real_words:
            if isinstance(real_words, list) and idx < len(real_words):
                real_word = real_words[idx]
            elif isinstance(real_words, str):
                # Nếu là string, tách và lấy từ tương ứng
                words_list = real_words.split()
                if idx < len(words_list):
                    real_word = words_list[idx]
        results.append(ipa_compare(real, recorded, real_word))

    if return_as_string:
        return ' '.join(results)
    return results
