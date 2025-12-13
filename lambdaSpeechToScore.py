
import torch
import json
import os
import WordMatching as wm
import utilsFileIO
import pronunciationTrainer
import base64
import time
import audioread
import numpy as np
from torchaudio.transforms import Resample
import io
import tempfile
import function
import tempfile
import os
import subprocess
from pydub import AudioSegment
# import librosa
import numpy as np
# from scipy.signal import butter, lfilter
import soundfile as sf
trainer_SST_lambda = {}
trainer_SST_lambda['en'] = pronunciationTrainer.getTrainer("en")
trainer_SST_lambda['en-gb'] = pronunciationTrainer.getTrainer("en-gb")
transform = Resample(orig_freq=48000, new_freq=16000)


def lambda_handler(event, context):

    data = json.loads(event['body'])

    real_text = data['title']
    file_bytes = base64.b64decode(
        data['base64Audio'][22:].encode('utf-8'))
    language = data['language']

    if len(real_text) == 0:
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

    tmp_ogg = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    tmp_ogg_name = tmp_ogg.name
    tmp_ogg.write(file_bytes)
    tmp_ogg.flush()
    tmp_ogg.close()

    signal, fs = audioread_load(tmp_ogg_name)
    tmp_wav_path = tmp_ogg.name.replace(".ogg", ".wav")
  # 🔹 Dùng ffmpeg để convert .ogg -> .wav
    subprocess.run([
        "ffmpeg",
        "-i", tmp_ogg.name,   # ⚠️ phải là .name (string path)
        "-ar", "16000",       # tần số mẫu 16kHz
        "-ac", "1",           # âm thanh mono
        tmp_wav_path,
        "-y"                  # ghi đè nếu file tồn tại
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=True  # đảm bảo ffmpeg báo lỗi nếu thất bại
    )
    
    # 🔹 Lọc tạp âm trước khi xử lý
    # print("🔧 Đang lọc tạp âm...")
    # try:
    #     cleaned_audio, cleaned_sr = clean_voice(tmp_wav_path)
    #     # Lưu lại file đã lọc vào tmp_wav_path (ghi đè)
    #     sf.write(tmp_wav_path, cleaned_audio, cleaned_sr)
    #     # Cập nhật signal từ file đã lọc để đồng bộ
    #     signal = cleaned_audio
    #     fs = cleaned_sr
    #     print("✅ Đã lọc tạp âm xong")
    # except Exception as e:
    #     print(f"⚠️ Lỗi khi lọc tạp âm: {e}, tiếp tục với file gốc")
    #     # Nếu lỗi, tiếp tục với file gốc (signal và fs đã có sẵn)
    
    try:
        print("🔍 Đang gọi model để chấm điểm...")
        # ✅ Gọi model xử lý từ đường dẫn file .wav (đã được lọc tạp âm)
        # Resample signal về 16kHz nếu cần (file đã được convert về 16kHz bằng ffmpeg)
        if fs != 16000:
            signal_tensor = transform(torch.Tensor(signal)).unsqueeze(0)
        else:
            signal_tensor = torch.Tensor(signal).unsqueeze(0)
        result = trainer_SST_lambda[language].processAudioForGivenText(
            tmp_wav_path, signal_tensor, real_text, language
        )
    finally:
        # Dọn file tạm .wav sau khi xong
        os.remove(tmp_wav_path)
        os.remove(tmp_ogg_name)

    start = time.time()
    real_transcripts_ipa = ' '.join(
        [word[0] for word in result['real_and_transcribed_words_ipa']])
    matched_transcripts_ipa = ' '.join(
        [word[1] for word in result['real_and_transcribed_words_ipa']])
    real_and_transcribed_words_ipa = result['real_and_transcribed_words_ipa']
    print(4)
    print(real_and_transcribed_words_ipa)
    real_transcripts = ' '.join(
        [word[0] for word in result['real_and_transcribed_words']])
    matched_transcripts = ' '.join(
        [word[1] for word in result['real_and_transcribed_words']])

    words_real = real_transcripts.lower().split()
    mapped_words = matched_transcripts.split()
    is_letter_correct_all_words = ''
    is_letter_correct_all_words = function.compare_ipa_pairs(
        real_and_transcribed_words_ipa,
        return_as_string=True,
        real_words=result['real_text']
    )

    total_letters, correct_letters, letters_accuracy_percent = calculate_letter_accuracy(
        is_letter_correct_all_words
    )

    
    # for idx, word_real in enumerate(words_real):

    #     mapped_letters, mapped_letters_indices = wm.get_best_mapped_words(
    #         mapped_words[idx], word_real)
    #     # is_letter_correct  =  wm.getWhichPhomenesWereTranscribedCorrectly(real_ipa,recorded_ipa)
    #     is_letter_correct = wm.getWhichLettersWereTranscribedCorrectly(
    #         word_real, mapped_letters)
    #     is_letter_correct_all_words += ''.join([str(is_correct)
    #                                             for is_correct in is_letter_correct]) + ' '
    print("Debug - is_letter_correct_all_words:", is_letter_correct_all_words)
    pair_accuracy_category = ' '.join(
        [str(category) for category in result['pronunciation_categories']])
    print('Time to post-process results: ', str(time.time()-start))
    
    res = {'real_transcript': result['recording_transcript'],
           'ipa_transcript': result['recording_ipa'],
           #'pronunciation_accuracy': str(int(result['pronunciation_accuracy'])),
           'pronunciation_accuracy': letters_accuracy_percent,
           'real_transcripts': real_transcripts, 'matched_transcripts': matched_transcripts,
           'real_transcripts_ipa': real_transcripts_ipa, 'matched_transcripts_ipa': matched_transcripts_ipa,
           'pair_accuracy_category': pair_accuracy_category,
           'start_time': result['start_time'],
           'end_time': result['end_time'],
           'is_letter_correct_all_words': is_letter_correct_all_words,
           'AIFeedback': result['AIFeedback']}
    print("Debug - result:", res)
    return json.dumps(res)



# Tạo bộ lọc Butterworth
# def calculate_letter_accuracy(letter_string: str):
#     """Return total letters, correct letters, and accuracy percent from 1/0 string."""
#     if not letter_string:
#         return 0, 0, 0

#     segments = [segment for segment in letter_string.strip().split(' ') if segment]
#     total_letters = sum(len(segment) for segment in segments)
#     correct_letters = sum(segment.count('1') for segment in segments)

#     if total_letters == 0:
#         accuracy_percent = 0
#     else:
#         accuracy_percent = round((correct_letters / total_letters) * 100)

#     return total_letters, correct_letters, accuracy_percent


# def butter_filter(data, cutoff, sr, btype, order=4):
#     nyq = 0.5 * sr
#     normal_cutoff = cutoff / nyq
    
#     # Đảm bảo normal_cutoff nằm trong khoảng hợp lệ (0 < Wn < 1)
#     if normal_cutoff >= 1.0:
#         # Nếu cutoff >= Nyquist, giảm xuống 95% của Nyquist để an toàn
#         normal_cutoff = 0.95
#     elif normal_cutoff <= 0:
#         # Nếu cutoff <= 0, đặt giá trị tối thiểu
#         normal_cutoff = 0.01
    
#     b, a = butter(order, normal_cutoff, btype=btype)
#     return lfilter(b, a, data)

# Lọc tạp âm nâng cao
# def clean_voice(path):
#     """
#     Lọc tạp âm từ file audio:
#     - High-pass filter để giảm rung nền
#     - Low-pass filter để giảm hiss
#     - Noise gate để loại bỏ tín hiệu yếu
#     """
#     y, sr = librosa.load(path, sr=None)

#     # High-pass để giảm rung nền
#     y = butter_filter(y, 80, sr, "high")

#     # Low-pass để giảm hiss (đảm bảo cutoff < Nyquist frequency)
#     # Với sr=16kHz, Nyquist=8kHz, nên dùng 7000 Hz để an toàn
#     lowpass_cutoff = min(7000, 0.9 * (sr / 2))
#     y = butter_filter(y, lowpass_cutoff, sr, "low")

#     # Noise gate: loại bỏ tín hiệu yếu hơn ngưỡng
#     y = np.where(np.abs(y) < 0.015, 0, y)

#     return y, sr

def audioread_load(path, offset=0.0, duration=None, dtype=np.float32, text=None):
    """Load an audio buffer using audioread.

    This loads one block at a time, and then concatenates the results.
    """

    y = []
    with audioread.audio_open(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels
        print(3)
        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + \
                (int(np.round(sr_native * duration)) * n_channels)

        n = 0

        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[: s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

        # 🔹 Nếu có truyền text => tự động phân tích stress
    # stress = None
    # if text is not None:
    #     print("🔍 Đang phân tích trọng âm...")
    #     stress = detect_stress(y, sr_native, text)
    return y, sr_native

def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)
