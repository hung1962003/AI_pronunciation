
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
    try:
        print("🔍 Đang gọi model để chấm điểm...")
        # ✅ Gọi model xử lý từ đường dẫn file .wav
        signal = transform(torch.Tensor(signal)).unsqueeze(0)
        result = trainer_SST_lambda[language].processAudioForGivenText(
            tmp_wav_path, signal , real_text, language
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
    is_letter_correct_all_words = function.compare_ipa_pairs(real_and_transcribed_words_ipa, return_as_string=True,real_words=result['real_text'])
    # 

    
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
           'pronunciation_accuracy': str(int(result['pronunciation_accuracy'])),
           'real_transcripts': real_transcripts, 'matched_transcripts': matched_transcripts,
           'real_transcripts_ipa': real_transcripts_ipa, 'matched_transcripts_ipa': matched_transcripts_ipa,
           'pair_accuracy_category': pair_accuracy_category,
           'start_time': result['start_time'],
           'end_time': result['end_time'],
           'is_letter_correct_all_words': is_letter_correct_all_words,
           'AIFeedback': result['AIFeedback']}
    print("Debug - result:", res)
    return json.dumps(res)




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
