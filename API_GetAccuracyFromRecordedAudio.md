## API `/GetAccuracyFromRecordedAudio`

### 1. Mục đích

**Chấm điểm độ chính xác phát âm** cho một câu tiếng Anh dựa trên:
- Text chuẩn (người học cần đọc).
- File ghi âm giọng đọc của người học.

API trả về:
- Transcript hệ thống nghe được.
- IPA transcript.
- Điểm phát âm theo chữ cái (letter-level).
- Thời gian bắt đầu/kết thúc từng từ.
- Feedback AI cho từng từ sai.

---

### 2. Phương thức & URL

- **Method**: `POST`
- **Path**: `/GetAccuracyFromRecordedAudio`
- **Content-Type**: `application/json`

---

### 3. Request body

```json
{
  "title": "This is a sample sentence",
  "base64Audio": "data:audio/ogg;base64,AAA...",
  "language": "en"
}
```

- **title**:  
  - Câu tiếng Anh chuẩn mà học viên cần đọc.  
  - Dạng string, ví dụ `"good"` hoặc `"This is a sample sentence"`.

- **base64Audio**:  
  - Chuỗi audio **OGG** được encode base64 **kèm prefix data URL**.  
  - Hệ thống sẽ bỏ `data['base64Audio'][22:]` trước khi decode, nên phía client cần gửi đúng dạng:
    - Ví dụ: `"data:audio/ogg;base64,AAAAIGZ0eXBv..."`.

- **language**:
  - Ngôn ngữ mô hình ASR/IPA:
    - `"en"`: tiếng Anh Mỹ.
    - `"en-gb"`: tiếng Anh Anh (UK).

---

### 4. Luồng xử lý nội bộ (tóm tắt theo code)

1. **FastAPI nhận request** tại `webApp.py`:
   - Endpoint: `@app.post('/GetAccuracyFromRecordedAudio')`.
   - Đọc JSON body → gói vào `event = {'body': json.dumps(body)}`.
   - Gọi `lambdaSpeechToScore.lambda_handler(event, [])`.

2. **Giải mã & chuẩn bị audio** (`lambdaSpeechToScore.lambda_handler`):
   - Parse body:
     - `real_text = data['title']`
     - `file_bytes = base64.b64decode(data['base64Audio'][22:])`
     - `language = data['language']`
   - Nếu `real_text` rỗng → trả về body rỗng.
   - Ghi `file_bytes` thành file tạm `.ogg`.
   - Dùng `ffmpeg` chuyển `.ogg` → `.wav` 16kHz, mono:
     - Lệnh: `ffmpeg -i input.ogg -ar 16000 -ac 1 output.wav -y`.
   - Đọc audio bằng `audioread_load` để có `signal, fs`.
   - Nếu `fs != 16000`, resample về 16kHz.

3. **Chấm điểm phát âm** (`pronunciationTrainer.PronunciationTrainer.processAudioForGivenText`):

   - Gọi `function.get_phonemes(tmp_wav_path)`:
     - Gửi file `.wav` lên Hugging Face Space (`lgtitony/doan`) để nhận **IPA transcript** (`recording_ipa`).
   - Gọi `function.ipa_to_english(recording_ipa, real_text)`:
     - Chuyển IPA về **transcript tiếng Anh** gần nhất (`recording_transcript`) dựa trên từ trong `real_text`.
   - Gọi `self.getAudioTranscript(recordedAudio1)`:
     - Chạy ASR Whisper (hoặc model khác tuỳ `models.getASRModel`) để lấy **vị trí từng từ** trong audio.
   - Lấy IPA reference:
     - `reference_ipa = RuleBasedModels.get_phonem_converter(language).convertToPhonem(real_text)`.
   - Căn chỉnh text chuẩn vs text ghi âm:
     - Nếu `recording_transcript` quá ngắn/rỗng → fallback: mỗi từ chuẩn map với dấu `'-'`.
     - Ngược lại dùng `function.align_real_and_transcribed(real_text, recording_transcript)` để ghép cặp từ.
   - Tạo cặp IPA theo từng từ:
     - `real_and_transcribed_words_ipa = function.getComparationPhonemes(reference_ipa, recording_ipa)`.
   - Tính độ chính xác phát âm theo **phoneme**:
     - `getPronunciationAccuracy(real_and_transcribed_words_ipa)`:
       - Dùng `function.compare_ipa_pairs` (so sánh IPA theo phoneme + âm tiết).
       - Tạo list độ chính xác từng từ (`current_words_pronunciation_accuracy`).
   - Gán category từng từ (good/medium/bad) dựa trên threshold.
   - Gọi `function.aiFeedback(real_text, real_and_transcribed_words_ipa)` để sinh feedback chi tiết.
   - Trả về:
     - `recording_transcript`
     - `real_and_transcribed_words`
     - `recording_ipa`
     - `start_time`, `end_time` theo từng từ
     - `real_and_transcribed_words_ipa`
     - `pronunciation_categories`
     - `real_text`
     - `AIFeedback`

4. **Tính điểm theo chữ cái (letter-level)** (`lambdaSpeechToScore.lambda_handler`):

   - Ghép lại IPA theo câu:
     - `real_transcripts_ipa`, `matched_transcripts_ipa`.
   - Ghép lại transcript theo câu:
     - `real_transcripts`, `matched_transcripts`.
   - Gọi:
     ```python
     is_letter_correct_all_words = function.compare_ipa_pairs(
         real_and_transcribed_words_ipa,
         return_as_string=True,
         real_words=result['real_text']
     )
     ```
     - Output: chuỗi dạng `"1110 101"`:
       - Mỗi chữ cái trong `real_text` → `'1'` nếu phát âm đúng, `'0'` nếu sai.
       - Tách từng từ bằng khoảng trắng.
   - Gọi `calculate_letter_accuracy(is_letter_correct_all_words)`:
     - Đếm tổng số chữ và số chữ đúng → tính `%` làm `letters_accuracy_percent`.

5. **Trả kết quả ra FastAPI**:

   - Tạo object:
     ```python
     res = {
       "real_transcript": result["recording_transcript"],
       "ipa_transcript": result["recording_ipa"],
       "pronunciation_accuracy": letters_accuracy_percent,
       "real_transcripts": real_transcripts,
       "matched_transcripts": matched_transcripts,
       "real_transcripts_ipa": real_transcripts_ipa,
       "matched_transcripts_ipa": matched_transcripts_ipa,
       "pair_accuracy_category": pair_accuracy_category,
       "start_time": result["start_time"],
       "end_time": result["end_time"],
       "is_letter_correct_all_words": is_letter_correct_all_words,
       "AIFeedback": result["AIFeedback"]
     }
     ```
   - `lambdaSpeechToScore.lambda_handler` trả về `json.dumps(res)` (string JSON).
   - `webApp.py` nhận kết quả:
     - Nếu là string → `json.loads` và trả thẳng cho client.
     - Nếu là dict có `body` → parse `body`.

---

### 5. Response mẫu

```json
{
  "real_transcript": "this is a sample sentence",
  "ipa_transcript": "ðɪs ɪz ə ˈsæmpəl ˈsɛntəns",
  "pronunciation_accuracy": 86,
  "real_transcripts": "this is a sample sentence",
  "matched_transcripts": "this is a sample sentence",
  "real_transcripts_ipa": "ðɪs ɪz ə ˈsæmpəl ˈsɛntəns",
  "matched_transcripts_ipa": "ðɪs ɪz ə ˈsæmpəl ˈsɛntəns",
  "pair_accuracy_category": "0 0 1 0 0",
  "start_time": "0.1 0.35 0.6 0.9 1.2",
  "end_time": "0.3 0.55 0.8 1.1 1.4",
  "is_letter_correct_all_words": "1110 101 11",
  "AIFeedback": "Word 'good': bạn đọc /ɡuɪt/, cần đọc /gʊd/. Giữ âm /ʊ/ ngắn, không chuyển thành /uː/ + /ɪ/..."
}
```

**Lưu ý:**
- `pair_accuracy_category`: mã hoá mức độ cho từng từ (theo `pronunciation_categories`), tuỳ logic trong `PronunciationTrainer`.
- `is_letter_correct_all_words`:  
  - Ví dụ `good` chuẩn IPA `/gʊd/`, nếu bạn đọc `/ɡuɪt/` và hệ thống thấy:
    - `g` đúng → `'1'`
    - `o`, `o`, `d` sai → `'0'`
    - Chuỗi: `"1000"`.

---

### 6. Các lỗi thường gặp & cách xử lý

- **Thiếu hoặc rỗng `title`**:
  - Hệ thống trả về:
    ```json
    {
      "statusCode": 200,
      "body": ""
    }
    ```
  - Client nên kiểm tra và chặn từ UI nếu người dùng chưa nhập câu mẫu.

- **`base64Audio` sai format (thiếu prefix, không phải OGG)**:
  - `ffmpeg` có thể lỗi, lambda ném exception, FastAPI trả về:
    ```json
    {
      "statusCode": 200,
      "body": ""
    }
    ```
    hoặc response rỗng/không parse được JSON.
  - Nên log thêm và validate ở phía client.

- **Ngôn ngữ không hỗ trợ**:
  - Nếu `language` khác `"en"` / `"en-gb"` → `pronunciationTrainer.getTrainer` sẽ raise `ValueError`.
  - Nên giới hạn chọn ngôn ngữ trên UI.

---

### 7. Tóm tắt tích hợp client

1. Ghi âm audio (OGG, 16kHz mono nếu có thể) trên frontend.
2. Convert sang base64 và thêm prefix:
   - `"data:audio/ogg;base64," + base64String`.
3. Gửi `POST /GetAccuracyFromRecordedAudio` với body:
   - `title`: câu chuẩn.
   - `base64Audio`: chuỗi ở bước 2.
   - `language`: `"en"` hoặc `"en-gb"`.
4. Đọc `pronunciation_accuracy` để hiển thị điểm tổng.
5. Dùng:
   - `is_letter_correct_all_words` để tô màu từng chữ cái.
   - `start_time`/`end_time` + `pair_accuracy_category` để highlight từng từ.
   - `AIFeedback` để show gợi ý sửa phát âm.


