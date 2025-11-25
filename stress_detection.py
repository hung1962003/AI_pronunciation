# import eng_to_ipa as ipa
# import librosa
# import numpy as np


# # def detect_stress(audio_path, sentence):
    
# def detect_stress(y, sr, sentence):
#     """
#     Phân tích trọng âm cho toàn câu (nhiều từ).
#     Mỗi từ sẽ được so sánh với vị trí stress trong IPA.
#     """
#     # 1️⃣ Load audio
#     # y, sr = librosa.load(audio_path, sr=16000)
#     # y, sr = audioread_load(audio_path)
#     # Giới hạn tối đa 10 giây để tránh lag
#     if len(y) > sr * 10:
#         y = y[:sr * 10]

#     # 2️⃣ Chia năng lượng theo từng từ (ước lượng)
#     words = sentence.split()
#     n = len(words)
#     rms = librosa.feature.rms(y=y)[0]
#     step = len(rms) // n
#     word_boundaries = [i * step for i in range(n)] + [len(rms)]

#     # 3️⃣ Tính năng lượng trung bình mỗi từ
#     energy = [np.mean(rms[word_boundaries[i]:word_boundaries[i+1]]) for i in range(n)]
#     predicted_stressed_word = int(np.argmax(energy))

#     # 4️⃣ Lấy trọng âm chuẩn từ IPA cho từng từ
#     ipa_words = [ipa.convert(w) for w in words]
#     stress_positions = []
#     for ipa_word in ipa_words:
#         if "ˈ" in ipa_word:
#             stress_positions.append(True)
#         else:
#             stress_positions.append(False)

#     # 5️⃣ So sánh
#     stressed_words = [w for w, s in zip(words, stress_positions) if s]
#     predicted_word = words[predicted_stressed_word]
#     correct = predicted_word in stressed_words

#     # 6️⃣ In kết quả
#     print("\n📖 KẾT QUẢ PHÂN TÍCH TOÀN CÂU")
#     print("Câu:", sentence)
#     print("IPA từng từ:")
#     for w, i in zip(words, ipa_words):
#         print(f"  {w:<12} → {i}")
#     print(f"\n🔹 Từ bạn nhấn mạnh nhất (dựa RMS): {predicted_word}")
#     print(f"🔸 Các từ nên nhấn (IPA): {', '.join(stressed_words)}")
#     print("✅ Đúng trọng âm câu!" if correct else "⚠️ Sai từ được nhấn!")

#     return {
#         "sentence": sentence,
#         "ipa_words": ipa_words,
#         "predicted_stressed_word": predicted_word,
#         "true_stressed_words": stressed_words,
#         "stress_correct": correct
#     }
