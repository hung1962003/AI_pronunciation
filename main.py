# import pickle
# from stress_detection import detect_stress   
from function import segment_word_into_graphemes, map_graphemes_to_phoneme_indices, split_ipa_into_phonemes
# path = r"D:\AI\ai-pronunciation-trainer\data_de_en_2.pickle"

# with open(path, "rb") as f:
#     data = pickle.load(f)

# print(type(data))
# print("Số phần tử:", len(data))
# print("Ví dụ phần đầu:", list(data.items())[:5])
# detect_stress("output.wav", "Can you imagine a world without books and stories")
# print("🚀 Bắt đầu test stress detection...")

# from stress_detection import detect_stress
# import os

# # 🔧 Kiểm tra file âm thanh
# audio_path = "output.wav"
# text = "Can you imagine a world without books and stories"

# if not os.path.exists(audio_path):
#     print(f"❌ Không tìm thấy file âm thanh: {audio_path}")
# else:
#     print(f"✅ Tìm thấy file âm thanh: {audio_path}")

# try:
#     print("🎧 Gọi hàm detect_stress() ...")
#     result = detect_stress(audio_path, text)
#     print("\n📊 KẾT QUẢ KIỂM TRA:")
#     for k, v in result.items():
#         print(f"{k}: {v}")
# except Exception as e:
#     print(f"⚠️ Lỗi khi chạy detect_stress: {e}")

# print("🏁 Kết thúc test.")


print(split_ipa_into_phonemes("tʃɜːtʃ"))
print(map_graphemes_to_phoneme_indices(segment_word_into_graphemes("church"), len(split_ipa_into_phonemes("tʃɜːtʃ"))))