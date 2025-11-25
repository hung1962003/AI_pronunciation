# import librosa

# path = "output.wav"
# print("🎧 Đang load file:", path)

# try:
#     y, sr = librosa.load(path, sr=16000)
#     print(f"✅ Đã load thành công! Thời lượng: {len(y)/sr:.2f} giây, Sample rate: {sr}")
#     print(f"Độ lớn tín hiệu trung bình: {abs(y).mean():.4f}")
# except Exception as e:
#     print("❌ Lỗi khi load:", e)
