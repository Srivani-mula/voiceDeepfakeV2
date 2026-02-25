import librosa
from config import ASVSPOOF2019_LA_TRAIN_FLAC

audio_path = ASVSPOOF2019_LA_TRAIN_FLAC / "LA_T_1000137.flac"

print("Checking path:", audio_path)
print("File exists:", audio_path.exists())

y, sr = librosa.load(audio_path, sr=16000)
print(y.shape, sr)
