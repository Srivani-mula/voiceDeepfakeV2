import librosa
import numpy as np

def extract_log_mel(
    audio_path,
    sr=16000,
    n_mels=128,
    n_fft=1024,
    hop_length=256
):
    y, _ = librosa.load(audio_path, sr=sr)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # 🔥 SAME NORMALIZATION AS TRAINING
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

    return log_mel
