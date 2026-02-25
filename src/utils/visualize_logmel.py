import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np


def show_logmel(file_path):

    y, sr = librosa.load(file_path, sr=16000)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        log_mel,
        sr=sr,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title("Log-Mel Spectrogram")
    plt.tight_layout()
    plt.show()