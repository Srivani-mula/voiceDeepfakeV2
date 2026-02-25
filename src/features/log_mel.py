import torch
import torchaudio


class LogMelExtractor:
    def __init__(self, sample_rate=16000, n_mels=80):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=160,
            n_mels=n_mels
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform):
        mel = self.mel(waveform)
        mel = self.db(mel)
        return mel