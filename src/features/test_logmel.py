from src.data.asvspoof2019_dataset import ASVspoof2019Dataset
from src.features.log_mel import LogMelExtractor

dataset = ASVspoof2019Dataset(split="train")
extractor = LogMelExtractor()

waveform, label = dataset[0]
logmel = extractor(waveform)

print("Waveform:", waveform.shape)
print("Log-Mel:", logmel.shape)
print("Label:", label)
