from src.data.asvspoof2019_dataset import ASVspoof2019Dataset

dataset = ASVspoof2019Dataset(split="train")

print("Total samples:", len(dataset))
print("First sample:", dataset[0])
