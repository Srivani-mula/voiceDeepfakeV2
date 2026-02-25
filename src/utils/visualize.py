import matplotlib.pyplot as plt


def show_spectrogram(mel_tensor, save_path=None):
    mel = mel_tensor[0].cpu().numpy()  # first channel

    plt.figure(figsize=(8, 4))
    plt.imshow(mel, aspect="auto", origin="lower")
    plt.colorbar()
    plt.title("Log-Mel Spectrogram")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()