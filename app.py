import streamlit as st
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import os
import gdown
import torch
from src.models.resnet_model import ResNet18
from src.utils.gradcam import GradCAM


# =========================================
# SETTINGS
# =========================================
TARGET_LENGTH = 200
MODEL_PATH = "best_model.pth"
FILE_ID = "1f7YcM4RRekWJkBLiw2Uj6ZOp-HahKaFf"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================
# PAD OR TRUNCATE
# =========================================
def pad_or_truncate(spec, target_length=200):
    if spec.shape[1] > target_length:
        spec = spec[:, :target_length]
    elif spec.shape[1] < target_length:
        pad_width = target_length - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")
    return spec


# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():
    model = ResNet18(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# =========================================
# PREPROCESS AUDIO
# =========================================
def preprocess(file):

    y, sr = librosa.load(file, sr=16000)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = pad_or_truncate(log_mel, TARGET_LENGTH)

    norm_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)

    tensor = np.stack([norm_mel, norm_mel, norm_mel], axis=0)
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

    return tensor.to(DEVICE), log_mel


# =========================================
# STREAMLIT UI
# =========================================
st.title("🎙 Voice Deepfake Detection System")
st.write("Trained on ASVspoof 2019 LA Dataset")

model = load_model()
gradcam = GradCAM(model)

uploaded_file = st.file_uploader(
    "Upload an audio file (.wav or .flac)",
    type=["wav", "flac"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    input_tensor, log_mel = preprocess(uploaded_file)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence = probs[0][1].item()

    prediction = "Bonafide" if confidence > 0.5 else "Spoof"

    st.subheader("🔍 Prediction")
    st.write(f"**Result:** {prediction}")
    st.write(f"**Confidence:** {confidence:.4f}")

    # =========================
    # Display Log-Mel
    # =========================
    fig, ax = plt.subplots(figsize=(8, 3))

    img = librosa.display.specshow(
        log_mel,
        x_axis='time',
        y_axis='mel',
        ax=ax
    )

    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    st.pyplot(fig)
    plt.close(fig)

    # =========================
    # Grad-CAM
    # =========================
    st.subheader("🔥 Grad-CAM Visualization")

    heatmap = gradcam.generate(input_tensor, class_idx=1)

    heatmap = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    st.image(heatmap, caption="Grad-CAM Heatmap")
