import streamlit as st
import os
import cv2
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from io import BytesIO

# Constants
FRAME_SIZE = (64, 64)
INPUT_FRAMES = 10
PRED_FRAMES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definitions (PredRNN, ConvLSTM, Transformer)
class PredRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PredRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class TransformerFramePredictor(torch.nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, max_seq_length):
        super(TransformerFramePredictor, self).__init__()
        self.embedding = torch.nn.Linear(input_dim, hidden_dim)
        self.transformer = torch.nn.Transformer(
            d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers
        )
        self.fc = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output


def preprocess_frames(folder_path, num_frames=INPUT_FRAMES):
    frame_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jpg")])
    frames = []
    for path in frame_paths[:num_frames]:
        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        frame_resized = cv2.resize(frame, FRAME_SIZE)
        frames.append(frame / 255.0)
    return np.array(frames)


def generate_video(frames, output_path):
    height, width = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), 5, (width, height), isColor=False)
    for frame in frames:
        out.write(cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    out.release()
    return output_path


def calculate_metrics(true_frames, pred_frames):
    ssim_scores, mse_scores = [], []
    for true, pred in zip(true_frames, pred_frames):
        ssim_scores.append(ssim(true, pred, data_range=1.0))
        mse_scores.append(mean_squared_error(true.flatten(), pred.flatten()))
    return np.mean(ssim_scores), np.mean(mse_scores)


def display_results(predicted_frames, true_frames, metrics, video_path):
    st.markdown("### Prediction Results")
    col1, col2 = st.columns(2)
    col1.markdown(f"**SSIM:** {metrics[0]:.4f}")
    col2.markdown(f"**MSE:** {metrics[1]:.4f}")

    st.markdown("#### Predicted Frames")
    for idx, frame in enumerate(predicted_frames):
        st.image(frame, caption=f"Predicted Frame {idx+1}", use_column_width=True)

    st.markdown("#### Ground Truth Frames")
    for idx, frame in enumerate(true_frames):
        st.image(frame, caption=f"Ground Truth Frame {idx+1}", use_column_width=True)

    st.markdown("#### Prediction Video")
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
    st.video(video_bytes)


def main():
    st.title("Video Frame Prediction Application")
    st.sidebar.markdown("## Select Options")
    frame_test_dir = st.sidebar.text_input("Frame Test Directory", r"C:\Users\Ali Arfa\Downloads\deep_learning_project\frames_test")
    models_dir = st.sidebar.text_input("Models Directory", r"C:\Users\Ali Arfa\Downloads\deep_learning_project")

    # Select video directory
    if os.path.exists(frame_test_dir):
        available_videos = [f for f in os.listdir(frame_test_dir) if os.path.isdir(os.path.join(frame_test_dir, f))]
        selected_video = st.sidebar.selectbox("Select a Video Folder", available_videos)
        video_path = os.path.join(frame_test_dir, selected_video)
    else:
        st.error("Frame Test Directory not found!")
        return

    # Select model
    model_choice = st.sidebar.selectbox("Select a Model", ["PredRNN", "ConvLSTM", "Transformer"])

    # Load model
    if model_choice == "PredRNN":
        model = PredRNN(input_dim=64 * 64, hidden_dim=512, output_dim=64 * 64, num_layers=2).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(models_dir, "predrnn_model.pth"), map_location=DEVICE))
    elif model_choice == "Transformer":
        model = TransformerFramePredictor(
            input_dim=64 * 64, num_heads=8, num_layers=4, hidden_dim=512, max_seq_length=INPUT_FRAMES
        ).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(models_dir, "transformer_model.pth"), map_location=DEVICE))
    else:
        model = tf.keras.models.load_model(os.path.join(models_dir, "conv_lstm_model.h5"))
    model.eval()

    if st.sidebar.button("Predict"):
        with st.spinner("Processing..."):
            # Load frames and ground truth
            input_frames = preprocess_frames(video_path, INPUT_FRAMES)
            ground_truth_frames = preprocess_frames(video_path, PRED_FRAMES)

            if model_choice in ["PredRNN", "Transformer"]:
                input_tensor = torch.FloatTensor(input_frames).view(1, INPUT_FRAMES, -1).to(DEVICE)
                predicted_frames = []
                for _ in range(PRED_FRAMES):
                    with torch.no_grad():
                        pred = model(input_tensor).cpu().numpy().reshape(-1, *FRAME_SIZE)
                    predicted_frames.append(pred[0])
                    input_tensor = torch.cat((input_tensor[:, 1:, :], pred), dim=1)
            else:
                input_frames_expanded = input_frames.reshape(1, INPUT_FRAMES, *FRAME_SIZE, 1)
                predicted_frames = model.predict(input_frames_expanded)

            # Calculate metrics
            ssim_score, mse_score = calculate_metrics(ground_truth_frames, predicted_frames)

            # Save and display results
            video_output_path = generate_video(predicted_frames, "predicted_video.avi")
            display_results(predicted_frames, ground_truth_frames, (ssim_score, mse_score), video_output_path)


if __name__ == "__main__":
    main()
