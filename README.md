# Video Sequence Prediction using Deep Learning 🎬🤖

A comprehensive video prediction system that generates future frames from short input sequences using three different deep learning architectures: ConvLSTM, PredRNN, and Transformer-based models on the UCF101 dataset.

## 🎯 Project Overview

This project develops deep learning models to predict and generate video sequences by learning motion patterns from human activity videos. The system takes short input sequences and generates coherent future frames, creating smooth video continuations.

## ✨ Key Features

- **Multi-Model Architecture**: Three different approaches for video prediction
- **Video Generation**: Combines predicted frames into smooth video sequences
- **Interactive UI**: Web interface for uploading videos and viewing predictions
- **Real-time Inference**: Runtime performance comparison across models
- **Comprehensive Evaluation**: MSE and SSIM metrics for prediction accuracy

## 🛠️ Technology Stack

- **Python** - Core programming language
- **PyTorch/TensorFlow** - Deep learning frameworks
- **OpenCV** - Video processing and frame manipulation
- **Streamlit/Gradio** - User interface development
- **NumPy/pandas** - Data processing and analysis
- **Matplotlib** - Visualization and plotting

## 🤖 Model Architectures

### 1. Convolutional LSTM (ConvLSTM)
- **Purpose**: Captures spatial-temporal patterns in video sequences
- **Architecture**: Combines CNN spatial feature extraction with LSTM temporal modeling
- **Input**: 10 consecutive frames (64x64 pixels)
- **Output**: 5-10 predicted future frames

### 2. PredRNN
- **Purpose**: Advanced temporal modeling with improved memory mechanisms
- **Architecture**: Enhanced RNN structure for better long-term dependencies
- **Features**: Spatial-temporal memory flow for coherent predictions

### 3. Transformer-based Model
- **Purpose**: Leverages attention mechanisms for long-term dependencies
- **Architecture**: Self-attention for frame-to-frame relationships
- **Advantages**: Better handling of complex motion patterns

## 📊 Dataset & Preprocessing

### UCF101 Dataset
- **Selected Classes**: Walking, Jumping, Biking, Running, Dancing (5+ action classes)
- **Frame Resolution**: 64x64 pixels for computational efficiency
- **Color Space**: RGB/Grayscale processing options
- **Sequence Length**: 10 input frames → 5-10 predicted frames

### Data Preprocessing Pipeline
1. **Video Extraction**: Extract frames from video files
2. **Resizing**: Normalize to 64x64 pixel resolution
3. **Sequence Creation**: Generate input-output frame pairs
4. **Normalization**: Pixel value scaling for model training

## 🚀 Installation & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/video-prediction-dl.git
cd video-prediction-dl

# Install dependencies
pip install -r requirements.txt

# Download UCF101 dataset
# Place dataset in data/ directory

# Launch UI
streamlit run app.py
```

## 💻 Usage

### Training Models
```bash
# Train ConvLSTM model
python models/convlstm.py --epochs 100 --batch_size 16

# Train PredRNN model
python models/predrnn.py --epochs 100 --batch_size 16

# Train Transformer model
python models/transformer.py --epochs 100 --batch_size 16
```

### Video Generation
```bash
# Generate video sequences
python generate_video.py --model convlstm --input sample_video.mp4
python generate_video.py --model predrnn --input sample_video.mp4
python generate_video.py --model transformer --input sample_video.mp4
```

### Web Interface
- **Upload Video**: Select input video clip from UCF101 dataset
- **Model Selection**: Choose from ConvLSTM, PredRNN, or Transformer
- **Generate Predictions**: View predicted frames and generated video
- **Performance Metrics**: Compare inference time and accuracy metrics

## 📈 Evaluation Metrics

### Quantitative Metrics
- **Mean Squared Error (MSE)**: Pixel-wise prediction accuracy
- **Structural Similarity Index (SSIM)**: Perceptual similarity measurement
- **Peak Signal-to-Noise Ratio (PSNR)**: Image quality assessment

### Qualitative Assessment
- **Visual Coherence**: Smoothness of generated sequences
- **Motion Consistency**: Realistic continuation of actions
- **Temporal Stability**: Frame-to-frame consistency

## 🎨 User Interface Features

- **Video Upload**: Drag-and-drop interface for input videos
- **Model Comparison**: Side-by-side prediction results
- **Real-time Processing**: Live inference with progress indicators
- **Results Visualization**: Generated frames and complete video playback
- **Performance Dashboard**: Runtime metrics and model statistics


## 🏆 Key Achievements

- Successfully implemented three different video prediction architectures
- Created smooth video generation pipeline with frame stitching
- Developed interactive web interface for model demonstration
- Achieved high-quality predictions on UCF101 human activity videos
- Comprehensive model comparison and evaluation framework

## 📁 Project Structure

```
video-prediction-dl/
├── data/                    # UCF101 dataset
├# Model implementations
│   ├── convlstm.py
│   ├── predrnn.py
│   └── transformer.py
├── utils/                   # Helper functions
├── ui/                      # Streamlit interface
├── evaluation/              # Metrics and analysis
├── generated_videos/        # Output videos
└── reports/                 # Project documentation
```

## 🔮 Future Enhancements

- **Higher Resolution**: Support for 128x128 or 256x256 frame prediction
- **Multi-Scale Prediction**: Different temporal horizons (short/long-term)
- **Adversarial Training**: GAN-based approaches for improved realism
- **Real-time Application**: Optimization for live video processing

---

*A comprehensive deep learning project demonstrating advanced video prediction techniques with practical applications in computer vision and video synthesis.*