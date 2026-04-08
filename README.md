# Tomato Leaf Disease Detector 🍅

A deep learning-based system designed to detect and classify 10 different tomato leaf conditions (9 diseases and 1 healthy state) using Computer Vision.

## 🌟 Key Features

- **High Accuracy**: Achieves ~99.98% AUC using Transfer Learning with a MobileNetV2 architecture.
- **Real-time Detection**: Optimized for live webcam feeds with high FPS.
- **Intelligent Pre-check**: Includes an HSV-based "leaf detector" that saves CPU resources by only running the AI model when a leaf is actually present in the frame.
- **Stable Inference**: Uses Test-Time Augmentation (TTA) to provide stable, non-flickering predictions by averaging multiple views of the same frame.
- **Multiple Modes**: Supports detection from single images, batch processing of directories, or live camera feeds.

## 🛠️ Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/RidheshMahajan/DataModel_CV.git
   cd DataModel_CV
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run

> [!IMPORTANT]
> Ensure you are in the `DataModel_CV` directory before running the script.

To start the detector, run:
```bash
python main_project/detect.py
```

### Dashboard Mode (Upload Image + Cause/Effects/Prevention)

To launch the web dashboard, run:
```bash
streamlit run main_project/dashboard.py
```

What the dashboard provides:
- Upload a tomato leaf image.
- AI prediction with confidence score.
- Likely cause of the disease.
- Possible effects on the crop.
- Practical prevention guidance.

### Options:
Once the script starts, you can choose from the following menu:
1. **Detect from Image**: Provide a full path to a `.jpg` or `.png` file.
2. **Detect from Live Camera**: Starts your webcam for real-time analysis. Press `q` to exit.
3. **Batch Detection**: Processes all images located in the `main_project/assets` directory.

## 🧠 Model Architecture

The system uses **Transfer Learning** with a **MobileNetV2** base model, pre-trained on ImageNet.
- **Input Size**: 224x224 pixels.
- **Training Strategy**: Two-phase training (Feature Extraction followed by Fine-Tuning).
- **Optimization**: Adam optimizer with Sparse Categorical Crossentropy loss.

### Supported Classes:
1. Tomato Bacterial Spot
2. Tomato Early Blight
3. Tomato Late Blight
4. Tomato Leaf Mold
5. Tomato Septoria Leaf Spot
6. Tomato Spider Mites (Two-spotted spider mite)
7. Tomato Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato Mosaic Virus
10. Tomato Healthy

## 📄 Project Structure
```text
DataModel_CV/
├── main_project/
│   ├── detect.py          # Main execution script
│   ├── assets/            # Directory for batch processing
│   └── images/            # Project assets
├── model_and_data/        # Contains the trained .h5 model file
├── venv/                  # Python virtual environment
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```
