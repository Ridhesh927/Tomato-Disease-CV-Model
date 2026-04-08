import os
from typing import Dict

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model_and_data")
PRIMARY_MODEL = os.path.join(MODEL_DIR, "tomato_disease_model_efficientnetb3.h5")
FALLBACK_MODEL = os.path.join(MODEL_DIR, "tomato_disease_model.h5")
IMG_SIZE = (224, 224)

TOMATO_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

DISEASE_GUIDE: Dict[str, Dict[str, str]] = {
    "Tomato_Bacterial_spot": {
        "name": "Bacterial Spot",
        "cause": "Caused by Xanthomonas bacteria, often spread by splashing water, tools, and infected seedlings.",
        "effects": "Creates dark leaf spots, reduces photosynthesis, causes defoliation, and can reduce fruit yield and quality.",
        "prevention": "Use disease-free seeds, avoid overhead irrigation, rotate crops, sanitize tools, and remove infected plant debris.",
    },
    "Tomato_Early_blight": {
        "name": "Early Blight",
        "cause": "Usually caused by Alternaria fungi, favored by warm, humid conditions and stressed plants.",
        "effects": "Shows concentric ring spots on older leaves, causes yellowing and leaf drop, weakening plant growth.",
        "prevention": "Mulch to reduce soil splash, prune lower leaves, improve airflow, rotate crops, and apply preventive fungicide if needed.",
    },
    "Tomato_Late_blight": {
        "name": "Late Blight",
        "cause": "Caused by Phytophthora infestans in cool, wet conditions with high humidity.",
        "effects": "Rapid leaf and stem lesions, fruit rot, and potentially total crop loss if unmanaged.",
        "prevention": "Plant resistant varieties, avoid leaf wetness, ensure spacing for airflow, remove infected plants quickly, and monitor weather alerts.",
    },
    "Tomato_Leaf_Mold": {
        "name": "Leaf Mold",
        "cause": "Caused by Passalora fulva, common in humid greenhouses with poor ventilation.",
        "effects": "Yellow patches on upper leaf surface and olive mold below, reducing leaf function and yield.",
        "prevention": "Lower humidity, improve greenhouse ventilation, avoid overcrowding, and remove infected foliage early.",
    },
    "Tomato_Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "cause": "Caused by Septoria lycopersici fungus, often spread by rain splash and contaminated tools.",
        "effects": "Many small circular spots that merge over time, causing severe defoliation and sunscald risk on fruits.",
        "prevention": "Rotate crops, mulch soil, water at base of plants, remove infected leaves, and use clean supports and tools.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Two-Spotted Spider Mite Damage",
        "cause": "Infestation by spider mites, usually worse in hot, dry conditions.",
        "effects": "Leaf stippling, bronzing, webbing, and reduced vigor that can lower fruit production.",
        "prevention": "Inspect undersides of leaves, use water sprays or biological control, avoid dusty stress conditions, and use miticides when necessary.",
    },
    "Tomato__Target_Spot": {
        "name": "Target Spot",
        "cause": "Fungal disease (Corynespora cassiicola) encouraged by warm and wet weather.",
        "effects": "Circular target-like lesions on leaves and fruits, leading to defoliation and lower marketable yield.",
        "prevention": "Improve field sanitation, rotate crops, avoid prolonged leaf wetness, and apply fungicide programs in high-risk periods.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "cause": "Virus transmitted mainly by whiteflies and infected plant material.",
        "effects": "Leaf curling, yellowing, stunted growth, poor flowering, and major fruit loss.",
        "prevention": "Control whiteflies, use resistant varieties, remove infected plants, use reflective mulches, and keep fields weed-free.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "cause": "Mechanical transmission from infected seeds, tools, hands, and plant contact.",
        "effects": "Mosaic mottling, leaf distortion, uneven fruit ripening, and overall reduced productivity.",
        "prevention": "Use certified seeds, disinfect hands and tools, avoid handling plants when wet, and remove infected plants promptly.",
    },
    "Tomato_healthy": {
        "name": "Healthy Leaf",
        "cause": "No visible disease symptoms detected.",
        "effects": "Plant is likely in good physiological condition for growth and fruit development.",
        "prevention": "Continue balanced nutrition, proper watering, good airflow, and regular scouting to maintain plant health.",
    },
}


@st.cache_resource
def load_model() -> tf.keras.Model:
    if os.path.exists(PRIMARY_MODEL):
        return tf.keras.models.load_model(PRIMARY_MODEL)
    if os.path.exists(FALLBACK_MODEL):
        return tf.keras.models.load_model(FALLBACK_MODEL)
    raise FileNotFoundError("No model file found in model_and_data directory.")


def is_leaf_present(frame_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    total_pixels = frame_bgr.shape[0] * frame_bgr.shape[1]
    green_ratio = cv2.countNonZero(mask) / total_pixels
    return green_ratio > 0.02


def extract_leaf_roi(frame_bgr: np.ndarray) -> np.ndarray:
    """Crop to the dominant leaf-like region to reduce background bias."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame_bgr

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    if w * h < 0.05 * frame_bgr.shape[0] * frame_bgr.shape[1]:
        return frame_bgr

    pad = int(0.08 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_bgr.shape[1], x + w + pad)
    y2 = min(frame_bgr.shape[0], y + h + pad)
    return frame_bgr[y1:y2, x1:x2]


def model_has_rescaling_layer(model: tf.keras.Model) -> bool:
    return any(layer.__class__.__name__ == "Rescaling" for layer in model.layers)


def prepare_batch_for_model(model: tf.keras.Model, batch_rgb: np.ndarray) -> np.ndarray:
    if model_has_rescaling_layer(model):
        # Model expects pixel range 0-255 and handles normalization internally.
        return batch_rgb.astype(np.float32)
    # Fallback for models trained with MobileNet-style external preprocessing.
    return tf.keras.applications.mobilenet_v2.preprocess_input(batch_rgb.astype(np.float32))


def predict_leaf(model: tf.keras.Model, frame_bgr: np.ndarray):
    leaf_bgr = extract_leaf_roi(frame_bgr)
    img = cv2.resize(leaf_bgr, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    base = tf.keras.preprocessing.image.img_to_array(img_rgb)
    flip = tf.keras.preprocessing.image.img_to_array(cv2.flip(img_rgb, 1))
    bright = tf.keras.preprocessing.image.img_to_array(cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=10))

    batch = np.stack([base, flip, bright])
    batch = prepare_batch_for_model(model, batch)
    predictions = model.predict(batch, verbose=0)
    avg_pred = np.mean(predictions, axis=0)

    class_index = int(np.argmax(avg_pred))
    confidence = float(np.max(avg_pred) * 100)
    class_name = TOMATO_CLASSES[class_index]
    top3_idx = np.argsort(avg_pred)[-3:][::-1]
    top3 = [(TOMATO_CLASSES[i], float(avg_pred[i] * 100)) for i in top3_idx]
    return class_name, confidence, top3


def clean_class_name(class_name: str) -> str:
    return class_name.replace("Tomato_", "").replace("__", "_").replace("_", " ")


def main() -> None:
    st.set_page_config(page_title="Tomato Disease Dashboard", page_icon="🍅", layout="wide")

    st.title("Tomato Leaf Disease Dashboard")
    st.write("Upload a tomato leaf image to get prediction, likely cause, effects, and prevention advice.")

    try:
        model = load_model()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()

    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is None:
        st.info("Add an image to start diagnosis.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if not is_leaf_present(frame_bgr):
        st.warning("No clear tomato leaf detected in this image. Please upload a closer leaf image.")
        return

    class_name, confidence, top3 = predict_leaf(model, frame_bgr)
    disease_info = DISEASE_GUIDE.get(class_name, {
        "name": clean_class_name(class_name),
        "cause": "Cause information is not available.",
        "effects": "Effects information is not available.",
        "prevention": "Prevention information is not available.",
    })

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prediction")
        st.success(f"Detected: {disease_info['name']}")
        st.metric("Confidence", f"{confidence:.2f}%")
        st.caption(f"Model class: {class_name}")
        if confidence < 60:
            st.warning("Low confidence prediction. Use a clearer close-up image of a single leaf.")

    with col2:
        st.subheader("Field Guidance")
        st.write(f"**Likely Cause:** {disease_info['cause']}")
        st.write(f"**Effects on Crop:** {disease_info['effects']}")
        st.write(f"**Prevention Tips:** {disease_info['prevention']}")

    with st.expander("Model Debug (Top-3 Probabilities)"):
        model_name = os.path.basename(PRIMARY_MODEL if os.path.exists(PRIMARY_MODEL) else FALLBACK_MODEL)
        preprocessing_mode = "Internal Rescaling" if model_has_rescaling_layer(model) else "MobileNetV2 preprocess_input"
        st.write(f"Loaded model: {model_name}")
        st.write(f"Preprocessing mode: {preprocessing_mode}")
        for rank, (cls_name, prob) in enumerate(top3, start=1):
            st.write(f"{rank}. {clean_class_name(cls_name)} ({prob:.2f}%)")


if __name__ == "__main__":
    main()
