import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Crop Disease Detector", layout="centered")

MODEL_PATH = "crop_model_15classes.keras"
FILE_ID = "PASTE_YOUR_FILE_ID_HERE"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
model = load_model(MODEL_PATH)

# Class labels
class_labels = {
    0: "Pepper Bacterial Spot",
    1: "Pepper Healthy",
    2: "Potato Early Blight",
    3: "Potato Late Blight",
    4: "Potato Healthy",
    5: "Tomato Bacterial Spot",
    6: "Tomato Early Blight",
    7: "Tomato Late Blight",
    8: "Tomato Leaf Mold",
    9: "Tomato Septoria Leaf Spot",
    10: "Tomato Spider Mites",
    11: "Tomato Target Spot",
    12: "Tomato Yellow Leaf Curl Virus",
    13: "Tomato Mosaic Virus",
    14: "Tomato Healthy"
}

# Remedy and description
disease_info = {
    "Tomato Target Spot": {
        "desc": "Fungal disease causing brown spots with concentric rings.",
        "remedy": "Use fungicides and remove infected leaves. Improve air circulation."
    },
    "Tomato Early Blight": {
        "desc": "Causes dark spots with concentric rings on older leaves.",
        "remedy": "Apply fungicide and remove affected leaves."
    },
    "Tomato Late Blight": {
        "desc": "Serious fungal disease causing dark lesions on leaves and fruit.",
        "remedy": "Use copper-based fungicides and avoid overhead watering."
    },
    "Tomato Healthy": {
        "desc": "The plant appears healthy with no visible disease.",
        "remedy": "Maintain proper watering, sunlight, and nutrition."
    }
}

# Default remedy for other classes
default_remedy = {
    "desc": "Disease detected in plant.",
    "remedy": "Remove infected parts and apply appropriate fungicide or pesticide."
}

# UI Header
st.markdown(
    """
    <h1 style='text-align: center; color: #2E8B57;'>
        🌿 Crop Disease Detection System
    </h1>
    <p style='text-align: center;'>
        Upload a leaf image to detect disease and get treatment advice.
    </p>
    """,
    unsafe_allow_html=True
)

# File upload
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Leaf", use_container_width=True)

    # Preprocess
    img_resized = img.resize((128,128))
    img_array = np.array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    disease = class_labels[predicted_class]

    # Get remedy info
    info = disease_info.get(disease, default_remedy)

    # Display results
    st.markdown("## 🔍 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Detected Disease", value=disease)

    with col2:
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

    st.markdown("### 🩺 Disease Description")
    st.info(info["desc"])

    st.markdown("### 🌱 Recommended Remedy")
    st.success(info["remedy"])
