import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import json

# PDF libraries
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crop Disease Detector", layout="centered")

MODEL_PATH = "crop_model_15classes.keras"
FILE_ID = "1UCwUCrrVmFL2NifYhbrJwW4NsVjGLZCV"

# ---------------- DOWNLOAD MODEL ----------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)

# ---------------- CLASS LABELS ----------------
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

# ---------------- SEVERITY FUNCTION ----------------
def get_severity(disease, confidence):
    if "Healthy" in disease:
        return "Healthy"
    elif confidence < 60:
        return "Low"
    elif confidence < 85:
        return "Medium"
    else:
        return "High"

# ---------------- DISEASE INFO ----------------
disease_info = {
    "Tomato Target Spot": {
        "desc": "Target spot causes circular brown lesions with concentric rings. It typically appears on leaves and stems. Severe cases can cause defoliation and fruit damage.",
        "remedy": "Remove infected leaves immediately. Apply recommended fungicides. Improve air circulation and avoid excessive moisture."
    },
    "Tomato Healthy": {
        "desc": "The tomato plant appears healthy with vibrant green leaves. There are no visible signs of disease or pest damage. Growth is uniform and stable.",
        "remedy": "Maintain regular watering and fertilization schedules. Ensure proper sunlight and spacing. Continue periodic inspection for early detection."
    }
}

default_remedy = {
    "desc": "Disease detected in plant. Symptoms may vary depending on environmental conditions. Early treatment is recommended to avoid spread.",
    "remedy": "Remove infected leaves, apply appropriate fungicide or pesticide, and maintain proper irrigation and spacing."
}

# ---------------- PDF FUNCTION ----------------
def generate_pdf(image, disease, confidence, severity, description, remedy, chart_fig):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = temp_file.name
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Crop Disease Diagnosis Report", styles['Title']))
    elements.append(Spacer(1, 10))

    img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(img_temp.name)
    elements.append(RLImage(img_temp.name, width=200, height=200))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"<b>Disease:</b> {disease}", styles['Normal']))
    elements.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']))
    elements.append(Paragraph(f"<b>Severity:</b> {severity}", styles['Normal']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>Description:</b>", styles['Heading2']))
    elements.append(Paragraph(description, styles['Normal']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>Recommended Remedy:</b>", styles['Heading2']))
    elements.append(Paragraph(remedy, styles['Normal']))
    elements.append(Spacer(1, 15))

    chart_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    chart_fig.savefig(chart_temp.name, bbox_inches='tight')
    elements.append(RLImage(chart_temp.name, width=400, height=250))

    doc.build(elements)
    return pdf_path

# ---------------- UI ----------------
st.markdown("<h1 style='text-align: center;'>Crop Disease Detection System</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_container_width=True)

    img_resized = img.resize((128,128))
    img_array = np.array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)
    disease = class_labels[predicted_class]

    severity = get_severity(disease, confidence)

    info = disease_info.get(disease, default_remedy)

    # Prediction result
    st.markdown("## Prediction Result")
    st.metric("Detected Disease", disease)
    st.metric("Confidence", f"{confidence:.2f}%")
    st.metric("Severity Level", severity)

    st.markdown("### Disease Description")
    st.info(info["desc"])

    st.markdown("### Recommended Remedy")
    st.success(info["remedy"])

    # Chart
    st.markdown("### Prediction Confidence")
    probs = prediction[0]
    top_indices = probs.argsort()[-3:][::-1]
    top_labels = [class_labels[i] for i in top_indices]
    top_values = [probs[i]*100 for i in top_indices]

    fig, ax = plt.subplots()
    ax.bar(top_labels, top_values)
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Top 3 Predictions")
    st.pyplot(fig)

    # Downloads
    st.markdown("## Download PDF")
    pdf_path = generate_pdf(img, disease, confidence, severity, info["desc"], info["remedy"], fig)

    report_data = {
        "disease": disease,
        "confidence": round(confidence,2),
        "severity": severity,
        "description": info["desc"],
        "remedy": info["remedy"]
    }
    json_data = json.dumps(report_data, indent=4)

    col1, col2 = st.columns(2)
    with col1:
        with open(pdf_path,"rb") as f:
            st.download_button("Download PDF", f, "report.pdf")
    with col2:
        st.download_button("Download JSON", json_data, "report.json")
