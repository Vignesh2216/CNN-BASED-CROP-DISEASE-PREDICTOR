import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os
import matplotlib.pyplot as plt
import tempfile
import json
import sqlite3
import hashlib
from datetime import datetime
from textwrap import dedent

# PDF libraries
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- DATABASE ----------------
DB_PATH = "crop_users.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(full_name, email, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (full_name, email, password, created_at) VALUES (?, ?, ?, ?)",
            (full_name, email, hash_password(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
        return True, "Registration successful. Please sign in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    except Exception as e:
        return False, f"Registration failed: {e}"


def login_user(email, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, full_name, email FROM users WHERE email = ? AND password = ?",
            (email, hash_password(password))
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            return True, {
                "id": user[0],
                "full_name": user[1],
                "email": user[2]
            }
        return False, "Invalid email or password."
    except Exception as e:
        return False, f"Login failed: {e}"


init_db()

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "landing"

# ---------------- CUSTOM CSS ----------------
st.markdown(dedent("""
<style>
    .stApp {
        background: linear-gradient(135deg, #08140f 0%, #0f1f17 45%, #14281d 100%);
        color: #f8fafc;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1812 0%, #132019 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    .hero-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(34,197,94,0.12), rgba(59,130,246,0.10));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 28px;
        padding: 48px 42px 34px 42px;
        backdrop-filter: blur(14px);
        box-shadow: 0 14px 40px rgba(0,0,0,0.28);
        margin-bottom: 22px;
    }

    .hero-card::before {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(34,197,94,0.25) 0%, rgba(34,197,94,0.0) 70%);
        border-radius: 50%;
    }

    .hero-badge {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        color: #dcfce7;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 16px;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.08;
        color: #ffffff;
        margin-bottom: 0.65rem;
        letter-spacing: -0.6px;
        max-width: 850px;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: #d1fae5;
        max-width: 820px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .landing-actions {
        display: flex;
        gap: 18px;
        margin-top: 26px;
        flex-wrap: wrap;
    }

    .landing-action-card {
        flex: 1;
        min-width: 240px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }

    .landing-action-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }

    .landing-action-text {
        color: #d1d5db;
        font-size: 0.94rem;
        line-height: 1.6;
        margin-bottom: 14px;
    }

    .info-chip-wrap {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 22px;
    }

    .info-chip {
        background: rgba(255,255,255,0.07);
        color: #ecfdf5;
        padding: 9px 15px;
        border-radius: 999px;
        font-size: 0.92rem;
        border: 1px solid rgba(255,255,255,0.10);
    }

    .landing-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 20px;
    }

    .landing-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        min-height: 180px;
    }

    .landing-icon {
        font-size: 1.9rem;
        margin-bottom: 10px;
    }

    .landing-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .landing-text {
        color: #d1d5db;
        font-size: 0.96rem;
        line-height: 1.7;
    }

    .section-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 12px;
        color: #ffffff;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(16,185,129,0.14), rgba(59,130,246,0.12));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
        word-break: break-word;
    }

    .metric-label {
        color: #d1d5db;
        font-size: 0.95rem;
    }

    .upload-box {
        background: rgba(255,255,255,0.03);
        border: 1.5px dashed rgba(148,163,184,0.4);
        border-radius: 20px;
        padding: 18px;
    }

    .result-banner {
        background: linear-gradient(135deg, rgba(16,185,129,0.16), rgba(34,197,94,0.14));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 18px 20px;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .result-main {
        font-size: 1.7rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
        word-break: break-word;
    }

    .result-sub {
        color: #d1fae5;
        font-size: 1rem;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .footer {
        margin-top: 32px;
        padding: 22px 24px;
        border-radius: 20px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        color: #d1d5db;
        font-size: 0.95rem;
        line-height: 1.8;
    }

    .footer-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }

    .footer-sub {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-top: 8px;
    }

    .auth-wrapper {
        max-width: 500px;
        margin: 30px auto;
    }

    .auth-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.22);
    }

    .auth-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
        text-align: center;
    }

    .auth-subtitle {
        color: #d1d5db;
        text-align: center;
        margin-bottom: 20px;
    }

    .stButton > button, .stDownloadButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(135deg, #10b981, #16a34a);
        color: white;
        font-weight: 700;
        padding: 0.72rem 1rem;
        box-shadow: 0 6px 18px rgba(16,185,129,0.25);
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: rgba(255,255,255,0.20);
        background: linear-gradient(135deg, #059669, #15803d);
        color: white;
    }

    div[data-testid="stFileUploader"] {
        background: transparent !important;
    }

    div[data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }

    @media (max-width: 900px) {
        .landing-grid {
            grid-template-columns: 1fr;
        }

        .hero-title {
            font-size: 2.2rem;
        }

        .landing-actions {
            flex-direction: column;
        }
    }
</style>
"""), unsafe_allow_html=True)

# ---------------- MODEL ----------------
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


# ---------------- LANDING PAGE ----------------
def show_landing_page():
    st.markdown(dedent("""
    <div class="hero-card">
        <div class="hero-badge">AI-Powered Crop Health Analysis</div>
        <div class="hero-title">Crop Disease Detection System</div>
        <p class="hero-subtitle">
            A modern AI dashboard for identifying crop leaf diseases from images, analyzing diagnosis confidence,
            assessing severity, and generating export-ready reports for monitoring and documentation.
        </p>

        <div class="landing-actions">
            <div class="landing-action-card">
                <div class="landing-action-title">Create New Account</div>
                <div class="landing-action-text">
                    Register to access the crop disease diagnosis dashboard and downloadable reports.
                </div>
            </div>
            <div class="landing-action-card">
                <div class="landing-action-title">Already a User?</div>
                <div class="landing-action-text">
                    Sign in and continue directly to the main diagnosis page.
                </div>
            </div>
        </div>

        <div class="info-chip-wrap">
            <div class="info-chip">🌿 Disease Detection</div>
            <div class="info-chip">📊 Confidence Analysis</div>
            <div class="info-chip">⚠️ Severity Estimation</div>
            <div class="info-chip">📄 PDF Report</div>
            <div class="info-chip">🔐 Secure Access</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    btn1, btn2, _ = st.columns([1, 1, 2])

    with btn1:
        if st.button("📝 Register", use_container_width=True):
            st.session_state.auth_mode = "register"
            st.rerun()

    with btn2:
        if st.button("🔐 Sign In", use_container_width=True):
            st.session_state.auth_mode = "login"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(dedent("""
    <div class="landing-grid">
        <div class="landing-card">
            <div class="landing-icon">🌱</div>
            <div class="landing-title">Smart Disease Identification</div>
            <div class="landing-text">
                Upload a crop leaf image and let the trained deep learning model identify the most likely disease
                or confirm whether the plant is healthy.
            </div>
        </div>
        <div class="landing-card">
            <div class="landing-icon">📈</div>
            <div class="landing-title">Clear Diagnostic Insights</div>
            <div class="landing-text">
                View confidence values, severity level, disease description, and recommended remedy
                in a polished, presentation-ready interface.
            </div>
        </div>
        <div class="landing-card">
            <div class="landing-icon">📦</div>
            <div class="landing-title">Report & Export Ready</div>
            <div class="landing-text">
                Download the diagnosis as PDF and JSON for documentation, field reporting,
                academic demonstration, or portfolio presentation.
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown(dedent("""
    <div class="footer">
        <div class="footer-title">Crop Disease Detection System</div>
        <div>
            A professional deep learning project for crop disease diagnosis using image-based analysis,
            classification modeling, severity assessment, and exportable reporting.
        </div>
        <div class="footer-sub">
            Built with Streamlit, TensorFlow, Matplotlib, SQLite, and ReportLab
        </div>
    </div>
    """), unsafe_allow_html=True)


# ---------------- REGISTER PAGE ----------------
def show_register_page():
    st.markdown('<div class="auth-wrapper"><div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Create Account</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Register to access the Crop Disease Detection dashboard</div>', unsafe_allow_html=True)

    with st.form("register_form"):
        full_name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

        if submitted:
            if not full_name.strip():
                st.error("Full name is required.")
            elif not email.strip():
                st.error("Email is required.")
            elif not password:
                st.error("Password is required.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = register_user(full_name.strip(), email.strip(), password)
                if success:
                    st.success(message)
                    st.session_state.auth_mode = "login"
                    st.rerun()
                else:
                    st.error(message)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", key="back_from_register"):
            st.session_state.auth_mode = "landing"
            st.rerun()
    with col2:
        if st.button("Already have an account? Sign In", key="to_login"):
            st.session_state.auth_mode = "login"
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)


# ---------------- LOGIN PAGE ----------------
def show_login_page():
    st.markdown('<div class="auth-wrapper"><div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Sign In</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Login to continue to the Crop Disease Detection dashboard</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")

        if submitted:
            if not email.strip():
                st.error("Email is required.")
            elif not password:
                st.error("Password is required.")
            else:
                success, result = login_user(email.strip(), password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user = result
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error(result)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", key="back_from_login"):
            st.session_state.auth_mode = "landing"
            st.rerun()
    with col2:
        if st.button("Create New Account", key="to_register"):
            st.session_state.auth_mode = "register"
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)


# ---------------- MAIN APP ----------------
def show_main_app():
    st.sidebar.markdown("## 🌿 Dashboard Controls")
    st.sidebar.markdown("Upload a crop leaf image and review the AI-based diagnosis.")

    st.sidebar.markdown("---")
    st.sidebar.success(f"Logged in as {st.session_state.user['full_name']}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.auth_mode = "landing"
        st.rerun()

    st.markdown(dedent(f"""
    <div class="hero-card">
        <div class="hero-badge">Welcome, {st.session_state.user['full_name']}</div>
        <div class="hero-title">Crop Disease Detection Dashboard</div>
        <p class="hero-subtitle">
            Upload a leaf image to identify crop disease, inspect confidence and severity,
            and download professional diagnosis reports.
        </p>
        <div class="info-chip-wrap">
            <div class="info-chip">🌿 Leaf Image Analysis</div>
            <div class="info-chip">📊 Diagnosis Confidence</div>
            <div class="info-chip">⚠️ Severity Level</div>
            <div class="info-chip">📄 PDF + JSON Export</div>
            <div class="info-chip">👤 {st.session_state.user['email']}</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Leaf Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    st.markdown('<div class="small-note">Supported formats: JPG, JPEG, PNG</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

        col_a, col_b = st.columns([1.15, 1])

        with col_a:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Uploaded Image</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown(f"<div class='small-note'>File name: {uploaded_file.name}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing leaf image and generating diagnosis..."):
            prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)
        disease = class_labels[predicted_class]
        severity = get_severity(disease, confidence)
        info = disease_info.get(disease, default_remedy)

        st.markdown("""
        <div class="section-card">
            <div class="section-title">Diagnosis Summary</div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{disease}</div>
                <div class="metric-label">Detected Disease</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{confidence:.2f}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{severity}</div>
                <div class="metric-label">Severity Level</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="result-banner">
                <div class="result-main">{disease}</div>
                <div class="result-sub">Diagnosis confidence: {confidence:.2f}% | Severity: {severity}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Disease Description</div>', unsafe_allow_html=True)
        st.info(info["desc"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recommended Remedy</div>', unsafe_allow_html=True)
        st.success(info["remedy"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Confidence</div>', unsafe_allow_html=True)

        probs = prediction[0]
        top_indices = probs.argsort()[-3:][::-1]
        top_labels = [class_labels[i] for i in top_indices]
        top_values = [probs[i] * 100 for i in top_indices]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(top_labels, top_values)
        ax.set_ylabel("Confidence (%)")
        ax.set_title("Top 3 Predictions")
        plt.xticks(rotation=10)
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

        pdf_path = generate_pdf(img, disease, confidence, severity, info["desc"], info["remedy"], fig)

        report_data = {
            "disease": disease,
            "confidence": round(confidence, 2),
            "severity": severity,
            "description": info["desc"],
            "remedy": info["remedy"]
        }
        json_data = json.dumps(report_data, indent=4)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Download Results</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇ Download PDF",
                    f,
                    "report.pdf",
                    mime="application/pdf"
                )
        with col2:
            st.download_button(
                "⬇ Download JSON",
                json_data,
                "report.json",
                mime="application/json"
            )

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown(dedent("""
        <div class="section-card">
            <div class="section-title">Getting Started</div>
            <p style="color:#d1d5db; margin-bottom:0; line-height:1.8;">
                Upload a crop leaf image to begin diagnosis. The system will analyze the image,
                predict the most likely disease class, estimate confidence and severity,
                and provide treatment guidance with downloadable outputs.
            </p>
        </div>
        """), unsafe_allow_html=True)

    st.markdown(dedent("""
    <div class="footer">
        <div class="footer-title">Crop Disease Detection System</div>
        <div>
            A professional deep learning project for crop disease diagnosis using image-based analysis,
            classification modeling, severity assessment, and exportable reporting.
        </div>
        <div class="footer-sub">
            Built with Streamlit, TensorFlow, Matplotlib, SQLite, and ReportLab
        </div>
    </div>
    """), unsafe_allow_html=True)


# ---------------- APP ROUTING ----------------
if st.session_state.logged_in:
    show_main_app()
else:
    if st.session_state.auth_mode == "register":
        show_register_page()
    elif st.session_state.auth_mode == "login":
        show_login_page()
    else:
        show_landing_page()
