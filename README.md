# 🌿 Crop Disease Detection System

An AI-powered web application that detects crop diseases from leaf images using a Convolutional Neural Network (CNN).  
The system provides disease predictions, confidence scores, severity levels, and recommended remedies.  
It also allows users to download reports in **PDF** and **JSON** formats.

---

## 🚀 Live Demo


https://cnn-based-crop-disease-predictor-vay.streamlit.app/

---

## 📌 Features
- Upload leaf image for disease detection
- Predicts disease using a trained CNN model
- Displays:
  - Detected disease
  - Confidence score
  - Severity level
- Provides:
  - Detailed disease description
  - Recommended remedies
- Shows top-3 prediction confidence chart
- Downloadable reports:
  - PDF report (with image and chart)
  - JSON report (structured data)

---

## 🧠 Model Details
- Model type: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Dataset: PlantVillage
- Number of classes: 15
- Image size: 128 × 128
- Output: Disease class probabilities

---

## 🗂️ Supported Classes
1. Pepper Bacterial Spot  
2. Pepper Healthy  
3. Potato Early Blight  
4. Potato Late Blight  
5. Potato Healthy  
6. Tomato Bacterial Spot  
7. Tomato Early Blight  
8. Tomato Late Blight  
9. Tomato Leaf Mold  
10. Tomato Septoria Leaf Spot  
11. Tomato Spider Mites  
12. Tomato Target Spot  
13. Tomato Yellow Leaf Curl Virus  
14. Tomato Mosaic Virus  
15. Tomato Healthy  

---

## ⚙️ Tech Stack
- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Pillow
- ReportLab (PDF generation)
- gdown (model download from Google Drive)

---


## 🧪 How to Run Locally

### 1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git  
cd your-repo-name

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the Streamlit app
streamlit run app.py

---

## ☁️ Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Go to: https://share.streamlit.io
3. Click **New App**
4. Select repository
5. Choose `app.py`
6. Click **Deploy**

---

## 📊 Severity Levels
Severity is calculated based on prediction confidence:

| Confidence | Severity |
|-----------|---------|
| < 60% | Low |
| 60–85% | Medium |
| > 85% | High |
| Healthy class | Healthy |

---

## 📄 Report Generation

### PDF Report Includes:
- Uploaded leaf image
- Disease name
- Confidence score
- Severity level
- Description
- Remedy
- Prediction confidence chart

### JSON Report Example
{
  "disease": "Tomato Target Spot",
  "confidence": 91.4,
  "severity": "High",
  "description": "...",
  "remedy": "..."
}

---

## 🌱 Real-World Applications
- Early crop disease detection
- Smart agriculture solutions
- Farmer assistance tools
- Agricultural advisory systems

---

## 🔮 Future Improvements
- Mobile app integration
- Multilingual support (Tamil, Hindi, etc.)
- Real-time camera detection
- GPS-based disease mapping
- Integration with government advisory systems

---



## 📜 License
This project is for educational and research purposes.

