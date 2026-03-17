import os
import sys

# 1. CRITICAL: Force Legacy Keras and fix module paths before other imports
os.environ["TF_USE_LEGACY_KERAS"] = "1"
try:
    import tensorflow.keras as keras
    sys.modules['keras'] = keras
except ImportError:
    pass

import streamlit as st
import numpy as np
import pickle
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="MedVision: Skin Cancer Detection", layout="centered")

# --- Model Loading ---
@st.cache_resource
def load_skin_model():
    model_path = 'trained_model.pkl'
    
    # Check if file is just a Git LFS pointer (common cause of 'v' error)
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        st.error("Error: Model file is missing or a Git LFS pointer. Please push actual LFS data.")
        st.stop()
        
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Unpickling Error: {e}")
        st.stop()

model = load_skin_model()

# --- Prediction Logic ---
def predict(uploaded_file, model):
    # Preprocessing: Match your training (224x224, normalized)
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    prediction = model.predict(img_array)
    
    # Logic: > 0.5 is Malignant, <= 0.5 is Benign
    prob = prediction[0][0] if hasattr(prediction[0], "__len__") else prediction[0]
    label = "Malignant" if prob > 0.5 else "Benign"
    confidence = prob if prob > 0.5 else (1 - prob)
    
    return label, confidence * 100

# --- UI Layout ---
st.title("🩺 MedVision Skin Detection")
st.write("Upload a lesion image for automated AI analysis.")

uploaded_image = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display image in a smaller centered column
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_image, width=300, caption="Uploaded Lesion")
        
        with st.spinner("Analyzing..."):
            label, score = predict(uploaded_image, model)
            
        if label == "Malignant":
            st.error(f"**Prediction: {label}**")
        else:
            st.success(f"**Prediction: {label}**")
            
        st.info(f"**Confidence:** {score:.2f}%")

st.caption("Developed by Venkatesh Gummadidala | Educational Project")