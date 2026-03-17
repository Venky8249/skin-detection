import os
import sys

# --- FIX 1: Force Legacy Keras & Redirect Imports ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
try:
    import tensorflow.keras as keras
    sys.modules['keras'] = keras
except ImportError:
    pass

import streamlit as st
import numpy as np
import pickle
from PIL import Image # Much lighter than tensorflow.keras.preprocessing

# --- FIX 2: Check for LFS Pointer Errors ---
@st.cache_resource
def load_my_model():
    model_path = 'trained_model.pkl'
    # If the file is tiny, it's just a Git LFS pointer text file
    if os.path.getsize(model_path) < 1000:
        st.error("Error: 'trained_model.pkl' is a Git LFS pointer. Please push the actual file data.")
        st.stop()
        
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Initialize model
try:
    model = load_my_model()
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

def predict_skin_cancer(uploaded_image, model):
    # Preprocessing with PIL instead of Keras
    img = Image.open(uploaded_image).convert('RGB').resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    
    # Handle both single value and array outputs
    prob = prediction[0][0] if hasattr(prediction[0], "__len__") else prediction[0]
    class_label = "Malignant" if prob > 0.5 else "Benign"
    
    return class_label

# --- Streamlit UI ---
st.title("🩺 Skin Cancer Detection")
st.write("Upload a lesion image to analyze.")

uploaded_file = st.file_uploader("Upload lesion image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with st.spinner("Analyzing..."):
        label = predict_skin_cancer(uploaded_file, model)
    
    # Center the smaller image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, width=300, caption="Uploaded Image")
        if label == "Malignant":
            st.error(f"Prediction: **{label}**")
        else:
            st.success(f"Prediction: **{label}**")