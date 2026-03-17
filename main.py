import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(page_title="Disease Detection AI", layout="centered")

# --- Model Loading ---
@st.cache_resource
def load_medical_model():
    model_path = 'trained_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Ensure it is uploaded to your repository.")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_medical_model()

# --- Image Preprocessing ---
def preprocess_image(image_data):
    # 1. Open and convert to RGB
    img = Image.open(image_data).convert('RGB')
    # 2. Resize to 224x224 (Standard for most Disease Detection CNNs)
    img = img.resize((224, 224))
    # 3. Convert to Numpy array and normalize
    img_array = np.array(img).astype('float32') / 255.0
    # 4. Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- UI Layout ---
st.title("🩺 AI Disease Diagnostic Tool")
st.write("Upload a clinical image to get an automated prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image at a medium size
    st.image(uploaded_file, caption="Uploaded Image", width=350)
    
    if model is not None:
        with st.spinner("Analyzing image..."):
            try:
                # Prepare data
                processed_img = preprocess_image(uploaded_file)
                
                # Make prediction
                prediction = model.predict(processed_img)
                
                # Logic: Adjust based on your model's output (0/1 or Probability)
                # This assumes a sigmoid output where > 0.5 is 'Positive'
                confidence = prediction[0][0] if hasattr(prediction[0], "__len__") else prediction[0]
                
                st.markdown("---")
                if confidence > 0.5:
                    st.error(f"### Result: Disease Detected")
                    st.write(f"**Confidence Level:** {confidence * 100:.2f}%")
                else:
                    st.success(f"### Result: Healthy / No Disease")
                    st.write(f"**Confidence Level:** {(1 - confidence) * 100:.2f}%")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")
    else:
        st.warning("Please ensure your 'trained_model.pkl' is correctly placed in the project folder.")

st.info("Disclaimer: This tool is for educational purposes and is not a substitute for professional medical advice.")