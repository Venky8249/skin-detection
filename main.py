import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image

# 1. Load the ONNX model (Much faster than TensorFlow)
@st.cache_resource
def load_onnx_model():
    return ort.InferenceSession("model.onnx")

session = load_onnx_model()

def predict_skin_cancer(uploaded_image):
    # Preprocessing with PIL instead of tensorflow.keras
    img = Image.open(uploaded_image).resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # 2. Run Inference
    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: img_array})[0]
    
    prob = prediction[0][0]
    return "Malignant" if prob > 0.5 else "Benign"

# Streamlit UI
st.title("Skin Cancer Detection (Ultra Lite)")
uploaded_file = st.file_uploader("Upload lesion image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    label = predict_skin_cancer(uploaded_file)
    st.image(uploaded_file, width=300)
    st.success(f"Prediction: **{label}**")