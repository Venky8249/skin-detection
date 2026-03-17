import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image

# 1. Use pickle.load instead of keras.load_model
@st.cache_resource
def load_my_model():
    # Ensure the path points correctly to your pickle file
    model_path = 'trained_model.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Initialize model
try:
    model = load_my_model()
except Exception as e:
    st.error(f"Failed to load the pickle model: {e}")
    st.stop()

def predict_skin_cancer(uploaded_image, model):
    # Preprocessing
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 2. Prediction
    # Since it's a pickle-loaded model, we use the standard .predict method
    prediction = model.predict(img_array)
    
    # Logic for Benign (0) vs Malignant (1)
    # Check if prediction is a single value or a list
    prob = prediction[0][0] if hasattr(prediction, "__len__") else prediction
    class_label = "Malignant" if prob > 0.5 else "Benign"
    
    return class_label

# Streamlit UI
st.title("Skin Cancer Detection")

uploaded_file = st.file_uploader("Upload lesion image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    label = predict_skin_cancer(uploaded_file, model)
    st.image(uploaded_file, width=300)
    st.success(f"Prediction: **{label}**")