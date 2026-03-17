import streamlit as st
import numpy as np
import pickle
from PIL import Image  # Replaces tensorflow.keras.preprocessing

@st.cache_resource
def load_my_model():
    model_path = 'trained_model.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_my_model()

def predict_skin_cancer(uploaded_image, model):
    # Preprocessing with PIL (Much lighter than TensorFlow)
    img = Image.open(uploaded_image).convert('RGB').resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Flatten if your pickle model is a Scikit-Learn model (RandomForest, etc.)
    # img_array = img_array.flatten().reshape(1, -1) 

    prediction = model.predict(img_array)
    
    # Standard prediction logic
    prob = prediction[0] if np.isscalar(prediction[0]) else prediction[0][0]
    return "Malignant" if prob > 0.5 else "Benign"

# UI Logic
st.title("Skin Cancer Detection")
uploaded_file = st.file_uploader("Upload lesion image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    label = predict_skin_cancer(uploaded_file, model)
    st.image(uploaded_file, width=300)
    st.success(f"Prediction: **{label}**")