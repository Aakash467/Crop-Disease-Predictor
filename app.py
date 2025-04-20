# app.py
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load model
model = load_model("model.h5")

# Class labels 
class_names = [
    'Rice_Bacterial Blight',
    'Rice_Blast',
    'Rice_Brown Spot',
    'Rice_Tungro',
    'Wheat_Brown Rust',
    'Wheat_Loose Smut',
    'Wheat_Septoria',
    'Wheat_Healthy',
    'Wheat_Yellow Rust'
] 
# App UI
st.set_page_config(page_title="Crop Leaf Disease Classifier", layout="centered")
st.title("🌾 Crop Leaf Disease Detection")
st.markdown("Upload a leaf image and choose crop type to detect the disease.")

# Crop type selection
crop_type = st.selectbox("Select Crop Type", ["Rice", "Wheat"])

# Image uploader
uploaded_file = st.file_uploader("Upload a Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image_resized = image.resize((224, 224))
    img_array = img_to_array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    pred_class = class_names[pred_index]
    confidence = float(np.max(predictions)) * 100

    # Show result
    st.subheader("Prediction:")
    st.success(f"**{pred_class}** with **{confidence:.2f}%** confidence")
