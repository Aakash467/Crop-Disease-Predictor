import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the trained model
model = tf.keras.models.load_model("crop_disease_model.h5")

# Define the class names
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

# Function to get advice from ChatGPT
def get_disease_measures(disease_name):
    prompt = f"""The crop disease is {disease_name}.
    Give 5 concise and practical measures a farmer must take to manage or control this disease."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an agriculture expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        reply = response['choices'][0]['message']['content']
        return reply
    except Exception as e:
        return f"❌ Error while getting advice: {e}"

# Streamlit UI
st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿")
st.title("🌾 Crop Disease Detection and Farmer Guidance")

uploaded_file = st.file_uploader("📷 Upload a crop leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    st.success(f"🧪 Predicted Disease: **{predicted_class}** (Confidence: {confidence:.2f})")

    # Get ChatGPT advice
    with st.spinner("🧠 Fetching expert advice..."):
        advice = get_disease_measures(predicted_class)

    st.subheader("✅ Suggested Measures for Farmers:")
    st.markdown(advice)
