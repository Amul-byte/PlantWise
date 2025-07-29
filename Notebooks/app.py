import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# === CONFIGURATION ===
MODEL_PATH = "model/plantwise_model.h5"
IMG_SIZE = (224, 224)

# === Load Model ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# === Get Class Names from Training Generator (manually inputted based on your training) ===
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_YellowLeafCurlVirus',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
]

# === Title ===
st.title("🌿 Plant Disease Classifier")
st.write("Upload an image of a plant leaf, and the model will predict the disease class.")

# === File Upload ===
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# === Prediction ===
if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    # Display result
    st.success(f"🧠 Predicted: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
