import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("mnist_cnn.h5")

st.title("Handwritten Digit Recognition")
st.write("Upload a 28x28 grayscale digit image.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=150)

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array
    img = np.array(image)

    # Normalize
    img = img.astype("float32") / 255.0

    # Reshape for CNN
    img = img.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img)
    digit = np.argmax(prediction)

    st.success(f"Predicted Digit: {digit}")
