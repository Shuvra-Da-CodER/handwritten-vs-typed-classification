import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import io

st.set_page_config(
    page_title="Text Classifier",
    page_icon="✍️",
    layout="centered"
)

@st.cache_resource
def load_keras_model(model_path):
    """Loads a Keras model from an H5 file."""
    model = tf.keras.models.load_model(model_path)
    return model

try:
    model= load_keras_model('handwriting_classifier_model.h5')
except (FileNotFoundError, IOError):
    st.error("Model file not found. Please make sure 'handwriting_classifier.h5' is in the same folder as app.py.")
    st.stop()

def preprocess_image(image_bytes):
    """Preprocesses the image for the Keras model."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((350, 350)) 
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

st.title("✍️ Handwritten vs. Typed Text Classifier")
st.write("Upload an image, and the model will determine if the text is handwritten or typed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","tiff","tif"])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("") # Add a little space

    # When the user clicks the button
    if st.button("Classify Text"):
        with st.spinner("Analyzing the image..."):
            # Preprocess the image and get prediction
            image_bytes = uploaded_file.getvalue()
            preprocessed_image = preprocess_image(image_bytes)
            
            prediction_scores = model.predict(preprocessed_image)
            score = prediction_scores[0][0] 
            
            if score > 0.5:
                prediction = "Typed"
                confidence_score = score
            else:
                prediction = "Handwritten"
                confidence_score = 1 - score

            st.success(f"**Prediction:** {prediction}")
            st.info(f"**Confidence:** {confidence_score*100:.2f}%")