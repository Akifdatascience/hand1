import streamlit as st
import onnx
import onnx_tf.backend as backend
import numpy as np
from PIL import Image
import io

# Load your ONNX model
onnx_model = onnx.load('model.onnx')
tf_rep = backend.prepare(onnx_model)

# Create a Streamlit sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Choose a handwritten image...", type=["jpg", "png", "jpeg"])

# Check if an image has been uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for prediction
    # You may need to adjust the preprocessing steps based on your model
    image = image.resize((224, 224))  # Resize the image to match your model's input size
    image = np.array(image)
    image = image / 255.0  # Normalize the pixel values

    # Make a prediction using the ONNX model
    input_data = np.expand_dims(image, axis=0)
    prediction = tf_rep.run(input_data)

    # Post-process the prediction (e.g., extracting text)

    # Display the extracted text
    st.subheader('Extracted Text')
    st.write("Your extracted text here")

# Add any other Streamlit components and layout as needed
