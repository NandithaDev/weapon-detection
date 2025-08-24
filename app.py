# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load your trained model
model = YOLO("runs\\detect\\train\\weights\\best.pt")

st.title("🔫 Weapon Detection with YOLOv8")

# Initialize session state for current image
if "current_image" not in st.session_state:
    st.session_state.current_image = None

# 1️⃣ Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.session_state.current_image = Image.open(uploaded_file).convert("RGB")

# Hardcoded 4 sample images
sample_images = {
    "Sample 1": "my_img/1.jpeg",
    "Sample 2": "my_img/2.jpeg",
    "Sample 3": "my_img/3.jpg",
    "Sample 4": "my_img/4.jpg"
}

cols = st.columns(4)
for i, (name, path) in enumerate(sample_images.items()):
    with cols[i]:
        img = Image.open(path).convert("RGB")
        st.image(img, caption=name, use_column_width=True)
        if st.button(f"Use {name}"):
            st.session_state.current_image = img

# 3️⃣ Show the current image
if st.session_state.current_image:
    st.image(st.session_state.current_image, caption="Current Image", use_column_width=True)

    # Run prediction
    if st.button("Run Prediction"):
        results = model.predict(st.session_state.current_image, conf=0.25, imgsz=640)
        # Overlay predictions on the same image
        st.session_state.current_image = Image.fromarray(results[0].plot())
        st.image(st.session_state.current_image, caption="Predictions", use_column_width=True)
