import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.set_page_config(page_title="Tire Damage Detector")

st.title("🚗 Tire Damage Detector")
st.write("Upload a tire image and let AI check for damage.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Damage"):

        with st.spinner("Detecting..."):

            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name

            # Run YOLO
            results = model(temp_path, conf=0.10)

            # Check detections
            has_detection = len(results[0].boxes) > 0

            if has_detection:
                st.error("⚠️ Damage detected on tire")
            else:
                st.success("✅ No problems detected. Tire looks good.")

            # Draw boxes
            plotted = results[0].plot()

            st.image(
                plotted,
                caption="Detection Result",
                use_container_width=True
            )
