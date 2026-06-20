import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🚗 Tire Damage Detector")

uploaded_file = st.file_uploader(
    "Upload a tire image",
    type=["jpg", "jpeg", "png"]
)

# AUTO DETECT
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Detecting..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp:

            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        results = model(temp_path, conf=0.10)

        has_detection = len(results[0].boxes) > 0

        if has_detection:
            st.error("⚠️ Damage detected on tire")
        else:
            st.success("✅ No problems detected. Tire looks good.")

        plotted = results[0].plot()

        st.image(
            plotted,
            caption="Detection Result",
            use_container_width=True
        )
