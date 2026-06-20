import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

@st.cache_resource
def load_model():
    return YOLO("runs/detect/train14/weights/best.pt")

model = load_model()

st.set_page_config(page_title="Tire Damage Detector")

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp:

        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    results = model(temp_path, conf=0.10)

    plotted = results[0].plot()

    st.image(
        plotted,
        use_container_width=True
    )
