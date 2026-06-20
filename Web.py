import streamlit as st
from ultralytics import YOLO
import tempfile

st.set_page_config(
    page_title="Tire Damage Detector",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-color: #000000;
}

h1 {
    color: #ff8800;
    text-align: center;
}

[data-testid="stFileUploader"] {
    background-color: #111111;
    border: 2px solid #ff8800;
    border-radius: 10px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("runs/detect/train14/weights/best.pt")

model = load_model()

st.title("🚗 Tire Damage Detector")

uploaded_file = st.file_uploader(
    "Upload a Tire Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp:

        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    with st.spinner("Detecting..."):

        results = model(temp_path, conf=0.10)

        plotted = results[0].plot()

        st.image(
            plotted,
            use_container_width=True
        )
