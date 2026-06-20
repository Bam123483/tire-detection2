import streamlit as st
from ultralytics import YOLO
import tempfile
import os

# =========================
# Page Settings
# =========================
st.set_page_config(
    page_title="Tire Damage Detector",
    layout="centered"
)

# =========================
# Orange & Black Theme
# =========================
st.markdown("""
<style>

.stApp {
    background-color: #000000;
}

[data-testid="stFileUploader"] {
    background-color: #111111;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================
st.markdown("""
<h1 style='
text-align:center;
color:#ff8800;
font-size:55px;
font-weight:bold;
'>
🚗 Tire Damage Detector
</h1>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;color:white;'>Upload a tire image for AI damage detection</p>",
    unsafe_allow_html=True
)

# =========================
# Load YOLO Model
# =========================
@st.cache_resource
def load_model():

    MODEL_PATH = "best.pt"

    if not os.path.exists(MODEL_PATH):
        st.error(
            "best.pt not found. Upload your YOLO model to the GitHub repository."
        )
        st.stop()

    return YOLO(MODEL_PATH)

model = load_model()

# =========================
# Upload Image
# =========================
uploaded_file = st.file_uploader(
    "Choose a tire image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# Auto Detection
# =========================
if uploaded_file is not None:

    with st.spinner("Detecting Damage..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp:

            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        results = model(
            temp_path,
            conf=0.10
        )

        plotted = results[0].plot()

        st.image(
            plotted,
            use_container_width=True
        )
