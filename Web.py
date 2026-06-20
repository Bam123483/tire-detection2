import streamlit as st
from ultralytics import YOLO
import tempfile

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
    border: 2px solid #ff8800;
    border-radius: 15px;
    padding: 10px;
}

[data-testid="stSpinner"] {
    color: #ff8800;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Glowing Title
# =========================
st.markdown("""
<h1 style='
text-align:center;
color:#ff8800;
font-size:55px;
text-shadow:
0 0 10px #ff8800,
0 0 20px #ff8800,
0 0 30px #ff8800;
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
    return YOLO("runs/detect/train14/weights/best.pt")

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
