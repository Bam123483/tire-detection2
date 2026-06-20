from flask import Flask, request
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = Flask(__name__)

# ===== Load YOLO Model =====
model = YOLO("runs/detect/train14/weights/best.pt")

# ===== Website HTML =====
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Tire Damage Detector</title>

    <style>

        body{
            margin:0;
            padding:0;
            font-family: Arial, sans-serif;

            background:
            linear-gradient(
                135deg,
                #dff4ff,
                #cfe8ff,
                #e8f6ff,
                #d6eeff
            );

            min-height:100vh;

            color:#333;
        }

        .container{

            max-width: 900px;

            margin: 40px auto;

            background: rgba(255,255,255,0.9);

            backdrop-filter: blur(10px);

            border-radius: 30px;

            padding: 40px;

            box-shadow:
            0px 10px 35px rgba(0,0,0,0.08);

            text-align:center;
        }

        h1{

            font-size:48px;

            margin-bottom:10px;

            background: linear-gradient(
                to right,
                #4da8ff,
                #2563eb,
                #6ec6ff
            );

            -webkit-background-clip:text;

            -webkit-text-fill-color:transparent;
        }

        .subtitle{

            color:#5f6f82;

            margin-bottom:35px;

            font-size:18px;
        }

        #dropArea{

            border:3px dashed #7bbcff;

            border-radius:25px;

            padding:60px;

            background:
            linear-gradient(
                to bottom right,
                #f2faff,
                #e7f3ff
            );

            transition:0.25s;

            cursor:pointer;

            font-size:20px;

            color:#4d647d;

            box-shadow:
            inset 0px 2px 8px rgba(255,255,255,0.8);
        }

        #dropArea:hover{

            transform:scale(1.015);

            background:
            linear-gradient(
                to bottom right,
                #eaf6ff,
                #dff0ff
            );
        }

        #dropArea.dragover{

            border-color:#2563eb;

            background:
            linear-gradient(
                to bottom right,
                #dff1ff,
                #cfe8ff
            );
        }

        button{

            margin-top:28px;

            background:
            linear-gradient(
                to right,
                #60a5fa,
                #3b82f6,
                #38bdf8
            );

            color:white;

            border:none;

            padding:15px 36px;

            font-size:18px;

            border-radius:16px;

            cursor:pointer;

            transition:0.25s;

            box-shadow:
            0px 6px 18px rgba(59,130,246,0.25);
        }

        button:hover{

            transform:translateY(-3px) scale(1.02);

            box-shadow:
            0px 10px 22px rgba(59,130,246,0.35);
        }

        .loading{

            margin-top:20px;

            font-size:18px;

            color:#2563eb;

            display:none;

            font-weight:bold;
        }

        #message{

            margin-top:25px;

            font-size:24px;

            font-weight:bold;

            color:#1d4ed8;
        }

        .image-container{

            margin-top:35px;

            display:flex;

            justify-content:center;
        }

        img{

            width:500px;

            height:500px;

            object-fit:contain;

            background:
            linear-gradient(
                to bottom right,
                #ffffff,
                #fff0f6
            );

            border-radius:25px;

            border:6px solid #ffb6d9;

            box-shadow:
            0px 8px 25px rgba(0,0,0,0.12);
        }

        .footer{

            margin-top:30px;

            color:#7c8ca0;

            font-size:14px;
        }

    </style>
</head>

<body>

<div class="container">

    <h1>Tire Damage Detector</h1>

    <div class="subtitle">
        Upload or drag a tire image for AI damage detection
    </div>

    <div id="dropArea">
        📷 Drag & Drop Image Here
        <br><br>
        or Click to Upload
    </div>

    <button onclick="detect()">
        Detect Damage
    </button>

    <div class="loading" id="loading">
        Detecting image...
    </div>

    <div id="message"></div>

    <div class="image-container">
        <img id="resultImage">
    </div>

    <div class="footer">
        Powered by YOLO AI Detection
    </div>

</div>

<script>

let selectedFile = null;

const dropArea = document.getElementById("dropArea");

dropArea.addEventListener("click", () => {

    let input = document.createElement("input");

    input.type = "file";

    input.onchange = e => {

        selectedFile = e.target.files[0];

        dropArea.innerHTML =
            "✅ Selected: <b>" + selectedFile.name + "</b>";
    };

    input.click();
});

dropArea.addEventListener("dragover", (e) => {

    e.preventDefault();

    dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", () => {

    dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", (e) => {

    e.preventDefault();

    dropArea.classList.remove("dragover");

    selectedFile = e.dataTransfer.files[0];

    dropArea.innerHTML =
        "✅ Selected: <b>" + selectedFile.name + "</b>";
});

async function detect(){

    if(!selectedFile){

        alert("Please upload an image first.");
        return;
    }

    document.getElementById("loading").style.display = "block";

    let formData = new FormData();

    formData.append("image", selectedFile);

    let response = await fetch("/predict", {

        method:"POST",
        body:formData
    });

    let data = await response.json();

    document.getElementById("resultImage").src =
        "data:image/jpeg;base64," + data.image;

    document.getElementById("message").innerText =
        data.message;

    document.getElementById("loading").style.display = "none";
}

</script>

</body>
</html>
"""

# ===== Home Page =====
@app.route("/")
def home():
    return HTML

# ===== Prediction Route =====
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    image_bytes = file.read()

    image = Image.open(io.BytesIO(image_bytes))

    # ===== Run YOLO =====
    results = model(image, conf=0.10)

    # ===== Check Detections =====
    has_detection = len(results[0].boxes) > 0

    if has_detection:
        message = "⚠️ Damage detected on tire"
    else:
        message = "✅ No problems detected. Tire looks good."

    # ===== Draw Bounding Boxes =====
    plotted = results[0].plot()

    plotted_image = Image.fromarray(plotted)

    # ===== Convert Image =====
    buffer = io.BytesIO()

    plotted_image.save(buffer, format="JPEG")

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "image": image_base64,
        "message": message
    }

# ===== Run Website =====
app.run(debug=True)