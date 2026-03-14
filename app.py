import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from models.csrnet import CSRNet

# ---------------------------- Flask Setup ----------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# ---------------------------- Device & Model ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CSRNet()
checkpoint_path = "weights/csrnet_model_best.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()
print("CSRNet loaded.")

# ---------------------------- Helpers ----------------------------
def allowed_file(filename, filetype="image"):
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    if filetype == "image":
        return ext in ALLOWED_IMAGE_EXTENSIONS
    else:
        return ext in ALLOWED_VIDEO_EXTENSIONS

def predict_count(image):
    """Predict number of people in a BGR image"""
    if image is None or image.size == 0:
        return 0
    img_resized = cv2.resize(image, (512, 384))
    img_resized = img_resized / 255.0
    img_resized = np.transpose(img_resized, (2, 0, 1))
    img_tensor = torch.tensor(img_resized).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_density = model(img_tensor)
        pred_density = F.interpolate(pred_density, size=(384, 512), mode='bilinear', align_corners=False)
        count = pred_density.sum().item()
    return int(round(count))

# ---------------------------- Routes ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename, "image"):
        return jsonify({"error": "Unsupported image type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    if image is None:
        return jsonify({"error": "Cannot read image"}), 400

    count = predict_count(image)
    return jsonify({"filename": filename, "predicted_count": count})

@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video part"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename, "video"):
        return jsonify({"error": "Unsupported video type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 400

    frame_counts = []
    frame_skip = 10  # Process every 10th frame
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is not None and frame.size > 0 and frame_idx % frame_skip == 0:
            count = predict_count(frame)
            frame_counts.append(count)
        frame_idx += 1

    cap.release()
    avg_count = int(round(np.mean(frame_counts))) if frame_counts else 0
    return jsonify({"filename": filename, "average_count": avg_count, "frames_processed": len(frame_counts)})

@app.route("/live_webcam", methods=["GET"])
def live_webcam():
    return jsonify({"message": "Live webcam streaming not yet implemented."})

# ---------------------------- Run Flask ----------------------------
if __name__ == "__main__":
    app.run(debug=True)