from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import tempfile

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Enhancement Functions
def apply_clahe(image, clip_limit=3.0, grid_size=(8,8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def gamma_correction(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def white_balance(image):
    wb = cv2.xphoto.createSimpleWB()
    return wb.balanceWhite(image)

def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def adjust_saturation_sharpness(image, saturation=1.0, sharpness=1.0):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    return jsonify({"filename": file.filename})

@app.route("/enhance", methods=["POST"])
def enhance():
    data = request.json
    filename = data["filename"]
    enhancements = data["enhancements"]

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Check if file exists
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 400

    # Read the image
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({"error": "Failed to load image. Check file path and integrity."}), 400

    # Apply enhancements safely
    if enhancements.get("clahe", False):
        image = apply_clahe(image, enhancements.get("clip_limit", 3.0), (enhancements.get("grid_size", 8), enhancements.get("grid_size", 8)))
    if enhancements.get("gamma", False):
        image = gamma_correction(image, enhancements.get("gamma_value", 1.5))
    if enhancements.get("white_balance", False):  
        image = white_balance(image)
    if enhancements.get("brightness_contrast", False):
        image = adjust_brightness_contrast(image, enhancements.get("brightness", 1.0), enhancements.get("contrast", 1.0))
    if enhancements.get("saturation_sharpness", False):
        image = adjust_saturation_sharpness(image, enhancements.get("saturation", 1.0), enhancements.get("sharpness", 1.0))

    # Save enhanced image
    temp_filename = tempfile.mktemp(suffix=".png")
    cv2.imwrite(temp_filename, image)

    return send_file(temp_filename, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
