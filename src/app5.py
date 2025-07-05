import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import rawpy
import imageio
import tempfile
import os

def load_image(file):
    suffix = os.path.splitext(file.name)[-1].lower()
    raw_formats = ['.dng', '.nef', '.cr2', '.arw', '.orf', '.rw2']
    if suffix in raw_formats:
        with rawpy.imread(file) as raw:
            rgb = raw.postprocess()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        image = Image.open(file).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Enhancement Functions
def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def gamma_correction(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def white_balance(image):
    wb = cv2.xphoto.createSimpleWB()
    return wb.balanceWhite(image)

def denoise_image(image, strength=10):
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)

def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def adjust_saturation_sharpness(image, saturation=1.0, sharpness=1.0):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = ImageEnhance.Color(img).enhance(saturation)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Streamlit App
st.set_page_config(page_title="Low-Light Image Enhancement", layout="wide")
st.title("🌙✨ Low-Light Image Enhancement (All Formats Supported)")
st.markdown("### Upload any image format including RAW (.dng, .nef, .cr2, etc.)")

option = st.radio("Choose an input method:", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "dng", "nef", "cr2", "arw", "orf", "rw2"])
    if uploaded_file:
        try:
            image = load_image(uploaded_file)
        except Exception as e:
            st.error(f"Failed to load image: {e}")
            image = None
elif option == "Use Webcam":
    webcam_image = st.camera_input("Capture a photo")
    if webcam_image:
        image = np.array(Image.open(webcam_image).convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Enhancement
if 'image' in locals() and image is not None:
    st.sidebar.header("🔧 Enhancement Controls")
    gamma = st.sidebar.slider("Gamma", 0.5, 2.5, 1.2)
    clahe_clip = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0)
    grid_size = st.sidebar.slider("CLAHE Grid Size", 4, 16, 8, step=2)
    brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)
    saturation = st.sidebar.slider("Saturation", 0.5, 2.0, 1.0)
    sharpness = st.sidebar.slider("Sharpness", 0.5, 2.0, 1.0)
    denoise_strength = st.sidebar.slider("Denoise Strength", 0, 30, 10)

    # Pipeline
    enhanced = apply_clahe(image, clahe_clip, (grid_size, grid_size))
    enhanced = gamma_correction(enhanced, gamma)
    enhanced = white_balance(enhanced)
    enhanced = adjust_brightness_contrast(enhanced, brightness, contrast)
    enhanced = adjust_saturation_sharpness(enhanced, saturation, sharpness)
    enhanced = denoise_image(enhanced, denoise_strength)

    # Show Results
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="📷 Original Image", use_column_width=True)
    with col2:
        st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), caption="✨ Enhanced Image", use_column_width=True)

    _, temp_filename = tempfile.mkstemp(suffix=".png")
    cv2.imwrite(temp_filename, enhanced)
    with open(temp_filename, "rb") as f:
        st.download_button("📥 Download Enhanced Image", f, file_name="enhanced.png", mime="image/png")

st.sidebar.markdown("---")
st.sidebar.write("Developed by **Your Name**")
