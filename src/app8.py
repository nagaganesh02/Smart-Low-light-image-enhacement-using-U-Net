import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tempfile
import os

# Page Configuration
st.set_page_config(page_title="Low-Light Image Enhancement", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
    h1, h2, h3 {
        font-size: 36px !important;
        color: #00cccc;
    }
    p, li {
        font-size: 18px !important;
        line-height: 1.7;
    }
    .stTabs [role="tab"] {
        font-size: 18px;
        padding: 12px 18px;
        color: #333;
    }
    .stSidebar .css-1v3fvcr {
        padding-top: 20px;
    }
    .stSlider > div {
        padding: 10px 0;
    }
    .css-1d391kg {
        padding-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Enhancement Functions --------------------
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
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def adjust_saturation_sharpness(image, saturation=1.0, sharpness=1.0):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = ImageEnhance.Color(img).enhance(saturation)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# -------------------- Navigation Tabs --------------------
tab1, tab2 = st.tabs(["🏠 Home", "🖼️ Generate"])

# -------------------- Home Page --------------------
with tab1:
    st.markdown("<h1 style='text-align: center;'>🌙✨ Low-Light Image Enhancement</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: justify;'>
        Welcome to our interactive platform designed to enhance images captured in low-light conditions.
        This application leverages a combination of traditional image processing methods and modern enhancements:
        </p>
        <ul>
            <li><b>CLAHE</b> (Contrast Limited Adaptive Histogram Equalization)</li>
            <li><b>Gamma Correction</b></li>
            <li><b>White Balance</b></li>
            <li><b>Brightness & Contrast Adjustment</b></li>
            <li><b>Saturation & Sharpness Tuning</b></li>
        </ul>
        <p style='text-align: justify;'>
        Upload an image or use your webcam to enhance visuals instantly and naturally, even in the darkest settings.
        </p>
    """, unsafe_allow_html=True)

# -------------------- Generate Page --------------------
with tab2:
    st.markdown("## 🖼️ Enhance Your Image")
    input_option = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])
    image = None

    if input_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)

    elif input_option == "Use Webcam":
        webcam_image = st.camera_input("Capture a photo")
        if webcam_image:
            image = Image.open(webcam_image)

    if image:
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Sidebar for adjustments
        st.sidebar.header("🔧 Enhancement Controls")
        gamma = st.sidebar.slider("Gamma Correction", 0.5, 3.0, 1.8)
        clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 3.0)
        grid_size = st.sidebar.slider("CLAHE Grid Size", 4, 16, 8, step=2)
        brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0)
        contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)
        saturation = st.sidebar.slider("Saturation", 0.5, 2.0, 1.0)
        sharpness = st.sidebar.slider("Sharpness", 0.5, 2.0, 1.0)

        # Apply enhancements
        enhanced_image = apply_clahe(image, clip_limit, (grid_size, grid_size))
        enhanced_image = gamma_correction(enhanced_image, gamma)
        enhanced_image = white_balance(enhanced_image)
        enhanced_image = adjust_brightness_contrast(enhanced_image, brightness, contrast)
        enhanced_image = adjust_saturation_sharpness(enhanced_image, saturation, sharpness)

        # Display images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="📷 Original Image", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB), caption="✨ Enhanced Image", use_container_width=True)

        # Download enhanced image
        _, tmp = tempfile.mkstemp(suffix=".png")
        cv2.imwrite(tmp, enhanced_image)
        with open(tmp, "rb") as file:
            st.download_button("📥 Download Enhanced Image", file, file_name="enhanced_image.png", mime="image/png")

