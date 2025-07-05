import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image

# --- Custom CSS for Background and Styling ---
page_bg_img = """
<style>
    body {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }
    .stApp {
        background: url("https://source.unsplash.com/1600x900/?dark,technology") no-repeat center fixed;
        background-size: cover;
    }
    .stTitle, .stHeader {
        text-align: center;
        font-family: 'Poppins', sans-serif;
    }
    .stDownloadButton {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
    }
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- UI HEADER ---
st.markdown("<h1 style='text-align: center;'>🌙 Low-Light Image Enhancement</h1>", unsafe_allow_html=True)
st.write("🚀 Upload a low-light image to enhance it. Customize settings for the best results.")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Enhancement Settings")
gamma_value = st.sidebar.slider("🎚️ Gamma Correction", 0.5, 3.0, 1.8, 0.1)
clahe_clip = st.sidebar.slider("🌟 CLAHE Clip Limit", 1.0, 5.0, 3.0, 0.1)
tile_size = st.sidebar.slider("📏 CLAHE Tile Grid Size", 4, 16, 8, 2)

# --- Enhancement Functions ---
def apply_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
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

# --- Upload Image ---
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="🖼️ Original Image", use_column_width=True)

    # --- Apply Enhancements ---
    enhanced_image = apply_clahe(image, clip_limit=clahe_clip, tile_grid_size=(tile_size, tile_size))
    enhanced_image = gamma_correction(enhanced_image, gamma=gamma_value)
    enhanced_image = white_balance(enhanced_image)

    with col2:
        st.image(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB), caption="✨ Enhanced Image", use_column_width=True)

    # --- Download Option ---
    _, temp_filename = tempfile.mkstemp(suffix=".png")
    cv2.imwrite(temp_filename, enhanced_image)
    with open(temp_filename, "rb") as file:
        st.download_button("📥 Download Enhanced Image", file, file_name="enhanced_image.png", mime="image/png")

# --- Footer ---
st.markdown("<h4 style='text-align: center;'>🔧 Developed by Your Name</h4>", unsafe_allow_html=True)
