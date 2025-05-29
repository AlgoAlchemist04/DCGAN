import streamlit as st
import os
import re
import time
from PIL import Image

# Page config
st.set_page_config(page_title="Lunar GAN Explorer")

# Background styling
def add_bg_from_local():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("file-B13rv22hYFZDNJwbZjZ3MP");
            background-size: cover;
        }}
        .css-18e3th9 {{ background-color: rgba(0,0,0,0.85) !important; }}
        .css-1d391kg {{ color: white !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

add_bg_from_local()

st.title("🌙 Lunar Image Enhancement with GANs")
st.markdown("Explore AI-enhanced moon images generated across training epochs.")

# Image directory setup
IMAGE_DIR = "generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Extract epoch number
def extract_epoch(filename):
    match = re.search(r"epoch_(\d+)", filename)
    return int(match.group(1)) if match else float("inf")

# Sorted list of image paths
def get_image_list():
    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]
    files.sort(key=extract_epoch)
    return [os.path.join(IMAGE_DIR, f) for f in files]

image_list = get_image_list()

# --- Main Mode Selection ---
mode = st.radio("Choose Mode", ["Single View", "Comparison", "Animation"], horizontal=True)

if image_list:

    # --- Single Image View ---
    if mode == "Single View":
        idx = st.slider("Select an image", 0, len(image_list) - 1, 0)
        img = Image.open(image_list[idx])
        st.image(img, caption=f"Generated Image - Epoch {extract_epoch(image_list[idx])}", use_column_width=True)

        with open(image_list[idx], "rb") as file:
            st.download_button("Download Image", file.read(), os.path.basename(image_list[idx]), mime="image/png")

    # --- Comparison Mode ---
    elif mode == "Comparison":
        col1, col2 = st.columns(2)
        with col1:
            idx1 = st.slider("Select First Image", 0, len(image_list) - 1, 0, key="img1")
            img1 = Image.open(image_list[idx1])
            st.image(img1, caption=f"Epoch {extract_epoch(image_list[idx1])}", use_column_width=True)

        with col2:
            idx2 = st.slider("Select Second Image", 0, len(image_list) - 1, len(image_list) - 1, key="img2")
            img2 = Image.open(image_list[idx2])
            st.image(img2, caption=f"Epoch {extract_epoch(image_list[idx2])}", use_column_width=True)

    # --- Animation Mode ---
    elif mode == "Animation":
        st.write("Use the slider or press 'Play' to animate through epochs.")
        speed = st.slider("Speed (seconds per frame)", 0.1, 2.0, 0.5, 0.1)
        
        if "anim_frame" not in st.session_state:
            st.session_state.anim_frame = 0
            st.session_state.animating = False

        col_play, col_slider = st.columns([1, 5])
        with col_play:
            if st.button("▶️ Play"):
                st.session_state.animating = True
            if st.button("⏸️ Pause"):
                st.session_state.animating = False

        with col_slider:
            st.session_state.anim_frame = st.slider("Epoch Frame", 0, len(image_list) - 1, st.session_state.anim_frame)

        img = Image.open(image_list[st.session_state.anim_frame])
        st.image(img, caption=f"Epoch {extract_epoch(image_list[st.session_state.anim_frame])}", use_column_width=True)

        if st.session_state.animating:
            time.sleep(speed)
            st.session_state.anim_frame += 1
            if st.session_state.anim_frame >= len(image_list):
                st.session_state.anim_frame = 0
            st.experimental_rerun()

else:
    st.warning("⚠️ No images found. Generate images to view them here.")
