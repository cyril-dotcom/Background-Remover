import streamlit as st
import cv2
from rembg import remove, new_session
from PIL import Image, ImageDraw
import numpy as np
import time
import os

# 1. Start Check
st.write("✨ App engine loaded...")

# 2. Model Check (This is usually where it hangs)
@st.cache_resource
def load_ai():
    st.write("🧠 Loading AI models (this takes a moment on first run)...")
    return new_session("u2net_human_seg")

session = load_ai()
st.write("✅ AI ready!")

# --- Rest of your UI ---
st.title("🎬 Green Screen Studio Pro")

# (Keep the rest of your settings logic here...)
removal_method = st.sidebar.radio("Removal Method", ["AI (Human Segment)", "Manual Color Key"])

# IMPORTANT: Wrap the video sampling in an IF statement 
# so it ONLY runs if a file is actually there.
uploaded_file = st.file_uploader("Upload File", type=["mp4", "mov", "avi", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.write("📁 File received, preparing preview...")
    # ... rest of your eyedropper and processing code ...
else:
    st.info("👋 Welcome! Please upload a video or photo to begin.")
