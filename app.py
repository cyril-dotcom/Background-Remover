import streamlit as st
import cv2
from rembg import remove, new_session
from PIL import Image, ImageDraw
import numpy as np
import time
import os

# 1. PAGE SETUP (Keep this at the very top)
st.set_page_config(page_title="Green Screen Studio Pro", layout="wide")

# 2. CACHE THE SESSION (This prevents it from hanging the UI)
@st.cache_resource
def get_ai_session(model_name):
    return new_session(model_name)

# --- UI STARTS HERE ---
st.title("🎬 Green Screen Studio Pro")

# Sidebar
st.sidebar.header("Settings")
removal_method = st.sidebar.radio("Removal Method", ["AI (Human Segment)", "Manual Color Key"])
model_choice = st.sidebar.selectbox("Choose AI Model", ["u2net", "u2netp", "u2net_human_seg", "silueta"], index=2)
use_transparent = st.sidebar.checkbox("Transparent Background (Alpha)", value=False)

# Background Color
bg_fill_rgb = (0, 0, 0)
if not use_transparent:
    bg_fill_color = st.sidebar.color_picker("New Background Color", "#000000") 
    bg_fill_rgb = tuple(int(bg_fill_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

mode = st.sidebar.radio("Output Mode", ["Video", "Individual Frames", "Photo"])

# 3. FILE UPLOADER (The app will now show up even if AI isn't ready)
uploaded_file = st.file_uploader("Upload File", type=["mp4", "mov", "avi", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # ONLY load the session when a file is uploaded
    session = get_ai_session(model_choice)
    st.success("AI Session Active")
    
    # ... (Keep all your Eyedropper and Processing logic here) ...
    # (Just make sure you use 'session' inside your process_frame function)
else:
    st.info("👋 Upload a file to see the settings and preview!")
