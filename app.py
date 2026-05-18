import streamlit as st
import cv2
from rembg import remove, new_session
from PIL import Image, ImageDraw
import numpy as np
import os
import zipfile
import io

# --- 1. Page Configuration & Setup ---
st.set_page_config(page_title="Green Screen Studio Pro", layout="wide")

# --- 2. Optimized AI Loading ---
@st.cache_resource
def get_ai_session(model_name):
    return new_session(model_name)

# --- 3. Helper: Despill Algorithm ---
def apply_despill(img_pil, strength):
    if strength <= 0: return img_pil
    data = np.array(img_pil).astype(float)
    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
    avg_rb = (r + b) / 2
    mask = g > avg_rb
    new_g = np.where(mask, g - (g - avg_rb) * strength, g)
    data[:,:,1] = new_g
    return Image.fromarray(data.astype(np.uint8))

# --- 4. Helper: Checkerboard for Transparency ---
def apply_checkerboard(img_rgba):
    width, height = img_rgba.size
    grid_size = 20
    checkerboard = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(checkerboard)
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            if (x // grid_size + y // grid_size) % 2 == 1:
                draw.rectangle([x, y, x + grid_size, y + grid_size], fill=(200, 200, 200))
    checkerboard.paste(img_rgba, mask=img_rgba.split()[-1])
    return checkerboard

# --- 5. Main Frame Processor ---
def process_frame(input_image, target_color, sens, d_strength, removal_method, session, use_transparent, bg_fill_rgb):
    input_image = apply_despill(input_image, d_strength)
    
    if removal_method == "AI (Human Segment)":
        no_bg = remove(input_image, session=session)
    else:
        cv_img = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        target_bgr = np.array(target_color)[::-1]
        lower = np.clip(target_bgr - sens, 0, 255)
        upper = np.clip(target_bgr + sens, 0, 255)
        mask = cv2.inRange(cv_img, lower, upper)
        mask = cv2.medianBlur(mask, 3)
        mask_inv = cv2.bitwise_not(mask)
        
        img_rgba = input_image.convert("RGBA")
        data = np.array(img_rgba)
        data[:, :, 3] = mask_inv
        no_bg = Image.fromarray(data)
    
    if use_transparent:
        return no_bg
    else:
        final_img = Image.new("RGB", no_bg.size, bg_fill_rgb)
        final_img.paste(no_bg, mask=no_bg.split()[-1])
        return final_img

# --- 6. Sidebar UI ---
st.title("🎬 Green Screen Studio Pro")

st.sidebar.header("Global Settings")
removal_method = st.sidebar.radio("Removal Method", ["AI (Human Segment)", "Manual Color Key"])
model_choice = st.sidebar.selectbox("Choose AI Model", ["u2net", "u2netp", "u2net_human_seg", "silueta"], index=2)
use_transparent = st.sidebar.checkbox("Transparent Background (Alpha)", value=True)

bg_fill_rgb = (0, 0, 0)
if not use_transparent:
    bg_fill_color = st.sidebar.color_picker("New Background Color", "#000000")
    bg_fill_rgb = tuple(int(bg_fill_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

mode = st.sidebar.radio("Output Mode", ["Video", "Photo", "PNG Frames"])

# --- 7. File Uploader ---
uploaded_file = st.file_uploader("Upload File", type=["mp4", "mov", "avi", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    if 'picked_color' not in st.session_state:
        st.session_state.picked_color = (0, 255, 0)
    sensitivity = 50
    d_strength = 0.5
    current_session = get_ai_session(model_choice)
    st.success("✅ AI Session Active")

    if removal_method == "Manual Color Key":
        st.subheader("🛠️ Keying Setup & Live Preview")
        if uploaded_file.type.startswith('image'):
            sample_img = Image.open(uploaded_file).convert("RGB")
        else:
            with open("temp_sample.mp4", "wb") as f:
                f.write(uploaded_file.getvalue())
            v = cv2.VideoCapture("temp_sample.mp4")
            ret, frame = v.read()
            sample_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            v.release()

        w, h = sample_img.size
        col_eye, col_settings, col_preview = st.columns([1.5, 1, 1.5])
        with col_settings:
            st.markdown("**1. Eyedropper**")
            pick_x = st.slider("X Position", 0, w, w//2)
            pick_y = st.slider("Y Position", 0, h, h//2)
            st.session_state.picked_color = sample_img.getpixel((pick_x, pick_y))
            st.color_picker("Sampled Color", "#%02x%02x%02x" % st.session_state.picked_color, disabled=True)
            st.markdown("**2. Keyer Settings**")
            sensitivity = st.slider("Color Range", 10, 150, 50)
            d_strength = st.slider("Despill Strength", 0.0, 1.0, 0.5)

        with col_eye:
            draw_img = sample_img.copy()
            draw = ImageDraw.Draw(draw_img)
            draw.line((pick_x, 0, pick_x, h), fill="red", width=2)
            draw.line((0, pick_y, w, pick_y), fill="red", width=2)
            st.image(draw_img, caption="Targeting Crosshair")

        with col_preview:
            live_result = process_frame(sample_img, st.session_state.picked_color, sensitivity, d_strength, removal_method, current_session, use_transparent, bg_fill_rgb)
            st.image(apply_checkerboard(live_result) if use_transparent else live_result, caption="Live Preview")

    st.divider()
    if st.button(f"🚀 Start {mode} Processing"):

        if mode == "Photo":
            img = Image.open(uploaded_file).convert("RGB")
            final = process_frame(img, st.session_state.picked_color, sensitivity, d_strength, removal_method, current_session, use_transparent, bg_fill_rgb)
            st.image(apply_checkerboard(final) if use_transparent else final)
            img_format = "PNG" if use_transparent else "JPEG"
            final.save("result_photo.png", format=img_format)
            with open("result_photo.png", "rb") as f:
                st.download_button("💾 Download Image", f, f"processed_image.{img_format.lower()}", "image/png")

        elif mode in ["Video", "PNG Frames"]:
            with open("temp_run.mp4", "wb") as f:
                f.write(uploaded_file.getvalue())
            cap = cv2.VideoCapture("temp_run.mp4")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            zip_buffer = io.BytesIO() if mode == "PNG Frames" else None
            out = None

            if mode == "Video":
                ext = "mov" if use_transparent else "mp4"
                result_filename = f"result_video.{ext}"
                fourcc = cv2.VideoWriter_fourcc('png ') if use_transparent else cv2.VideoWriter_fourcc('mp4v')
                out = cv2.VideoWriter(result_filename, fourcc, fps, (width, height))

            p_col1, p_col2 = st.columns(2)
            orig_placeholder = p_col1.empty()
            proc_placeholder = p_col2.empty()
            prog = st.progress(0)

            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                pil_f = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                final = process_frame(pil_f, st.session_state.picked_color, sensitivity, d_strength, removal_method, current_session, use_transparent, bg_fill_rgb)

                if mode == "Video":
                    if use_transparent:
                        out.write(cv2.cvtColor(np.array(final), cv2.COLOR_RGBA2BGRA))
                    else:
                        out.write(cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR))
                elif mode == "PNG Frames":
                    buf = io.BytesIO()
                    final.save(buf, format="PNG")
                    buf.seek(0)
                    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, True) as zip_file:
                        zip_file.writestr(f"frame_{count:04d}.png", buf.getvalue())

                orig_placeholder.image(frame, channels="BGR")
                proc_placeholder.image(np.array(apply_checkerboard(final) if use_transparent else final))
                count += 1
                prog.progress(min(count / frame_count, 1.0))

            cap.release()
            if out:
                out.release()
            st.success("✅ Complete!")

            if mode == "Video":
                with open(result_filename, "rb") as f:
                    st.download_button("💾 Download Processed Video", f, result_filename, "video/quicktime" if use_transparent else "video/mp4")
            elif mode == "PNG Frames":
                zip_buffer.seek(0)
                st.download_button("💾 Download PNG Frames (ZIP)", zip_buffer, "frame_sequence.zip", "application/zip")
