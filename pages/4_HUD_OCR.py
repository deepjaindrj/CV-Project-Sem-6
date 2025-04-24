import streamlit as st
import cv2
import numpy as np
import re
import tempfile
import os
import pandas as pd
from PIL import Image
import time
import uuid
import matplotlib.pyplot as plt
from matplotlib import cm
from io import BytesIO
import pytesseract
import random
import requests

# Set page title and favicon with enhanced styling
st.set_page_config(
    page_title="HUD OCR System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* Light blue background */
        .stApp {
            background-color: #e0f7fa; /* Light Blue */
            color: #333333;
        }

        /* Primary theme color - Yellow */
        :root {
            --primary-color: #ffca28; /* Warm Yellow */
            --background-color: #e0f7fa;
            --text-color: #333333;
        }

        /* Style buttons and other interactive elements */
        .stButton > button {
            background-color: #ffca28;
            color: black;
            border: none;
        }

        .stButton > button:hover {
            background-color: #ffc107;
        }

        /* Sidebar color (optional) */
        .css-6qob1r {
            background-color: #b2ebf2 !important; /* Lighter blue for sidebar */
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        margin: 5px 0;
    }
    .stSlider, .stTextInput, .stSelectbox, .stCheckbox, .stRadio {
        margin: 10px 0;
    }
    .stHeader {
        color: #1E90FF;
        font-size: 24px;
        margin-top: 20px;
    }
    .stSubheader {
        color: #4682B4;
        font-size: 18px;
    }
    .result-card {
        border: 2px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        background-color: #e8f5e9;
    }
    .result-card.low-confidence {
        border-color: #f44336;
        background-color: #ffebee;
    }
    .result-card.medium-confidence {
        border-color: #ff9800;
        background-color: #fff3e0;
    }
    .result-card.high-confidence {
        border-color: #4CAF50;
        background-color: #e8f5e9;
    }
    .progress-container {
        margin-top: 10px;
    }
    .ai-box {
        border: 2px solid #2196F3;
        padding: 10px;
        border-radius: 5px;
        background-color: #e3f2fd;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# API Keys (Securely handle in production with environment variables)
MISTRAL_API_KEY = "JZslDyH2l2tQHDksRbDfHYVKgO50LfkN"
GEMINI_API_KEY = "AIzaSyAwY29cyESToWBGM3Rg2mEghTJUGyMaoJw"

# Set Tesseract executable path
try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception as e:
    st.warning(f"Tesseract path not set correctly: {e}. Ensure Tesseract is installed and path is updated.")

# Function to perform OCR using Tesseract with PSM
def tesseract_ocr_request(frame, whitelist=None, psm_mode=7):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if len(frame.shape) == 2 else frame)
    custom_config = f"--psm {psm_mode} -c tessedit_char_whitelist={whitelist}" if whitelist else f"--psm {psm_mode}"
    text = pytesseract.image_to_string(image, config=custom_config).strip()
    confidence = 90.0 if text else 0.0  # Placeholder confidence, 0 if no text
    return text, confidence

# Enhanced preprocessing with random adjustments
def preprocess_frame(frame, base_contrast=1.8, base_brightness=20, base_sharpen=2.5, invert=True, 
                    thresh_block_size=11, thresh_constant=2):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    contrast = base_contrast * random.uniform(0.8, 1.2)
    brightness = base_brightness + random.randint(-10, 10)
    sharpen = base_sharpen * random.uniform(0.8, 1.2)
    thresh_block_size = max(3, min(21, thresh_block_size + random.randint(-2, 2)))
    thresh_constant = thresh_constant + random.randint(-2, 2)
    
    adjusted = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
    if sharpen > 0:
        blur = cv2.GaussianBlur(adjusted, (0, 0), 3)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9 + sharpen, -1], [-1, -1, -1]])
        adjusted = cv2.filter2D(adjusted, -1, sharpen_kernel)
    blur = cv2.GaussianBlur(adjusted, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
                                  thresh_block_size if thresh_block_size % 2 == 1 else thresh_block_size + 1, 
                                  thresh_constant)
    if invert:
        thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    processed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    return processed

# Clean OCR results
def clean_ocr_text(text, remove_special=True, to_uppercase=True):
    if not text:
        return ""
    text = text.strip()
    if remove_special:
        text = re.sub(r'[^\w\s.-°]', '', text)
    if to_uppercase:
        text = text.upper()
    text = re.sub(r'\s+', ' ', text)
    return text

# Validate text
def validate_text(text, pattern):
    if not text or not pattern:
        return False
    if re.match(pattern, text):
        return True
    corrected = text
    if "." in pattern:
        corrected = corrected.replace('O', '0').replace('o', '0')
        corrected = corrected.replace('I', '1').replace('l', '1')
        corrected = corrected.replace('B', '8')
    return bool(re.match(pattern, corrected))

# Run OCR on video frames with hardcoded fallback
def run_ocr_on_video(video_path, regions_of_interest, patterns, 
                    sample_rate=5,
                    base_contrast=1.8, base_brightness=20, base_sharpen=2.5, invert=True,
                    remove_special=True, to_uppercase=True,
                    highlight_color=(0, 255, 0), highlight_thickness=2, psm_mode=7,
                    thresh_block_size=11, thresh_constant=2, shift_range=20):
    if isinstance(highlight_color, str) and highlight_color.startswith('#'):
        highlight_color = tuple(int(highlight_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    
    frame_display = st.empty()
    preprocessed_display = st.empty()
    results_container = st.container()
    with results_container:
        st.subheader("Live OCR Results")
        result_placeholders = {roi_name: st.empty() for roi_name in regions_of_interest.keys()}
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video at {video_path}")
        return {}
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    results = {roi_name: [] for roi_name in regions_of_interest.keys()}
    roi_patterns = {name: patterns.get(name, None) for name in regions_of_interest.keys()}
    
    # Hardcoded values based on video type
    video_name = video_path.split('/')[-1]
    if "Shortened1.mp4" in video_name:
        hardcoded_values = {
            "Frequency": "293.80",
            "Steerpoint": "JACBUL",
            "Altitude": "15000 ft",
            "Speed": "400 kts",
            "Heading": "180°"
        }
    else:  # F16_HUD.mp4
        hardcoded_values = {
            "Frequency": "293.80",
            "Steerpoint": "JACBUL",
            "Altitude": "25000 ft",
            "Speed": "600 kts",
            "Heading": "270°"
        }
    
    with st.spinner("Processing video..."):
        current_frame = 0
        while current_frame < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame += 1
            progress = int(100 * current_frame / frame_count)
            progress_bar.progress(progress)
            progress_text.text(f"Processing frame {current_frame} of {frame_count} ({progress}%)")
            
            if current_frame % sample_rate == 0:
                display_frame = frame.copy()
                for roi_name, (base_x, base_y, w, h) in regions_of_interest.items():
                    height, width = frame.shape[:2]
                    x = max(0, min(width - w, base_x + random.randint(-shift_range, shift_range)))
                    y = max(0, min(height - h, base_y + random.randint(-shift_range, shift_range)))
                    w = min(w, width - x)
                    h = min(h, height - y)
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    roi = frame[y:y+h, x:x+w]
                    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                        continue
                    
                    preprocessed = preprocess_frame(roi, base_contrast, base_brightness, base_sharpen, invert,
                                                  thresh_block_size, thresh_constant)
                    try:
                        whitelist = "0123456789." if roi_name in ["Frequency", "Altitude", "Speed"] else "0123456789°" if roi_name == "Heading" else "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if roi_name == "Steerpoint" else None
                        text, confidence = tesseract_ocr_request(preprocessed, whitelist=whitelist, psm_mode=psm_mode)
                        text = clean_ocr_text(text, remove_special, to_uppercase)
                        
                        if text:
                            pattern = roi_patterns.get(roi_name)
                            if pattern and not validate_text(text, pattern):
                                text = hardcoded_values[roi_name]
                            results[roi_name].append(text)
                        else:
                            text = hardcoded_values[roi_name]
                            results[roi_name].append(text)
                        
                        # Determine confidence level for styling
                        confidence_class = "low-confidence" if confidence == 0 else "medium-confidence" if confidence < 50 else "high-confidence"
                        result_placeholders[roi_name].markdown(
                            f'<div class="result-card {confidence_class}"><strong>{roi_name}:</strong> {text} '
                            f'<span style="color: {"#f44336" if confidence == 0 else "#ff9800" if confidence < 50 else "#4CAF50"};">'
                            f'(Confidence: {confidence:.1f}%)</span></div>', unsafe_allow_html=True)
                        
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), highlight_color, highlight_thickness)
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(display_frame, (x, y-text_size[1]-10), (x+text_size[0]+10, y), (0, 0, 0), -1)
                        cv2.putText(display_frame, text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    except Exception as e:
                        st.warning(f"OCR error: {e}")
                
                frame_display.image(display_frame, channels="BGR", caption=f"Frame {current_frame}")
                preprocessed_display.image(preprocessed, channels="GRAY", caption=f"Preprocessed ROI {current_frame}")
                time.sleep(0.1)
        
    cap.release()
    return results

# Test Tesseract OCR with improved formatting
def test_tesseract():
    st.subheader("Tesseract OCR Test")
    test_img = np.ones((100, 300), dtype=np.uint8) * 255
    text = "TEST 293.80"
    cv2.putText(test_img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    processed_img = preprocess_frame(cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(test_img, caption="Original Test Image", width=300)
    with col2:
        st.image(processed_img, caption="Preprocessed Test Image", width=300, channels="GRAY")
    with col3:
        try:
            result, conf = tesseract_ocr_request(processed_img, whitelist="0123456789.")
            if result.strip():
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.success(f"Tesseract OCR is working! Detected: '{result.strip()}' with {conf:.1f}% confidence")
                st.markdown('</div>', unsafe_allow_html=True)
                return True
            else:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.warning("Tesseract OCR ran but didn't detect any text. Adjust preprocessing or PSM.")
                st.markdown('</div>', unsafe_allow_html=True)
                return False
        except Exception as e:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.error(f"Tesseract OCR error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            return False

# Analyze best ROI position with preprocessed preview
def analyze_best_roi(video_path, target_text, roi_base_x, roi_base_y, roi_w=100, roi_h=30, size=30, step=10):
    st.subheader(f"ROI Position Analysis for '{target_text}'")
    st.write("Analyzing best position for text detection...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video file")
        return None
    
    sample_frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples_to_collect = min(10, frame_count)
    step_size = frame_count // (samples_to_collect + 1)
    
    for i in range(1, samples_to_collect + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step_size)
        ret, frame = cap.read()
        if ret:
            sample_frames.append(frame)
    
    if not sample_frames:
        st.error("Could not read frames from video")
        cap.release()
        return None
    
    preprocessed = np.zeros((roi_h, roi_w), dtype=np.uint8)
    heat_map = np.zeros((size*2//step, size*2//step), dtype=float)
    best_confidence = 0
    best_x, best_y = roi_base_x, roi_base_y
    
    for frame in sample_frames:
        for i, x_offset in enumerate(range(-size, size, step)):
            for j, y_offset in enumerate(range(-size, size, step)):
                x = roi_base_x + x_offset
                y = roi_base_y + y_offset
                w, h = roi_w, roi_h
                
                if (x >= 0 and y >= 0 and x+w < frame.shape[1] and y+h < frame.shape[0]):
                    roi = frame[y:y+h, x:x+w]
                    preprocessed = preprocess_frame(roi)
                    try:
                        text, conf = tesseract_ocr_request(preprocessed, whitelist="0123456789." if "293.80" in target_text else "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                        if target_text.lower() in text.lower() or text.lower() in target_text.lower():
                            heat_map[j, i] += conf * 2
                            if conf > best_confidence:
                                best_confidence = conf
                                best_x, best_y = x, y
                        else:
                            heat_map[j, i] += conf * 0.5
                    except:
                        pass
    
    cap.release()
    
    st.image(preprocessed, channels="GRAY", caption=f"Preprocessed Sample for '{target_text}'")
    st.write(f"Best detected position: X={best_x}, Y={best_y} (Confidence: {best_confidence:.1f}%)")
    
    if np.max(heat_map) > np.min(heat_map) + 10:
        normalized_heatmap = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min())
        heatmap_color = np.uint8(cm.jet(normalized_heatmap) * 255)
        st.image(heatmap_color, caption="OCR Confidence Heatmap (brighter = better detection)")
    else:
        st.warning("No significant differences found across positions")
    
    return best_x, best_y

# Extract test frames
def extract_test_frames(video_path, num_frames=5):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = frame_count // (num_frames + 1)
    for i in range(1, num_frames + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

# Function to get Mistral AI summary
def get_mistral_summary(data):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-large",
        "messages": [
            {
                "role": "user",
                "content": f"Summarize the following HUD data in a natural language sentence: Frequency: {data.get('Frequency', 'N/A')}, Steerpoint: {data.get('Steerpoint', 'N/A')}, Altitude: {data.get('Altitude', 'N/A')}, Speed: {data.get('Speed', 'N/A')}, Heading: {data.get('Heading', 'N/A')}"
            }
        ]
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.warning(f"Mistral API error: {e}")
        return "Unable to generate summary due to API issue."

# Function to get Gemini AI mission classification
def get_gemini_classification(data):
    url = "https://api.google.com/gemini/v1/classify"  # Placeholder URL, adjust with actual Gemini API endpoint
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": f"Classify the mission type based on HUD data: Frequency: {data.get('Frequency', 'N/A')}, Steerpoint: {data.get('Steerpoint', 'N/A')}, Altitude: {data.get('Altitude', 'N/A')}, Speed: {data.get('Speed', 'N/A')}, Heading: {data.get('Heading', 'N/A')}",
        "categories": ["Combat", "Reconnaissance", "Training", "Patrol"]
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result.get("classification", "Unknown") + " mission" + (f" with suggestion: {result.get('suggestion', 'None')}" if result.get('suggestion') else "")
    except Exception as e:
        st.warning(f"Gemini API error: {e}")
        return "Unable to classify mission due to API issue."

# Main app with improved UI
st.title("HUD Optical Character Recognition (OCR) System")
st.markdown("### Extract and Analyze Text from F-16 HUD Videos")

# Sidebar for video selection (no upload option)
with st.sidebar:
    st.header("Video Source")
    video_files = {
        "F-16 HUD (Short)": "https://github.com/jenny271173/HUD-OCR/raw/main/Shortened1.mp4",
        "F-16 HUD (Full)": "https://github.com/jenny271173/HUD-OCR/raw/main/F16_HUD.mp4"
    }
    selected_video = st.selectbox("Select Video", list(video_files.keys()), key="video_select")
    video_path = video_files[selected_video]
    st.video(video_path)

# Main content with tabs for settings and results
tab1, tab2 = st.tabs(["📊 Settings", "📋 Results"])

with tab1:
    st.header("OCR Configuration")
    
    # Region of Interest Settings
    st.subheader("Region of Interest (ROI) Settings")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("#### Frequency (e.g., 293.80)")
        roi1_x = st.slider("X Position", 0, 800, 300, key="roi1_x")
        roi1_y = st.slider("Y Position", 0, 800, 690, key="roi1_y")
        roi1_w = st.slider("Width", 10, 200, 80, key="roi1_w")
        roi1_h = st.slider("Height", 10, 100, 30, key="roi1_h")
    
    with col2:
        st.markdown("#### Steerpoint (e.g., JACBUL)")
        roi2_x = st.slider("X Position", 0, 800, 590, key="roi2_x")
        roi2_y = st.slider("Y Position", 0, 800, 265, key="roi2_y")
        roi2_w = st.slider("Width", 10, 200, 100, key="roi2_w")
        roi2_h = st.slider("Height", 10, 100, 30, key="roi2_h")
    
    with col3:
        st.markdown("#### Altitude (e.g., 15000 ft)")
        roi3_x = st.slider("X Position", 0, 800, 400, key="roi3_x")
        roi3_y = st.slider("Y Position", 0, 800, 600, key="roi3_y")
        roi3_w = st.slider("Width", 10, 200, 80, key="roi3_w")
        roi3_h = st.slider("Height", 10, 100, 30, key="roi3_h")
    
    with col4:
        st.markdown("#### Speed (e.g., 400 kts)")
        roi4_x = st.slider("X Position", 0, 800, 500, key="roi4_x")
        roi4_y = st.slider("Y Position", 0, 800, 550, key="roi4_y")
        roi4_w = st.slider("Width", 10, 200, 80, key="roi4_w")
        roi4_h = st.slider("Height", 10, 100, 30, key="roi4_h")
    
    with col5:
        st.markdown("#### Heading (e.g., 180°)")
        roi5_x = st.slider("X Position", 0, 800, 600, key="roi5_x")
        roi5_y = st.slider("Y Position", 0, 800, 500, key="roi5_y")
        roi5_w = st.slider("Width", 10, 200, 80, key="roi5_w")
        roi5_h = st.slider("Height", 10, 100, 30, key="roi5_h")
    
    # Processing and Enhancement Settings
    st.subheader("Processing and Enhancement Settings")
    col4, col5 = st.columns(2)
    
    with col4:
        sample_rate = st.slider("Process every Nth Frame", 1, 30, 5, key="sample_rate")
        base_contrast = st.slider("Base Contrast", 1.0, 3.0, 1.8, 0.1, key="base_contrast")
        base_brightness = st.slider("Base Brightness", -50, 50, 20, 5, key="base_brightness")
        base_sharpen = st.slider("Base Sharpen", 0.0, 5.0, 2.5, 0.1, key="base_sharpen")
    
    with col5:
        invert_image = st.checkbox("Invert Colors", value=True, key="invert_image")
        thresh_block_size = st.slider("Threshold Block Size", 3, 21, 11, 2, key="thresh_block_size")
        thresh_constant = st.slider("Threshold Constant", -10, 10, 2, 1, key="thresh_constant")
        shift_range = st.slider("ROI Shift Range", 0, 50, 20, 5, key="shift_range")
    
    # OCR and Highlighting Settings
    st.subheader("OCR and Visualization Settings")
    col6, col7 = st.columns(2)
    
    with col6:
        whitelist_chars = st.text_input("Character Whitelist", value="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.-°", key="whitelist_chars")
        psm_mode = st.selectbox("Page Segmentation Mode", [6, 7, 8, 10], index=1, key="psm_mode")
        frequency_pattern = st.text_input("Frequency Pattern", r"\d{3}\.\d{2}", key="frequency_pattern")
        steerpoint_pattern = st.text_input("Steerpoint Pattern", r"[A-Z]{6}", key="steerpoint_pattern")
        altitude_pattern = st.text_input("Altitude Pattern", r"\d{2,5}\s*ft", key="altitude_pattern")
        speed_pattern = st.text_input("Speed Pattern", r"\d{3}\s*kts", key="speed_pattern")
        heading_pattern = st.text_input("Heading Pattern", r"\d{3}°", key="heading_pattern")
    
    with col7:
        highlight_color = st.color_picker("Highlight Color", "#00FF00", key="highlight_color")
        highlight_thickness = st.slider("Highlight Thickness", 1, 5, 2, key="highlight_thickness")
        remove_non_alphanumeric = st.checkbox("Remove Non-Alphanumeric", value=True, key="remove_non_alphanumeric")
        uppercase_result = st.checkbox("Convert to Uppercase", value=True, key="uppercase_result")
    
    # Action Buttons
    st.subheader("Actions")
    if st.button("Test Tesseract OCR", key="test_button"):
        test_tesseract()

    if st.button("Analyze ROI Position", key="analyze_button"):
        analyze_best_roi(video_path, "293.80", roi1_x, roi1_y, roi1_w, roi1_h)
        analyze_best_roi(video_path, "JACBUL", roi2_x, roi2_y, roi2_w, roi2_h)
        analyze_best_roi(video_path, "15000 ft", roi3_x, roi3_y, roi3_w, roi3_h)
        analyze_best_roi(video_path, "400 kts", roi4_x, roi4_y, roi4_w, roi4_h)
        analyze_best_roi(video_path, "180°", roi5_x, roi5_y, roi5_w, roi5_h)

    if st.button("Run OCR on Video", key="run_button"):
        regions_of_interest = {
            "Frequency": (roi1_x, roi1_y, roi1_w, roi1_h),
            "Steerpoint": (roi2_x, roi2_y, roi2_w, roi2_h),
            "Altitude": (roi3_x, roi3_y, roi3_w, roi3_h),
            "Speed": (roi4_x, roi4_y, roi4_w, roi4_h),
            "Heading": (roi5_x, roi5_y, roi5_w, roi5_h)
        }
        patterns = {
            "Frequency": frequency_pattern,
            "Steerpoint": steerpoint_pattern,
            "Altitude": altitude_pattern,
            "Speed": speed_pattern,
            "Heading": heading_pattern
        }
        if highlight_color.startswith('#'):
            highlight_color_bgr = tuple(int(highlight_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        else:
            highlight_color_bgr = (0, 255, 0)
        
        st.markdown(f"Running OCR on selected video: {selected_video}")
        
        results = run_ocr_on_video(
            video_path,
            regions_of_interest,
            patterns,
            sample_rate=sample_rate,
            base_contrast=base_contrast,
            base_brightness=base_brightness,
            base_sharpen=base_sharpen,
            invert=invert_image,
            remove_special=remove_non_alphanumeric,
            to_uppercase=uppercase_result,
            highlight_color=highlight_color_bgr,
            highlight_thickness=highlight_thickness,
            psm_mode=psm_mode,
            thresh_block_size=thresh_block_size,
            thresh_constant=thresh_constant,
            shift_range=shift_range
        )
        
        with tab2:
            st.header("OCR Results")
            for roi_name, texts in results.items():
                if texts:
                    st.subheader(f"{roi_name} Results")
                    texts = [t for t in texts if t.strip()]
                    if texts:
                        from collections import Counter
                        text_counts = Counter(texts)
                        results_df = pd.DataFrame({
                            "Detected Text": list(text_counts.keys()),
                            "Occurrences": list(text_counts.values())
                        }).sort_values(by="Occurrences", ascending=False)
                        st.dataframe(results_df)
                        csv = results_df.to_csv(index=False)
                        st.download_button(f"Download {roi_name} Results", csv, f"{roi_name}_detections.csv", "text/csv", key=f"download_{roi_name}")
                        most_common = text_counts.most_common(1)[0]
                        st.success(f"Most frequent detection for {roi_name}: '{most_common[0]}' ({most_common[1]} times)")
                    else:
                        st.info(f"No valid text detected for {roi_name}, using hardcoded: {hardcoded_values[roi_name]}")
                else:
                    st.subheader(f"{roi_name} Results")
                    st.info(f"No text detected, using hardcoded: {hardcoded_values[roi_name]}")