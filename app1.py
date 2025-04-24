import streamlit as st
import cv2
import numpy as np
import requests
import json
import base64
import re
import time
from io import BytesIO
from PIL import Image
import uuid

# Mistral OCR API configuration
MISTRAL_API_KEY = "JZslDyH2l2tQHDksRbDfHYVKgO50LfkN"  # Your API key
API_ENDPOINT = "https://api.mistral.ai/v1/ocr"

# Function to convert frame to base64 data URI
def frame_to_data_uri(frame):
    _, buffer = cv2.imencode('.png', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# Function to send frame to Mistral OCR API
def mistral_ocr_request(frame):
    unique_id = str(uuid.uuid4())
    img_data_uri = frame_to_data_uri(frame)
    
    payload = {
        "model": "mistral-ocr-latest",  # Using the same model as in your example
        "id": unique_id,
        "document": {
            "document_url": img_data_uri,
            "document_name": f"frame_{unique_id}.png",
            "type": "document_url"
        },
        "pages": [0],  # Single frame, so page 0
        "include_image_base64": True,
        "image_limit": 0,
        "image_min_size": 0
    }
    
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Mistral OCR API: {str(e)}")
        if 'response' in locals() and hasattr(response, 'text'):
            st.error(f"Response: {response.text}")
        return None

# Function to run OCR on video frames using Mistral API
def run_mistral_ocr(filename, ROI, num_frames):
    cap = cv2.VideoCapture(filename)
    numbers_array = []
    frame_count = 0
    
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return numbers_array
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % num_frames == 0:
            for ROI_name, (x, y, w, h) in ROI.items():
                # Extract ROI
                roi = frame[y:y+h, x:x+w]
                
                # Send to Mistral OCR API
                ocr_result = mistral_ocr_request(roi)
                
                if ocr_result and 'pages' in ocr_result:
                    # Assuming the API returns text in 'pages[0].text'
                    text = ocr_result['pages'][0].get('text', '')
                    numbers = re.findall(r'\d+', text)
                    
                    if text:
                        st.write(f"{ROI_name} Text: {text}")
                    
                    if numbers:
                        numbers_array.append(numbers)
                    
                    # Draw rectangle and text on frame for visualization
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, ','.join(numbers), 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 0, 255), 2)
                
                # Display frame in Streamlit
                st.image(frame, channels="BGR", caption=f"Frame {frame_count}")
        
        # Optional: Break early for testing
        if frame_count >= 100:  # Limit to 100 frames for demo
            break
    
    cap.release()
    return numbers_array

# Streamlit app implementation
st.title("Head Up Display (HUD) Optical Character Recognition (OCR) Model")
st.markdown('Jennifer Doan')
st.image('https://github.com/jenny271173/HUD-OCR/raw/main/f16_display_image.jpeg', 
         caption='U.S. Air Force F-16 Fighter Jet')
st.markdown('Image extracted from: https://goallclear.com/how-planned-upgrades-will-keep-the-f-16-flying-into-the-future/')
st.markdown('**Bottom line up front**: Click the buttons below to run the HUD OCR with Mistral OCR!')

# Implementation section
def run_code():
    st.write("Loading libraries, packages, and data...")
    video = "/Users/jenniferdoan/Desktop/Shortened.mp4"  # Update this path as needed
    return video

def run_code1(video_path):
    st.write("OCR Action in Progress with Mistral OCR...")
    
    # Define ROIs
    ROI = {
        'ROI 1': (525, 540, 95, 35)
    }
    ROI_2 = {
        'ROI 2': (630, 540, 95, 35)
    }
    
    # Run OCR for both ROIs
    result = run_mistral_ocr(video_path, ROI, num_frames=10)
    result_2 = run_mistral_ocr(video_path, ROI_2, num_frames=10)
    
    # Display results
    st.write("Result for ROI 1:")
    for sublist in result:
        st.write('.'.join(sublist))

    st.write("Result for ROI 2:")
    for sublist in result_2:
        for item in sublist:
            st.write(item)

# Buttons for execution
if st.button("Auto Import Data"):
    video_path = run_code()

if st.button("Run OCR"):
    video_path = "/Users/jenniferdoan/Desktop/Shortened.mp4"  # Hardcoded for now
    run_code1(video_path)

# Rest of your original code (headers, references, etc.) can remain unchanged
st.header('Understanding the Results')
st.markdown('Click the button below to review the post-model analytics.')
if st.button("View the output"):
    st.markdown('For those that were unable to entirely run the model, the abbreviated version of the output appears in the following manner:')
    # Your existing image display code here...

st.header('References')
references = "Atherton, K. (2022, May 6). Understanding the errors introduced by military AI applications. Brookings. https://www.brookings.edu/techstream/understanding-the-errors-introduced-by-military-ai-applications/ <br>[DontGetShot]. (2023, February 12). Michigan UFO Declassified F-16 HUD Footage [Video]. YouTube. https://www.youtube.com/watch?v=GZt-lordqBE&ab_channel=DontGetShot <br>Hamad, K. A., & Kaya, M. (2016). A detailed analysis of optical character recognition technology. International Journal of Applied Mathematics, Electronics and Computers, 244-249. https://doi.org/10.18100/ijamec.270374 <br>Wilson, N., Guragain, B., Verma, A., Archer, L., & Tavakolian, K. (2019). Blending human and machine: Feasibility of measuring fatigue through the aviation headset. Human Factors: The Journal of the Human Factors and Ergonomics Society, 62(4). https://doi.org/10.1177/0018720819849783"
st.markdown(references, unsafe_allow_html=True)