import datetime

import streamlit as st
from streamlit_image_comparison import image_comparison
import detection
import tempfile
from PIL import Image
import io
import entities.airport
import insights_db


def display_header():
    st.header('Imagery Analysis Tool 🔎 :airplane:', divider='blue')


def display_file_upload_ui(file_type):  # file upload ui for image and video
    uploaded_video, uploaded_image = None, None

    if file_type == ['Video']:
        uploaded_video = st.file_uploader("Upload Video", type='mp4')
    else:
        uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    return uploaded_video, uploaded_image


def get_file_type(file_source):  # returns the type of file
    delete_dot = file_source.rsplit('.',1)
    if len(delete_dot) > 1:
        file_extension = delete_dot[1].lower()
        # Check the file extension to determine the file type
        if file_extension in ['png', 'jpg', 'jpeg','mp4']:
            return file_extension
        else:
            return 'Unknown'
    else:
        return 'No Extension'


def video_bytes(video_path):  # read video in chunks
    with open(video_path, "rb") as f:
        video_data = b""
        while True:
            video_chunk = f.read(1024)
            if not video_chunk:
                break
            video_data += video_chunk
    return video_data


def download_content(airport, detected_content, vid_or_img, aircraft_count):  # displays download image button for analyzed image
    current_datetime = datetime.datetime.now()
    date_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    cleaned_airport = str(airport.name[0][0:-1].replace(' (', '_')) if airport else "No Airport Specified"  # clean airport name
    downloaded_name = cleaned_airport + "_" + date_string + "-" + str(aircraft_count)
    if vid_or_img == 'Image':
        image = Image.open(detected_content)

        # Convert the image to bytes
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")

        # Display the download button
        st.download_button(
            label="Download Detection",
            data=image_bytes.getvalue(),
            file_name=f"{downloaded_name}.png",
            key="download_detected_image",
        )
    else:
    # Display the download button for video
        st.download_button(
            label="Download Detection Video",
            data=video_bytes(detected_content),
            file_name=f"{downloaded_name}.mp4",
            key="download_detected_video",
        )


def enter_to_db(airport, aircraft_count):  # enters image analysis to db
    cleaned_airport = str(airport.name[0][0:-1].replace(' (', '_')) if airport else "No Airport Specified"  # clean airport name
    current_datetime = datetime.datetime.now()
    threshold = "Not Over"
    if aircraft_count > airport.threshold:  # check if aircraft count is over airport's threshold
        threshold = "Over"
    insights_db.add_imagery_data(cleaned_airport, aircraft_count, threshold, current_datetime)


def show_success(airport, aircraft_count, detected_content):  # show success, aircraft count & download button
    st.markdown(
        f"<div style='text-align: center; padding: 10px; border: 2px solid black; border-radius: 5px;'>"
        f"<h3>{aircraft_count} Aircrafts Detected</h3>"
        "</div>",
        unsafe_allow_html=True
    )
    st.success('Detection Completed', icon="✅")
    st.balloons()

    # enter analysis to db
    enter_to_db(airport, aircraft_count)

    # download the image
    download_content(airport, detected_content, vid_or_img, aircraft_count)


def begin_detection(content_type, image_content, vid_or_img):  # beginning detection algorithm
    # Create a temporary image file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{content_type}') as temp_image:
        # Write the image content to the temporary file
        temp_image.write(image_content)
        temp_content_path = temp_image.name
        if vid_or_img == 'Image':  # save to temporary file
            detected_img, aircraft_count = detection.evaluate_image(temp_content_path)
            image_comparison(temp_content_path, detected_img)
            success = True
        else:
            detected_vid, aircraft_count = detection.evaluate_video(temp_content_path)
            st.video(detected_vid)
            success = True

    if vid_or_img == 'Image':
        return success, detected_img, aircraft_count
    return success, detected_vid, aircraft_count


def analyze_button_clicked(vid_or_img, content):
    success = False
    if airport_name != "Airport Name":
        airport = entities.airport.Airport(airport_name, 40)
        airport_fill = True
    else:
        st.error("Select Airport")
    if content:
        content_type = get_file_type(content.name)
        image_content = content.read()
        # detection proccess
        success, detected_content, aircraft_count = begin_detection(content_type, image_content, vid_or_img)
    else:
        st.error("File wasn't uploaded")
    if success and airport_fill:  # upon successful detection
        show_success(airport, aircraft_count, detected_content)
    else:
        st.error("Detection Was Not Successful")
        st.snow()


# Set page configuration
st.set_page_config(page_title="Aircraft Intelligence Hub", layout="centered",
                   page_icon="images/Aircraft Intelligence Hub Badge.ico")

# Display header
display_header()

# File type selection
file_type = st.multiselect("Type of file for analysis ",
                           placeholder="Image Default",
                           options=['Image', 'Video'],
                           max_selections=1,
                           label_visibility="visible")

# Airport selection
airport_name = st.multiselect("Choose Airport for Analysis",
    placeholder="Airport Name",
    options=[
        "Amsterdam - Amsterdam Airport Schiphol (AMS)",
        "Atlanta - Hartsfield-Jackson International Airport (ATL)",
        "Bangkok - Suvarnabhumi Airport (BKK)",
        "Beijing - Capital International Airport (PEK)",
        "Chicago - O'Hare International Airport (ORD)",
        "Dallas/Fort Worth - Dallas/Fort Worth International Airport (DFW)",
        "Denver - Denver International Airport (DEN)",
        "Dubai - Dubai International Airport (DXB)",
        "Frankfurt - Frankfurt Airport (FRA)",
        "Guangzhou - Baiyun International Airport (CAN)",
        "Hong Kong - Hong Kong International Airport (HKG)",
        "Istanbul - Istanbul Airport (IST)",
        "London - Heathrow Airport (LHR)",
        "Los Angeles - Los Angeles International Airport (LAX)",
        "New York City - John F. Kennedy International Airport (JFK)",
        "Paris - Charles de Gaulle Airport (CDG)",
        "Seoul - Incheon International Airport (ICN)",
        "Shanghai - Pudong International Airport (PVG)",
        "Singapore - Changi Airport (SIN)",
        "Tokyo - Haneda Airport (HND)"
        ],
    max_selections=1,
    label_visibility="visible")

# File upload UI
uploaded_video, uploaded_image = display_file_upload_ui(file_type)


# Initialize relevant variables
# aircraft_counter = 0
vid_or_img = None
content = None


# Determine relevant source
if uploaded_image is not None:
    vid_or_img = 'Image'
    content = uploaded_image
elif uploaded_video is not None:
    vid_or_img = 'Video'
    content = uploaded_video

# Analyze button

analyze_button = st.button("Analyze", on_click=lambda: analyze_button_clicked(vid_or_img, content))
