from ultralytics import YOLO
import cv2
import os
import math
import tempfile

model = YOLO("yolov8s.yaml")


def train_model(epoch):  # training the model for epoch runs
    results = model.train(data="config.yaml", epochs=epoch)


def evaluate_image(image_source):  # analyzing image for aircraft
    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
    threshold = 0.0
    aircraft_counter = 0
    # Load a model
    model = YOLO(model_path)  # load a custom model
    results = model.predict(image_source)
    results_obj = results[0]
    # print("Results structure:", type(results), results)

    original_img = cv2.imread(image_source)
    for result in results_obj.boxes.data.tolist():  # returns the detections
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:  # choosing which detections to draw
            aircraft_counter += 1
            cv2.rectangle(original_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(original_img, model.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # saving the new image with detections
    detected_file = 'detected_' + os.path.basename(image_source)
    cv2.imwrite(detected_file, original_img)
    return detected_file, aircraft_counter


def evaluate_video(video_source):
    # Check if the video file exists
    if not os.path.exists(video_source):
        print(f"Error: Video file not found at {video_source}")
        exit()

    # Open the video capture
    cap = cv2.VideoCapture(video_source)
    frame_count = 0

    # Check if the video capture is successful
    if not cap.isOpened():
        print("Error: Unable to open video capture")
        sys.exit()

    ret, frame = cap.read()

    # Check if the frame is successfully read
    if frame is None:
        print("Error: Unable to read the first frame from the video")
        sys.exit()

    H, W, _ = frame.shape

    # Create a temporary directory to store the video file
    temp_dir = tempfile.mkdtemp()

    # Create a temporary file path for the video
    temp_video_path = os.path.join(temp_dir, 'detected_video.mp4')

    # Create a video writer using the temporary file
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'h264'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

    # Load a model
    model = YOLO(model_path)  # load a custom model

    threshold = 0.0
    aircraft_counter = 0

    while ret:
        # Your existing code
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                aircraft_counter += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        out.write(frame)
        ret, frame = cap.read()
        frame_count += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    aircrafts = math.ceil(aircraft_counter/frame_count)  # number of airplanes

    # Provide the temporary file path and clean up the temporary directory
    return temp_video_path, aircrafts

def export_model():
    return model

# train_model(100)
# export_model()
# aircrafts = evaluate_video()
# print("total aircrafts = ", aircrafts)