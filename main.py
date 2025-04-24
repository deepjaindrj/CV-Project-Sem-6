import image_filing
import detection
import front
from ultralytics import YOLO
# set directories for images and labels for YOLO structure

# rename images
image_directory = ""
image_filing.rename_images(rf"{image_directory}")

# rename labels
labels_directory = ""
image_filing.rename_labels(rf"{labels_directory}")

# import and train YOLO model
model = YOLO(detection.export_model())
epoch_runs = None  # set how many epochs for model training
detection.train_model(epoch_runs)  # train model for 100 epochs
# set detection evaluate_image and evaluate_video directory for the new weights

# in console, write: "streamlit run front.py"

# begin analyzing




