import os


#  renaming images to counter
def rename_images(image_dir_path):
    if os.path.exists(image_dir_path):
        files = os.listdir(image_dir_path)
        for index, file in enumerate(files):
            os.rename(os.path.join(image_dir_path, file), os.path.join(image_dir_path, f"{index}.jpg"))
    else:
        print(f"The path '{image_dir_path}' does not exist.")

def rename_labels(labels_dir_path):
    if os.path.exists(labels_dir_path):
        files = os.listdir(labels_dir_path)
        for index, file in enumerate(files):
            os.rename(os.path.join(labels_dir_path, file), os.path.join(labels_dir_path, f"{index}.txt"))
    else:
        print(f"The path '{labels_dir_path}' does not exist.")
