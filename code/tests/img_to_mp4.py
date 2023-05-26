import os
from PIL import Image
import cv2
import numpy as np

def create_video(image_folder, video_name):
    images = [Image.open(f"{image_folder}/{img}") for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]

    # Get the dimensions of the first image
    width, height = images[0].size

    # Define the video codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Create a video writer
    video = cv2.VideoWriter(video_name, fourcc, 25.0, (width, height))

    # Iterate over the images and write each frame to the video
    for image in images:
        # Convert the PIL image to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Write the frame to the video
        video.write(frame)

    # Release the video writer
    video.release()

# Specify the image folder and video name
image_folder = "/home/dikshant/3D-Net-Monocular-3D-Object-Recognition-for-Traffic-Monitoring/data/Insight-MVT_Annotation_Train/MVI_20011"
video_name = "output.mp4"

# Call the function to create the video
create_video(image_folder, video_name)
