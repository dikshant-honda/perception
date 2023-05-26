import os
import numpy as np
from PIL import Image, ImageSequence
import cv2

def create_video(image_folder, video_name):
    images = [f"{image_folder}/{img}" for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]

    with Image.open(images[0]) as first_image:
        width, height = first_image.size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video = cv2.VideoWriter(video_name, fourcc, 24.0, (width, height))

    for image_path in images:
        with Image.open(image_path) as image:
            # Convert the PIL image to OpenCV format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            video.write(frame)

    video.release()


images_folder = "/home/dikshant/3D-Net-Monocular-3D-Object-Recognition-for-Traffic-Monitoring/data/Insight-MVT_Annotation_Train/"
videos_folder = "/home/dikshant/3D-Net-Monocular-3D-Object-Recognition-for-Traffic-Monitoring/code/tests/ua_detrac_videos/"

for folder in os.listdir(images_folder):
    create_video(images_folder+folder, videos_folder+folder+".mp4")