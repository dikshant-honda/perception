#!/usr/local/bin/python

import cv2
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

dir_path = '/home/dikshant/3D-Net-Monocular-3D-Object-Recognition-for-Traffic-Monitoring/data/Insight-MVT_Annotation_Train/MVI_20035'
ext = args['extension']
output = args['output']

images = []
for f in os.listdir(dir_path):
    if f.endswith(ext):
        images.append(f)

image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output, fourcc, 25, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) 

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))

# slowing down the video 

fileName="output.mp4" 
slomo_frame = int(input("Enter the frames you want to change to \n"))  
cap = cv2.VideoCapture(fileName)       # load the video
while(cap.isOpened()):                    # play the video by reading frame by frame
    ret, frame = cap.read()
    if ret==True:
        cv2.imshow('frame',frame)              # show the video
        if cv2.waitKey(slomo_frame) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()