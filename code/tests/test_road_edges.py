import cv2
import numpy as np

image = cv2.imread("/home/dikshant/3D-Net-Monocular-3D-Object-Recognition-for-Traffic-Monitoring/code/tests/ua_detrac_background/MVI_40141.mp4.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (3, 3), 0)

edges = cv2.Canny(blurred, 150, 200, apertureSize = 3)  

height, width = edges.shape[:2]
roi_vertices = np.array([[(0, height), (width/2, 0), (width, height)]], dtype=np.int32)

mask = np.zeros_like(edges)

cv2.fillPoly(mask, roi_vertices, 255)

masked_edges = cv2.bitwise_and(edges, mask)

# Extract the road edges
road_edges = masked_edges

cv2.imshow("Road Edges", road_edges)
cv2.waitKey(0)
