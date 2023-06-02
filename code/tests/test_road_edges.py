import cv2
import numpy as np

# Load the image
image = cv2.imread("/home/dikshant/3D-Net-Monocular-3D-Object-Recognition-for-Traffic-Monitoring/code/tests/ua_detrac_background/MVI_40141.mp4.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 70)  # Adjust the thresholds as needed

# Define the region of interest (ROI) vertices
height, width = edges.shape[:2]
roi_vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)

# Create a blank mask
mask = np.zeros_like(edges)

# Fill the ROI polygon with white color (255)
cv2.fillPoly(mask, roi_vertices, 255)

# Apply the mask to the edges image
masked_edges = cv2.bitwise_and(edges, mask)

# Extract the road edges
road_edges = masked_edges

# Display the road edges
cv2.imshow("Road Edges", road_edges)
cv2.waitKey(0)
