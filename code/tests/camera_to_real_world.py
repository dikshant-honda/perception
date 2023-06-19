import cv2
import numpy as np

# Load the camera matrix and distortion coefficients obtained from calibration
camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')

pattern_points = np.array([
    [0, 0, 0],  # top-left
    [1, 0, 0],  # top-right
    [1, 1, 0],  # bottom-right
    [0, 1, 0],  # bottom-left
], dtype=np.float32)

# Load the image and detect the corner points
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (4, 3), None)

if ret:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Refine the corner positions
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Undistort the image points
    undistorted_corners = cv2.undistortPoints(corners, camera_matrix, dist_coeffs)

    # Perform triangulation
    rvec, tvec = np.zeros((3, 1), dtype=np.float32), np.zeros((3, 1), dtype=np.float32)
    _, rvec, tvec = cv2.solvePnP(pattern_points, undistorted_corners, camera_matrix, dist_coeffs, rvec, tvec)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Compute the 3D position of a point in the real world given its image coordinates
    image_point = (100, 200)  # Example image coordinates
    image_point_homogeneous = np.array([[image_point[0]], [image_point[1]], [1]], dtype=np.float32)
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    inverse_camera_matrix = np.linalg.inv(camera_matrix)
    world_point_homogeneous = np.matmul(np.matmul(inverse_rotation_matrix, inverse_camera_matrix), image_point_homogeneous)
    world_point = world_point_homogeneous[:3].flatten() * tvec[2]

    print("Real-world coordinates:", world_point)
else:
    print("Chessboard corners not found in the image.")
