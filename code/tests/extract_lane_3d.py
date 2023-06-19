import cv2
import numpy as np

def get_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 100)  

    height, width = edges.shape[:2]
    roi_vertices = np.array([[(0, height), (0, 0), (width, 0),(width, height)]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Extract the road edges
    road_edges = masked_edges

    return road_edges

def convert_2D_to_3D(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect the corner points   -- 2D --> 3D
    ret, corners = cv2.findChessboardCorners(gray, (4, 3), None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        undistorted_corners = cv2.undistortPoints(corners, camera_matrix, dist_coeffs)

        # Perform triangulation
        rvec, tvec = np.zeros((3, 1), dtype=np.float32), np.zeros((3, 1), dtype=np.float32)
        _, rvec, tvec = cv2.solvePnP(pattern_points, undistorted_corners, camera_matrix, dist_coeffs, rvec, tvec)
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        return rotation_matrix, tvec

    else:
        print("Chessboard corners not found in the image.")
        return _, _
    
if __name__ == "__main__":
    # Load the camera matrix and distortion coefficients obtained from calibration
    camera_matrix = np.load('camera_matrix.npy')
    dist_coeffs = np.load('dist_coeffs.npy')

    # Load the image
    image = cv2.imread("/home/dikshant/3D-Net-Monocular-3D-Object-Recognition-for-Traffic-Monitoring/code/tests/Road mask.png")

    pattern_points = np.array([
        [0, 0, 0],  # top-left
        [1, 0, 0],  # top-right
        [1, 1, 0],  # bottom-right
        [0, 1, 0],  # bottom-left
    ], dtype=np.float32)

    road_edges = get_lanes(image)

    rotation_matrix, tvec = convert_2D_to_3D(image)

    # Compute the 3D position of a point in the real world given its image coordinates
    image_point = (100, 200)  # Example image coordinates
    image_point_homogeneous = np.array([[image_point[0]], [image_point[1]], [1]], dtype=np.float32)
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    inverse_camera_matrix = np.linalg.inv(camera_matrix)
    world_point_homogeneous = np.matmul(np.matmul(inverse_rotation_matrix, inverse_camera_matrix), image_point_homogeneous)
    world_point = world_point_homogeneous[:3].flatten() * tvec[2]

    print("Real-world coordinates:", world_point)

