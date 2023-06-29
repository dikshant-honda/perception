import cv2
import numpy as np

class perception_lane_info:
    def __init__(self, camera_matrix, dist_coeffs) -> None:
        # Load the camera matrix and distortion coefficients obtained from calibration
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        # Load the image
        self.image = cv2.imread("/home/dikshant/catkin_ws/src/intgeration_module/src/integration/data/Leeds/Background.bmp")

    def get_lanes(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    def convert_2D_to_3D(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # detect the corner points   -- 2D --> 3D
        ret, corners = cv2.findChessboardCorners(gray, (4, 3), None)

        pattern_points = np.array([
            [0, 0, 0],  # top-left
            [1, 0, 0],  # top-right
            [1, 1, 0],  # bottom-right
            [0, 1, 0],  # bottom-left
        ], dtype=np.float32)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            undistorted_corners = cv2.undistortPoints(corners, self.camera_matrix, self.dist_coeffs)

            # Perform triangulation
            rvec, tvec = np.zeros((3, 1), dtype=np.float32), np.zeros((3, 1), dtype=np.float32)
            _, rvec, tvec = cv2.solvePnP(pattern_points, undistorted_corners, self.camera_matrix, self.dist_coeffs, rvec, tvec)
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            return rotation_matrix, tvec

        else:
            print("Chessboard corners not found in the image.")
            return _, _
        
    def lane_coordinates(self):
        road_edges = self.get_lanes(self.image)

        rotation_matrix, tvec = self.convert_2D_to_3D(self.image)
        # Compute the 3D position of a point in the real world given its image coordinates
        lane_coords = []
        for x,y in road_edges:
            image_point = (x, y)
            image_point_homogeneous = np.array([[image_point[0]], [image_point[1]], [1]], dtype=np.float32)
            inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
            inverse_camera_matrix = np.linalg.inv(self.camera_matrix)
            world_point_homogeneous = np.matmul(np.matmul(inverse_rotation_matrix, inverse_camera_matrix), image_point_homogeneous)
            world_point = world_point_homogeneous[:3].flatten() * tvec[2]
            lane_coords.append(world_point)

        return lane_coords
    
    def get_turning_route(self, lane):
        # obtain from pre-processing
        next_lane = None
        # next_lane = lane -> [left_lane, straight_lane, right_lane]
        return next_lane
    
    def stack_lanes(self, prev_lane, next_lane):
        if len(next_lane) == 0:
            return prev_lane
        prev_arr_x, next_arr_x = [], []
        prev_arr_y, next_arr_y = [], []
        for x, y in prev_lane:
            prev_arr_x.append(x)
            prev_arr_y.append(y)
        for x, y in next_lane:
            next_arr_x.append(x)
            next_arr_y.append(y)
        lane_x = np.hstack((prev_arr_x, next_arr_x))
        lane_y = np.hstack((prev_arr_y, next_arr_y))
        return [lane_x, lane_y]

    
if __name__ == "__main__":
    perception_lane_info()