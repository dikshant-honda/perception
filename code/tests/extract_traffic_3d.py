import cv2
import numpy as np

class traffic_info:
    def __init__(self) -> None:
        # Load the camera matrix and distortion coefficients obtained from calibration
        self.camera_matrix = np.load('camera_matrix.npy')
        self.dist_coeffs = np.load('dist_coeffs.npy')

        # Load from the traffic video frame
        self.video_stream = cv2.VideoCapture("cam_recording.mp4")

        if not self.video_stream.isOpened():
            print("error in opening the video file")
            exit()

        self.traffic_coords()
        # end the stream
        self.video_stream.release()
        cv2.destroyAllWindows()

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
        
    def traffic_coords(self):
        while True:
            ret, frame = self.video_stream.read()
            # end of the video
            if not ret: break

            rotation_matrix, tvec = self.convert_2D_to_3D(frame)  # test later if both matrices are similar for lanes extraction and traffic participants info ectraction

            # add the traffic participants info from the perception system
            traffic_participants = {}                       # dict type: {'id': [position, velocity, orientation, type]}

            traffic_participants_3D = {}

            # Compute the 3D position of a point in the real world given its image coordinates
            for key, val in traffic_participants.items():
                image_point = val[0]
                image_point_homogeneous = np.array([[image_point[0]], [image_point[1]], [1]], dtype=np.float32)
                inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
                inverse_camera_matrix = np.linalg.inv(self.camera_matrix)
                world_point_homogeneous = np.matmul(np.matmul(inverse_rotation_matrix, inverse_camera_matrix), image_point_homogeneous)
                world_point = world_point_homogeneous[:3].flatten() * tvec[2]

                traffic_participants_3D[key] = [world_point, val[1], val[2], val[3]]

            return traffic_participants_3D
    
if __name__ == "__main__":
    traffic_info()