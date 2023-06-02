import os
import cv2
import numpy as np

# --------------------------------------- Birds eye view conversion ----------------------------------------------
class birds_eye:
    def __init__(self, image, cordinates, size=None):
        self.original = image.copy()
        self.image =  image
        self.c, self.r = image.shape[0:2]
        if size:self.bc, self.br = size
        else:self.bc, self.br = self.c, self.r
        pst2 = np.float32(cordinates)
        pst1 = np.float32([[0,0], [self.r,0], [0,self.c], [self.r,self.c]])
        self.transferI2B = cv2.getPerspectiveTransform(pst1, pst2)
        self.transferB2I = cv2.getPerspectiveTransform(pst2, pst1)
        self.bird = self.img2bird()

    def img2bird(self):
        self.bird = cv2.warpPerspective(self.image, self.transferI2B, (self.br, self.bc))
        return self.bird
    
    def bird2img(self):
        self.image = cv2.warpPerspective(self.bird, self.transferB2I, (self.r, self.c))
        return self.image
    
    def setImage(self, img):
        self.image = img

    def setBird(self, bird):
        self.bird = bird

    def convert2Bird(self, img):
        return cv2.warpPerspective(img, self.transferI2B, (self.bird.shape[1], self.bird.shape[0]))
    
    def convert2Image(self, bird):
        return cv2.warpPerspective(bird, self.transferB2I, (self.image.shape[1], self.image.shape[0]))
    
    def projection_on_bird(self, p, float_type=False):
        M = self.transferI2B
        px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        if float_type: return px, py
        return int(px), int(py)
    
    def projection_on_image(self, p, float_type=False):
        M = self.transferB2I
        px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        if float_type: return px, py
        return int(px), int(py)
    
def project(M, p):
    px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
    py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
    return int(px), int(py)


if __name__ == "__main__":
    img = cv2.imread("/home/dikshant/3D-Net-Monocular-3D-Object-Recognition-for-Traffic-Monitoring/code/tests/ua_detrac_background/MVI_40141.mp4.jpg")
    cv2.imshow("camera view", img)
    bev = birds_eye(img, [[17, -225], [1260, 422], [405, 899], [596, 949]], [1048, 1028])
    bird = bev.bird
    cv2.imshow("BEV", bird)
    
    cv2.waitKey(0)