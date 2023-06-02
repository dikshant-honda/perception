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

# -------------------------------- Region of Interest generation --------------------------------------------------
def putROI(image, roiCoord):
    [sX, sY], [eX, eY] = roiCoord
    if len(image.shape) > 2:
        roi = image[sY:eY, sX:eX,:]
    else:
        roi = image[sY:eY, sX:eX]
    return roi

# ------------------------------------- Generating Masks ----------------------------------------------------------
def drawROIMask(image, coords, Save=None):
    [sX, sY], [eX, eY] = coords
    mask = np.zeros_like(image)
    mask[sY:eY, sX:eX, :] = 255
    if Save: cv2.imwrite(Save, mask)
    return mask

def drawBEVMask(image, coords, Save=None):
    mask = np.zeros_like(image)
    C = np.array([[coords[0], coords[1], coords[3], coords[2]]])
    cv2.fillPoly(mask, C, (255,255,255))
    if Save: cv2.imwrite(Save, mask)
    return mask[:,:,0]


if __name__ == "__main__":
    # calibration parameters
    ROI_coords = [[0, 0], [1920, 1080]]
    BEV_coords = [[17, -225], [1260, 422], [405, 899], [596, 949]]

    # original image
    img = cv2.imread("/home/dikshant/3D-Net-Monocular-3D-Object-Recognition-for-Traffic-Monitoring/code/tests/ua_detrac_background/MVI_40141.mp4.jpg")
    cv2.imshow("camera view", img)

    # preprocessing
    roi = putROI(img, ROI_coords)
    cv2.imshow("ROI", roi)
    bev = birds_eye(img, BEV_coords, [1048, 1028]).bird
    cv2.imshow("BEV", bev)
    roi_mask = drawROIMask(roi, ROI_coords)
    cv2.imshow("ROI mask", roi_mask)
    bev_mask = drawBEVMask(bev, BEV_coords)
    cv2.imshow("BEV mask", bev_mask)
    cv2.waitKey(10000)