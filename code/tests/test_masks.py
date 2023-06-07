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

# ------------------------------------- Road Edge Masks ---------------------------------------------------------------
def road_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 140 , 250, apertureSize = 3)  
    height, width = edges.shape[:2]
    roi_vertices = np.array([[(0, height), (width/3, height/10), (2*width/3, height/10), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    road_edges = masked_edges

    return road_edges

if __name__ == "__main__":
    img = cv2.imread("/home/dikshant/3D-Net-Monocular-3D-Object-Recognition-for-Traffic-Monitoring/code/tests/Background.bmp")
    # img = cv2.imread("/home/dikshant/3D-Net-Monocular-3D-Object-Recognition-for-Traffic-Monitoring/code/tests/background_1.jpg")
    height, width = len(img), len(img[0])

    # calibration parameters
    ROI_coords = [[0, 0], [width, height]]
    BEV_coords = [[0, 0], [int(width),0], [int(width/2-100), int(height)], [int(width/2+100), int(height)]]
    # BEV_coords = [[int(-width/4), 0], [int(5*width/4), 0], [int(width/3), int(height)], [int(2*width/3), int(height)]]
    # BEV_coords = [[int(-width/2), 0], [int(3*width/2), 0], [int(width/4), int(height)], [int(3*width/4), int(height)]]

    # original image
    cv2.imshow("camera view", img)

    # preprocessing
    roi = putROI(img, ROI_coords)
    cv2.imshow("ROI", roi)
    bev = birds_eye(roi, BEV_coords, ).bird
    cv2.imshow("BEV", bev)
    cv2.imwrite("Bird Eye View.bmp", bev)
    roi_mask = drawROIMask(roi, ROI_coords, "ROI Mask.bmp")
    cv2.imshow("ROI mask", roi_mask)
    bev_mask = drawBEVMask(bev, BEV_coords, "ROI BEV Mask.bmp")
    cv2.imshow("BEV mask", bev_mask)
    road_border = road_edges(img)
    cv2.imshow("Road border in image frame", road_border)
    road_border = road_edges(bev)
    cv2.imshow("Road border in BEV frame", road_border)

    # combined view of camera view and BEV
    bev = cv2.resize(bev, (img.shape[1], img.shape[0]))
    comparison = np.hstack((img, bev))
    cv2.imshow("Camera View vs. Bird's-eye View", comparison)
    
    cv2.waitKey(0)