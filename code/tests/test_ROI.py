import os
import cv2
import numpy as np

R = 114
ESC_KEY = 27
SPACE = 32

# ---------------------------------------- Road Edges Extraction -------------------------------------------------
def getRoadEdge(img, a, b, Save=None):
    edges = cv2.Canny(img,a,b,apertureSize = 3)
    if Save:cv2.imwrite(Save, edges)
    return edges

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

# ------------------------------------- Region of Interest Determination -------------------------------------

def applyROI(coord, roiCoord, reverse=False):
    x1, y1, x2, y2 = coord
    [sX, sY], [eX, eY] = roiCoord
    if reverse:
        return x1 + sX, y1 + sY, x2 + sX, y2 + sY
    else:
        return x1 - sX, y1 - sY, x2 - sX, y2 - sY

def applyROIxy(coord, roiCoord, reverse=False):
    x, y = coord
    [sX, sY], [eX, eY] = roiCoord
    if reverse:
        return x + sX, y + sY
    else:
        return x - sX, y - sY

def putROI(image, roiCoord):
    [sX, sY], [eX, eY] = roiCoord
    if len(image.shape) > 2:
        roi = image[sY:eY, sX:eX,:]
    else:
        roi = image[sY:eY, sX:eX]
    return roi

def getMask(image, coords, Save=None):
    [sX, sY], [eX, eY] = coords
    mask = np.zeros_like(image)
    mask[sY:eY, sX:eX, :] = 255
    if Save: cv2.imwrite(Save, mask)
    return mask

def ShowROI(image, roiCoord):
    mask = getMask(image, roiCoord)
    [sX, sY], [eX, eY] = roiCoord
    Show = cv2.addWeighted(image, 0.5, np.where(mask > 1, image, 0), 1 - 0.5, 0)
    cv2.rectangle(Show,(sX, sY),(eX, eY), (255,255,255),2)
    cv2.putText(Show, 'Region Of Interest', (sX, sY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return Show


if __name__ == "__main__":
    background = cv2.imread("background.jpg")
    # cv2.imshow("background", background)
    rois = putROI(background, [[0, 0], [1920, 1080]])
    show = ShowROI(background, [[0, 0], [1920, 1080]])
    cv2.imshow("test", show)
    cv2.waitKey(10000)