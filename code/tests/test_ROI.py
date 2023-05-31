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

# ------------------------------------- Region of Interest Determination -------------------------------------

def getROI(image, Save=None):
    while True:
        roi, coords , roiImage = os.getROI('Select a Region of Interst for caliibration | Actions: Space = OK,  r = Retry |', image).run()
        zeroDim = False
        for i in roi.shape:
            if i ==0: zeroDim = True
        if zeroDim: continue
        cv2.imshow('Your Region of Interrest | Actions: Space = OK,  r = Retry |', roi)
        k = cv2.waitKey(0)
        if k%256 == R: cv2.destroyAllWindows(); continue
        elif k%256 == SPACE: cv2.destroyAllWindows(); break
    if Save: cv2.imwrite(Save, roiImage)
    return roi, coords

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