import os
import cv2
import numpy as np

ESC_KEY = 27

def checkReduction(image):
    reduction = 1
    if image.shape[0]>10000 or image.shape[1]>10000:
        reduction = 10
    elif image.shape[0]>5000 or image.shape[1]>5000:
        reduction = 5
    elif image.shape[0]>2000 or image.shape[1]>2000:
        reduction = 3
    elif image.shape[0]>1100 or image.shape[1]>1100:
        reduction = 2
    return reduction

def calcBackgound(VideoPath, reduc, Save=None):
    cap = cv2.VideoCapture(VideoPath)
    _, f = cap.read()
    f= cv2.resize(f, (f.shape[1]// reduc , f.shape[0] // reduc))
    img_bkgd = np.float32(f)
    reduc = checkReduction(img_bkgd)
    while True:
        ret, f = cap.read()
        if not ret: break
        cv2.imshow('Main Video', cv2.resize(f, (f.shape[1]// reduc , f.shape[0] // reduc)))
        cv2.accumulateWeighted(f, img_bkgd, 0.01)
        res2 = cv2.convertScaleAbs(img_bkgd)
        cv2.imshow('When you feel the background is good enough, press ESC to terminate and save the background.', cv2.resize(res2, (res2.shape[1]// reduc , res2.shape[0] // reduc)))
        k = cv2.waitKey(20)
        if k == 27: break
    if Save: cv2.imwrite(Save, res2)
    cv2.destroyAllWindows()
    cap.release()
    return res2

if __name__ == "__main__":
    videos_folder = "/home/dikshant/3D-Net-Monocular-3D-Object-Recognition-for-Traffic-Monitoring/code/tests/ua_detrac_videos/"
    background_folder = "/home/dikshant/3D-Net-Monocular-3D-Object-Recognition-for-Traffic-Monitoring/code/tests/ua_detrac_background/"
    for folder in os.listdir(videos_folder):
        video_path = videos_folder+folder
        background_path = background_folder+folder+".jpg"
        calcBackgound(video_path, 1, background_path)