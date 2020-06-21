# Task1 Hint: (with sample code for the SIFT detector)
# Initialize SIFT detector, detect keypoints, store and show SIFT keypoints of original image in a Numpy array
# Define parameters for SIFT initializations such that we find only 10% of keypoints
import cv2
import matplotlib.pyplot as plt
import numpy as np
from template import *

def q1(gray):
    sift = SiftDetector()
    kp = sift.detector.detect(gray,None)
    print(len(kp))
    # gray1 = cv2.drawKeypoints(gray,kp,gray)
    params={}
    params["n_features"]=623
    params["n_octave_layers"]=3
    params["contrast_threshold"]=0.1
    params["edge_threshold"]=10
    params["sigma"]=1.6
    sift1 = SiftDetector(params=params)

    kp1 = sift1.detector.detect(gray,None)
    print(len(kp1))
    return sift1
#In order to reduce the keypoints, increase the threshould value to reduce the weak feafure, and set up the number of keypoints directly through the nfeature.
def q2(gray,sift1):
    scale =115
    scale=scale/100
    width = int(gray.shape[1] * scale)
    height = int(gray.shape[0] * scale)
    dim = (width, height)
    resized = cv2.resize(gray,dim)
    ####
    kp1,des1 = sift1.detector.detectAndCompute(gray,None)
    kp2,des2 = sift1.detector.detectAndCompute(resized,None)

    bf = cv2.BFMatcher()
# Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 5 matches.
    img3 = cv2.drawMatches(gray,kp1,resized,kp2,matches[:5],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    cv2.imwrite('d.jpg',img3)
### the keypoints in both graph are pretty similar. it indicates that they share the same common features.
def q3(gray,sift1):
    center = get_img_center(gray)
    img = rotate(gray,center[0],center[1],60)
    plt.imshow(img),plt.show()
    ###
    kp1,des1 = sift1.detector.detectAndCompute(gray,None)
    kp2,des2 = sift1.detector.detectAndCompute(img,None)

    bf = cv2.BFMatcher()
# # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 5 matches.
    img3 = cv2.drawMatches(gray,kp1,img,kp2,matches[:5],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    cv2.imwrite('d1.jpg',img3)

### the keypoints in both graph are pretty similar. it indicates that they share the same common features.
if __name__ == "__main__":
    
    img = cv2.imread('pic.jpg')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift1 = q1(gray)
    # q2(gray,sift1)
    q3(gray,sift1)
    cv2.waitKey(0)
