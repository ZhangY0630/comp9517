import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
def q1(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('grey',gray)
    img1 = np.array(gray)
    a = 0
    b = 255
    c = np.min(img1)
    print(c)
    d = np.max(img1)
    print(d)
    new_Image = (img1-c)*((b-a)/(d-c))+a
    print(new_Image/255)
    cv2.imshow("Question1",new_Image/255)

def q2(img):
    plt.hist(img.ravel(),256,[0,256])
    plt.show()


def q3(img):
    img = cv2.cvtColor(img,cv2.IMREAD_GRAYSCALE)
    img = np.array(img)
    sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float)
    sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float)
   
    gx =cv2.filter2D(img,-1,sx)
    gy =cv2.filter2D(img,-1,sy)
    cv2.imshow('q3x',gx/np.max(gx))
    cv2.imshow('q3y',gy/np.max(gy))


def q4(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img,(3,3),1,1)
    img1 = cv2.subtract(img,blur)
    img1 = 1.25*img1 #result would be more clear as this constant goes up
    img = img +img1
    cv2.imshow('4',blur)
    cv2.imshow('q4',(img).astype(np.uint8))
    cv2.imwrite("q4.png",(img).astype(np.uint8))


if __name__ == "__main__":
    img = cv2.imread('cat.png')
    print(img.shape)
    # q1(img)
    # q2(img)
    # q3(img)
    q4(img)
    cv2.waitKey(0)