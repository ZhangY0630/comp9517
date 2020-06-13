import cv2
import numpy as np
from matplotlib import pyplot as plt
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
    plt.hist(img.ravel(),256,[0,256]);
    plt.show()


  



if __name__ == "__main__":
    img = cv2.imread('cat.png')
    q1(img)
    q2(img)
    