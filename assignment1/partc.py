import cv2
import numpy as np
import argparse
import os

def maxFilter(img,height,width,kernal_N):
    max_filter = np.zeros([height,width],dtype=np.uint8)
    ksize = int((kernal_N-1)/2)

    for i in range(height):
        for j in range(width):
            heightLowBound = max(0,i-ksize)
            heightHighBound = min(height-1,i+ksize)
            widthLowBound = max(0,j-ksize)
            widthHighBound =min(width-1,j+ksize)

            Max_value = 0
            for ki in range(heightLowBound,heightHighBound+1):
                for kj in range(widthLowBound,widthHighBound+1):
                    value =img[ki][kj]
                    if(value > Max_value):
                        Max_value = value
            max_filter[i][j] = Max_value
    return max_filter

def minFilter(img,height,width,kernal_N):
    min_filter = np.zeros([height,width],dtype=np.uint8)
    ksize = int((kernal_N-1)/2)

    for i in range(height):
        for j in range(width):
            heightLowBound = max(0,i-ksize)
            heightHighBound = min(height-1,i+ksize)
            widthLowBound = max(0,j-ksize)
            widthHighBound =min(width-1,j+ksize)

            Min_value = 999
            for ki in range(heightLowBound,heightHighBound+1):
                for kj in range(widthLowBound,widthHighBound+1):
                    value =img[ki][kj]
                    if(value < Min_value):
                        Min_value = value
            min_filter[i][j] = Min_value
    return min_filter

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--image",required=True,help="path to input image")
    ap.add_argument("-n",'--number',required=True,help="kernal size for filter")
    ap.add_argument('-m','--max',required=True,help="reverse or not")

    args = vars(ap.parse_args())
    N=int(args['number'] )
    if(N<1 or N%2==0):
        exit("Error: kernal filter size should be odd or positive")

    img = cv2.imread(args["image"])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.uint8)
    height = img.shape[0]
    width = img.shape[1]


    if(int(args["max"])== 0):
        max_filter = maxFilter(img,height,width,N).astype(np.uint8)
        min_filter = minFilter(max_filter,height,width,N).astype(np.uint8)
        backgroundSub = img - max_filter +255
    else:
        min_filter = minFilter(img,height,width,N).astype(np.uint8)
        max_filter = maxFilter(min_filter,height,width,N).astype(np.uint8)
        backgroundSub = img - max_filter
    cv2.imshow('A',max_filter/np.max(max_filter))
    cv2.imshow('B',min_filter/np.max(min_filter))

    cv2.imshow('task2',backgroundSub/np.max(backgroundSub))
 
    cv2.waitKey(0)