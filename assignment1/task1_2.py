import cv2
import numpy as np

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

def correctNeg(img,height,width):
    for i in range(height):
        for j in range(width):
            # if(img[i][j]!=0):
            img[i][j]+=255

    return img

if __name__ == "__main__":
    
    img = cv2.imread("Particles.png") # assignment1/Particles.png
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.uint8)

    height = img.shape[0]
    width = img.shape[1]
    N=9
    max_filter = maxFilter(img,height,width,N).astype(np.uint8)
    min_filter = minFilter(max_filter,height,width,N).astype(np.uint8)
    cv2.imshow('A',max_filter/np.max(max_filter))
    cv2.imshow('B',min_filter/np.max(min_filter))
    cv2.imwrite("B.png",min_filter.astype(np.uint8))
    # print(max_filter)
    # print(min_filter)

    backgroundSub = correctNeg( img - min_filter ,height,width)

    print(backgroundSub)
    cv2.imshow('task2',backgroundSub/np.max(backgroundSub))
    cv2.imwrite("output.png",backgroundSub.astype(np.uint8))
 
    cv2.waitKey(0)