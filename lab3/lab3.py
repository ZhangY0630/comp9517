import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage as ndi
from skimage.morphology import watershed,disk
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift
from skimage.filters import rank
from PIL import Image
from PIL import ImageFilter
import cv2
size = 100, 100

img_names = ["shapes.png", "strawberry.png"]
ext_names = ["coins.png", "kiwi.png"]

images = [i for i in img_names]
ext_images = [i for i in ext_names]

def applyMaximumFilter(image):
    image = image.filter(ImageFilter.MinFilter)
    return image.filter(ImageFilter.MaxFilter)
def edgeDetection(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) 
    return sobelx

def plot_three_images(figure_title, image1, label1,
                      image2, label2, image3, label3):
    fig = plt.figure()
    fig.suptitle(figure_title)

    # Display the first image
    fig.add_subplot(1, 3, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title(label1)

    # Display the second image
    fig.add_subplot(1, 3, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.title(label2)

    # Display the third image
    fig.add_subplot(1, 3, 3)
    plt.imshow(image3)
    plt.axis('off')
    plt.title(label3)

    plt.show()


for img_path in images:
    img1 = Image.open(img_path)
    img1.thumbnail(size)
    width ,height = img1.size
    print(f"{width}{height}")
    print((width,height))
    img1_m_r = np.array(img1)[:,:,0].flatten()
    img1_m_g = np.array(img1)[:,:,1].flatten()
    img1_m_b = np.array(img1)[:,:,2].flatten()

    img1_f = np.column_stack([img1_m_r,img1_m_g,img1_m_b])

    ms_clf = MeanShift(bin_seeding=True)
    ms_labels = ms_clf.fit_predict(img1_f)
    ms_labels = ms_labels.reshape(height,width)


    #task2
    img1_g = img1.convert("L")
    img1_g = np.array(img1_g)
    print(img1_g.shape)
    distance = ndi.distance_transform_edt(img1_g)
    local_maxi = peak_local_max(-distance, indices=False, footprint=np.ones((5, 5)),labels=img1_g)
    markers = ndi.label(local_maxi)[0]
    ws_labels = watershed(-distance, markers, mask=img1_g)

    plot_three_images(img_path, img1, "Original Image", ms_labels, "MeanShift Labels",
                      ws_labels, "Watershed Labels")
    

for img_path in ext_images:
    img1 = Image.open(img_path)
    img1.thumbnail(size)

    width ,height = img1.size
    print(f"{width}{height}")
    print((width,height))
    img1_m_r = np.array(img1)[:,:,0].flatten()
    img1_m_g = np.array(img1)[:,:,1].flatten()
    img1_m_b = np.array(img1)[:,:,2].flatten()

    img1_f = np.column_stack([img1_m_r,img1_m_g,img1_m_b])

    ms_clf = MeanShift(bin_seeding=True)
    ms_labels = ms_clf.fit_predict(img1_f)
    ms_labels = ms_labels.reshape(height,width)


    #task2
    img1_g = img1.convert("L")
    img1_g = np.array(img1_g)
    print(img1_g.shape)
    distance = ndi.distance_transform_edt(img1_g)
    local_maxi = peak_local_max(-distance, indices=False, footprint=np.ones((5, 5)),labels=img1_g)
    markers = ndi.label(local_maxi)[0]
    ws_labels = watershed(-distance, markers, mask=img1_g)

    # Display the results
    plot_three_images(img_path, img1, "Original Image", ms_labels, "MeanShift Labels",
                      ws_labels, "Watershed Labels")

    img1_g = img1.convert("L")
    img1_g = np.array(img1_g)

    thresh = cv2.threshold(img1_g, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    distance = ndi.distance_transform_edt(img1_g) 
    local_maxi = peak_local_max(distance, min_distance=10, indices=False,labels=thresh)
    markers = ndi.label(local_maxi,structure=np.ones((3, 3)))[0]
    ws_labels = watershed(-distance, markers, mask=thresh) 
    
    plot_three_images(img_path, img1, "Original Image", ms_labels, "MeanShift Labels",
                      ws_labels, "Watershed Labels")

    # ws_labels = watershed(-distance, markers, mask=img1_g)
    # plt.imshow(ws_labels)
    # plt.show()