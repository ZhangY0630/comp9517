import cv2

img = cv2.imread("Particles.png") # assignment1/Particles.png
print(img.shape   )
# cap = cv2.VideoCapture(0)
# cap.set(3,50)
# cap.set(4,50)
# cap.set(10,100)
# while True:
#     success,img = cap.read()
#     cv2.imshow("Video",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         breacack