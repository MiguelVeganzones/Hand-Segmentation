import cv2
import matplotlib.pyplot as plt

path = "D:/OneDrive - Universidad de Valladolid/TFG/miscelanea/toolkit.mp4"

vidcap = cv2.VideoCapture(path)
success,image = vidcap.read()
count = 0
while success:
    success,image = vidcap.read()
    if count % 25 == 0:
        plt.imshow(image[200:-100,200:-200,::-1])
        #plt.imshow(image[:,:,::-1])
        plt.axis('Off')
        plt.show()
    count += 1
