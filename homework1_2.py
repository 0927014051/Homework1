#MSSV: N20DCCN134
#HO VA TEN: TRỊNH THANH SƠN
import matplotlib.pyplot as plt
import cv2 as cv2
from skimage import exposure, io

img = cv2.imread('img/dental.jpg',cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.imshow(img,cmap='gray')
plt.title('Original image')

imgEq = cv2.equalizeHist(img)
plt.subplot(2,2,2)
plt.imshow(imgEq,cmap='gray')
plt.title('Global histogram equalization')

img1_ahe = exposure.equalize_adapthist(img,kernel_size=(8,8))
plt.subplot(2,2,3)
plt.imshow(img1_ahe,cmap='gray')
plt.title('Adaptive histogram equalization, 8x8 tiles')

img2_ahe =  exposure.equalize_adapthist(img,kernel_size=(16,16))
plt.subplot(2,2,4)
plt.imshow(img1_ahe,cmap='gray')
plt.title('Adaptive histogram equalization, 16x16 tiles')

plt.show()