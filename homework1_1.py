#MSSV: N20DCCN134
#HO VA TEN: TRỊNH THANH SƠN

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2

img = cv2.imread('img/moon.jpg',cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe1 = clahe.apply(img)

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title('Original image')

plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(clahe1,cv2.COLOR_BGR2RGB))
plt.title('Contrast limit histogram equalization')

hist_og = cv2.calcHist([img],[0],None,[256],[0,256])
plt.subplot(2,2,3)
plt.bar(np.arange(256),hist_og.ravel())
plt.xlim([0,256])
plt.xlabel('Gray level')
plt.ylabel('Pixels')
plt.title('Histogram of original image')

hist_og = cv2.calcHist([clahe1],[0],None,[256],[0,256])
plt.subplot(2,2,4)
plt.bar(np.arange(256),hist_og.ravel())
plt.xlim([0,256])
plt.xlabel('Gray level')
plt.ylabel('Pixels')
plt.title('Histogram of Contrast limit histogram equalization')
plt.show()
