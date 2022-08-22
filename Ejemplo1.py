import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('download.jpg', 0)
Fimg=np.fft.fft2(img)
Fimg=np.fft.fftshift(Fimg)

cv2.imshow('Fourier', 255*np.abs(Fimg)/np.max(np.max(abs(Fimg))))
cv2.waitKey(5000)
cv2.destroyAllWindows()

radio=1
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)

while (radio<rows-2):
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-radio:crow+radio, ccol-radio:ccol+radio] = 1

    FimgFilt = Fimg*mask
    FimgFilt_shift = np.fft.ifftshift(FimgFilt)
    imgResult=np.fft.ifft2(FimgFilt_shift)
    cv2.imshow('reconstruccion', np.abs(imgResult)/np.max(np.max(abs(imgResult))))
    cv2.waitKey(500)

    print(radio)
    radio=radio+1

cv2.destroyAllWindows()