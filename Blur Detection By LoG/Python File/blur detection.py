'''Categorisation of Image whether it is Blurred Image or a Clear Image cv2 numpy Laplacian of Gaussian

Notebook:- Using Laplacian of Gaussian(GoS) checking the bluriness of any image
Last Modified:- September 2020

Author:- AmRiyaz'''

import cv2
import numpy as np

img = cv2.imread("3.jpeg", cv2.IMREAD_GRAYSCALE)

laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
if laplacian_var < 5:
    print("Image is blur !")
else:   
    print("Image is Clear !")

print(laplacian_var)


cv2.imshow("Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
