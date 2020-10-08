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