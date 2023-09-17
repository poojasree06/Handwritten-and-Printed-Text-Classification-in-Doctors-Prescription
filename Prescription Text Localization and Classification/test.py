import numpy as np
import cv2 as cv
im = cv.imread('/home/pooja/Handwritten-and-Printed-Text-Classification-in-Doctors-Prescription/Prescription Text Localization and Classification/Prescription_Samples/C/sample1.jpg')
assert im is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(im, contours, -1, (0,255,0), 3)
print(contours)