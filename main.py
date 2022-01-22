# IMPORTING REQUIRED MODULES

import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import easyocr

# CODE

img = cv2.imread(r"images/img_1.png")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(grey, cv2.COLOR_BGR2RGB))  # To draw the image. Matplotlib wants the image to be in RGB format so we are converting the image to RGB.
# plt.show()  # To display the image

# FINDING EDGES OF THE IMAGE

bfilter = cv2.bilateralFilter(grey, 11, 17, 17)  # Noise Reduction
edged = cv2.Canny(bfilter, 30, 200)  # Edge Detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
# plt.show()

# FINDING CONTOURS

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Finding contours
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Sort and return top 10 contours

# FINDING THE APPROX LOCATION OF THE NUMBER PLATE

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
# print(location)

# FINDING THE EXACT LOCATION OF THE NUMBER PLATE

mask = np.zeros(grey.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
# plt.show()

# CROPPED NUMBER PLATE IMAGE

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))

cropped_image = grey[x1: x2 + 1,  y1: y2 + 1]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
# plt.show()

# USING EASY OCR TO READ THE TEXT

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
# print(result)

# RENDERING THE RESULT

text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.show()




