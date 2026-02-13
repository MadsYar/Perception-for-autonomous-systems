
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils 

# Exercises 1
# bgr_img = cv2.imread("jp.png")
# b,g,r = cv2.split(bgr_img)       # get b,g,r
# image = cv2.merge([r,g,b])
# (h, w, d) = image.shape

def one_one():
    print("width={}, height={}, depth={}".format(w, h, d))

    plt.imshow(image)

    plt.show()

def indexing():
    # Data / Image indexing
    (R, G, B) = image[100, 50]
    print("R={}, G={}, B={}".format(R, G, B))

    titles=['r','g','b']
    plt.figure(figsize = (16,4))
    for i in range(3):
        channel = np.zeros_like(image)
        channel[:,:,i] = image[:,:,i]
        plt.subplot(1,3,i+1), plt.imshow(channel)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    # Selecting a 100 x 100 frame of pixels from the picture
    roi = image[60:160, 320:420]
    plt.imshow(roi)
    plt.show()

def resize():
    # Resize 200x200
    resized = cv2.resize(image, (200, 200))
    plt.imshow(resized)
    plt.show()

    # Rezise dynamically
    r = 300.0 / w
    dim = (300, int(h * r))
    resized = cv2.resize(image, dim)
    plt.imshow(resized)
    plt.show()

    # Resize using imutils
    resized = imutils.resize(image, width=300)
    plt.imshow(resized)
    plt.show()

def rotate():
    # Rotate 45 degrees using opencsv
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -45, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    plt.imshow(rotated)  
    plt.show()
    
    # Rotate 45 degrees using imutils
    rotated = imutils.rotate(image, -45)
    plt.imshow(rotated)
    plt.show()
    
    # If you also want to keep the bounds of the picture
    rotated = imutils.rotate_bound(image, 45)
    plt.imshow(rotated)
    plt.show()
    
def blur():
    kernel = (11,11)

    blurred = cv2.GaussianBlur(image, kernel, 0)
    plt.imshow(blurred)
    plt.show()

def draw():
    output = image.copy()
    cv2.rectangle(output, (320, 60), (420, 160), (0, 0, 255), 2)
    plt.imshow(output)
    plt.show()

    output = image.copy()
    cv2.circle(output, (300, 150), 20, (255, 0, 0), -1)
    plt.imshow(output)
    plt.show()

    output = image.copy()
    cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
    plt.imshow(output)
    plt.show()

    output = image.copy()
    cv2.putText(output, "OpenCV + Jurassic Park!!!", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    plt.imshow(output)

    plt.show()

# Exercises 2
path = "tetris_blocks.png"

bgr_img = cv2.imread(path)
b,g,r = cv2.split(bgr_img)       # get b,g,r
image = cv2.merge([r,g,b])
# plt.imshow(image)
# plt.show()

def grayscale():
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plt.imshow(gray, cmap = 'gray')
    plt.show()

def edge_detection():
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edged = cv2.Canny(gray, 30, 150)
    plt.imshow(edged, cmap='gray')
    plt.show()

def treshhold():
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edged = cv2.Canny(gray, 30, 150)
    plt.imshow(edged, cmap='gray')
    plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

threshold = 225
threshold_value = 255

thresh = cv2.threshold(gray, threshold, threshold_value, cv2.THRESH_BINARY_INV)[1]
plt.imshow(thresh, cmap='gray')

# Finding the contours of the objects in the images
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

for c in cnts:
    # draw each contour on the output image with a 3px thick black outline
    cv2.drawContours(output, [c], -1, (0, 0, 0), 3)
    
plt.imshow(output)
plt.show()

# Showing the amount of objects in the picture
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (155, 0, 155), 2)
plt.imshow(output)
plt.show()

# Making the objects smaller
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations = 5)
plt.imshow(mask, cmap = 'gray')
plt.show()

# Making the objects larger
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations = 5)
plt.imshow(mask, cmap='gray')

# Removing the background and only keeping the objects with their original background
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
plt.imshow(output)


