#access to properties of image and videos

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img =cv.imread('test.png')
px = img[100,100]                   # the specific pixel of img [100, 100]
print(px)                           #print out properties

blue = img[100,100,0]               # show the properties of blue =0 green=1 red=2  as BRG
print(blue)

img[100, 100] = [255, 255, 255]     # change color of pixel [100,100]
print(px)

colory    = cv.imread('test.jpg', cv.IMREAD_COLOR)
alpha_img = cv.imread('test.jpg', cv.IMREAD_UNCHANGED)
gray_img  = cv.imread('test.jpg', cv.IMREAD_GRAYSCALE)

#shape of img
print('RGB shape   : ', colory.shape)
print('Alpha shape : ', alpha_img.shape)
print('gray shape  : ', gray_img.shape)
#img type
print('image datatype : ', colory.dtype)

#img size
print('image Size : ', colory.size)

# cropping selected ROI from img
roi = cv.selectROI(img)
print(roi)
roi_cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]   #get cropped position
cv.imshow("ROI Image", roi_cropped)                                                 #show cropped img
cv.imwrite("cropped.jpg", roi_cropped)                                              #save cropped to file


#split image to three channel
g,b,r = cv.split(img)

cv.imshow("green", g)
cv.imshow("blue" , b)
cv.imshow("red"  , r)

imag = cv.merge((g,b,r))          #merge two channel to one
cv.imshow("merge",imag)


# change color of image
lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
cv.imshow("lab view", lab)


# blending  two image
src1 = cv.imread('test.png', cv.IMREAD_COLOR)
src2 = cv.imread('index.jpg', cv.IMREAD_COLOR)
img1 = cv.resize(src1, (800,600))                   #resize image
img2 = cv.resize(src2, (800,600))
blended_img = cv.addWeighted(img1,0.5,img2,1,0.0)   #blend two image together
cv.imshow("blended image", blended_img)

# Apply filters
k_sharpen = np.array([[-1,-1,-1],
                      [-1,9 ,-1],
                      [-1,-1,-1]])
k_edge = np.array([[1,1,1],
                    [1,-9 ,1],
                    [1,1,1]])
# apply filters
sharpen = cv.filter2D(img,-1,k_edge)
sharpen = cv.filter2D(img,-1,k_sharpen)
cv.imshow("filtered", sharpen)


# some other filters
gray = cv.imread("index.jpg", cv.IMREAD_GRAYSCALE)
ret, thresh = cv.threshold(gray,127, 255, cv.THRESH_BINARY)
canny_img = cv.Canny(gray,50,100)
cv.imshow("original", gray)
cv.imshow("threshold", thresh)
cv.imshow("canny", canny_img)


# contour and shape detection
shape = cv.imread('shapes.png')
gray = cv.cvtColor(shape, cv.COLOR_BGR2GRAY)

#Setting threshold of the gray image
_, threshold = cv.threshold(gray,127,255, cv.THRESH_BINARY)

#countour using findcontours function
contours,_= cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

i=0
for contour in contours:
    if i==0:
        i=1
        continue
    appox = cv.approxPolyDP(contour,0.01*cv.arcLength(contour,True), True)
    cv.drawContours(shape, [contour], 0, (255,0,255), 5)

    #findind the center of diffrent shapes
    M = cv.moments(contour)
    if M['m00'] != 0.0:
        x= int(M['m10']/M['m00'])
        y= int(M['m01']/M['m00'])

    # I want to put names of shaoes inside of shape
    if len(appox) ==3:
        cv.putText(shape,'Triangle', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    elif len(appox) ==4:
        cv.putText(shape, 'Quadri', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif len(appox) ==5:
        cv.putText(shape, 'Pentagon', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif len(appox) ==6:
        cv.putText(shape, 'Hegzaton', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif len(appox) ==7:
        cv.putText(shape, '7-Gon', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv.putText(shape, 'Circle', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

cv.imshow('shapes', shape)

# Color detection
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
#range of blue color to show
lower_blue = np.array([0,50,50])
upper_blue = np.array([140,255,255])
#throshold the hdv image to get only blue colors
make_blue = cv.inRange(hsv, lower_blue,upper_blue)
res = cv.bitwise_and(img,img,mask=make_blue)        #put mask on image
cv.imshow('res', res)


# place an object
img1 = shape.copy()
mask=np.zeros((100,200,3))
print(mask.shape)
pos = (200,200)
var = img1[200:(200+mask.shape[0]), 200:(200+mask.shape[1])]=mask
cv.imshow("coloring",img1)


cv.waitKey(0)
cv.destroyAllWindows()