
#This example is from "Introduction to OpenCV for Beginners" Form "simplilearn"   tolearn Opencv on PYTHON


import cv2 as cv

image_path = r'test.jpg'
coordinate = (100,200)
font = cv.FONT_HERSHEY_SIMPLEX
fontscale = 1.5
color = (255,200,0)
tickness = 5
start_point = (150,300)
end_point = (300, 80)
center = (140,150)
radius = 200
axeslength = (100,50)
angle=30
startangle = 0
endangle = 360

def Start():
    image = cv.imread(image_path)  # this is for reading image from a file
    cv.imshow('image', image)  # This is for showing that image in screen

    cv.imwrite('test.png', image)  # this for write image to a new file

    # properties of picture
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  #to change image color like (COLOR_BGR2GRAY , ...)
    cv.imshow('gray', gray)

# to resize image
    resize_img =cv.resize(image, (640,480))

# put text on image
    cv.putText(resize_img,'Resized By opencv', coordinate, font, fontscale, color, tickness)

# Draw a line
    cv.line(resize_img,start_point, end_point, color, tickness)

#draw a circle
    cv.circle(resize_img,center,radius, color, tickness)

#draw a rectangle
    cv.rectangle(resize_img,start_point,end_point,color,tickness)

#draw ellipse
    cv.ellipse(resize_img,center,axeslength, angle, startangle, endangle,color,-1)   #tickness = -1  fill ellipse
    cv.imshow('resize img', resize_img)



    cv.waitKey(0)  # This is for waiting program until a key press
    cv.destroyAllWindows()  # this command destroy all windows


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Start()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
