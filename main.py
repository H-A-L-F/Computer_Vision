import cv2 as cv
import os
import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline

face_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')
video = cv.VideoCapture(0)

# train
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("training.yml")

names = []
for users in os.listdir("dataset"):
    names.append(users)

# to show multiple image
def gallery_image(scale, img_arr):
    rows = len(img_arr)
    cols = len(img_arr[0])
    available_rows = isinstance(img_arr[0], list)
    width = img_arr[0][0].shape[1]
    height = img_arr[0][0].shape[0]
    if available_rows:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if img_arr[x][y].shape[:2] == img_arr[0][0].shape [:2]:
                    img_arr[x][y] = cv.resize(img_arr[x][y], (0, 0), None, scale, scale)
                else:
                    img_arr[x][y] = cv.resize(img_arr[x][y], (img_arr[0][0].shape[1], img_arr[0][0].shape[0]), None, scale, scale)
                if len(img_arr[x][y].shape) == 2: img_arr[x][y]= cv.cvtColor(img_arr[x][y], cv.COLOR_GRAY2BGR)
        blank_image = np.zeros((height, width, 3), np.uint8)
        horizontal = [blank_image]*rows
        for x in range(0, rows):
            horizontal[x] = np.hstack(img_arr[x])
        vertic = np.vstack(horizontal)
    else:
        for x in range(0, rows):
            if img_arr[x].shape[:2] == img_arr[0].shape[:2]:
                img_arr[x] = cv.resize(img_arr[x], (0, 0), None, scale, scale)
            else:
                img_arr[x] = cv.resize(img_arr[x], (img_arr[0].shape[1], img_arr[0].shape[0]), None,scale, scale)
            if len(img_arr[x].shape) == 2: img_arr[x] = cv.cvtColor(img_arr[x], cv.COLOR_GRAY2BGR)
        horizontal = np.hstack(img_arr)
        vertic = horizontal
    return vertic

#defining a function
def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))

# custom effect
def Summer(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv.split(img)
    red_channel = cv.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv.merge((blue_channel, green_channel, red_channel ))
    return sum

if video.isOpened():
    while True:
        ret, frame = video.read()
        
        # grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for(x, y, w, h) in faces:
            cv.rectangle(frame, (x,y), (x + w, y + h), (0, 0, 255), 2)
            id, percentage = recognizer.predict(gray[y : y + h, x : x + w])
            if percentage <= 50:
                id = 0
            print(id)
            if id > 0:
                cv.putText(
                    frame,
                    names[id - 1],
                    (x, y - 4),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (147, 20, 255),
                    1,
                    cv.LINE_AA,
                )
            else:
                cv.putText(
                    frame,
                    "Unknown",
                    (x, y - 4),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (147, 20, 255),
                    1,
                    cv.LINE_AA,
                )
            
            img_canny = cv.Canny(gray, 240, 100)
        
        # show the frame
        image_stack = gallery_image(1, [[frame, img_canny], [Summer(frame), frame]])
        cv.imshow("Face Recognition", image_stack)
        
        
        key = cv.waitKey(1)
        
        # if space is pressed
        if key== 32:
            break

video.release
cv.destroyAllWindows()