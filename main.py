import cv2 as cv
import os
import numpy as np
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

def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))

def Empty(p):
    pass

cv.namedWindow("Params")
cv.resizeWindow("Params", 600, 300)
cv.createTrackbar("Threshold1", "Params", 128, 256, Empty)
cv.createTrackbar("Threshold2", "Params", 256, 256, Empty)
cv.createTrackbar("Threshold3", "Params", 160, 256, Empty)
cv.createTrackbar("Threshold4", "Params", 100, 256, Empty)

# custom effect
def Summer(img, a, b, c, d):
    increaseLookupTable = LookupTable([0, a / 2, a, b], [0, c / 2, c, b])
    decreaseLookupTable = LookupTable([0, a / 2, a, b], [0, d / 2, d, b])
    blue_channel, green_channel,red_channel  = cv.split(img)
    red_channel = cv.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv.merge((blue_channel, green_channel, red_channel ))
    
    return sum

def shape_detector(frame_s):
    frame_detector_gray = cv.cvtColor(frame_s, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(frame_detector_gray, 110, 255, cv.THRESH_BINARY)
    _, contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        cv.drawContours(frame_s, [contour], 0, (0, 0, 0), 2)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5

        if len(approx) == 3:
            cv.putText(frame_s, 'Triangle', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        elif len(approx) == 4:
            x, y, w, h = cv.boundingRect(approx)
            aspect_ratio = float (w) / h
            if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                cv.putText(frame_s, 'Square', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                cv.putText(frame_s, 'Rectangle', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            cv.putText(frame_s, 'Circle', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
    return frame_s

if video.isOpened():
    while True:
        ret, raw = video.read()
        frame = raw.copy()
        
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
        
        # edge detection
        img_canny = frame.copy()
        img_canny = cv.Canny(gray, 240, 100)   

        # apply filter
        threshold1 = cv.getTrackbarPos("Threshold1", "Params")
        threshold2 = cv.getTrackbarPos("Threshold2", "Params")
        threshold3 = cv.getTrackbarPos("Threshold3", "Params")
        threshold4 = cv.getTrackbarPos("Threshold4", "Params")
        img_filter = Summer(raw, threshold1, threshold2, threshold3, threshold4)
        
        # shape detector
        img_shape = shape_detector(raw)
        
        # TITLESSS
        cv.putText(frame, "Face Recognition", (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (68, 42, 32), 2)
        cv.putText(img_canny, "Edge Detection", (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (68, 42, 32), 2)
        cv.putText(img_filter, "Image Filter", (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (68, 42, 32), 2)
        cv.putText(img_shape, "Image Shape", (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (68, 42, 32), 2)
        
        # show the frame
        image_stack = gallery_image(1, [[frame, img_canny], [img_filter, img_shape]])
        cv.imshow("Computer Vision", image_stack)
        
        key = cv.waitKey(1)
        
        # if space is pressed
        if key== 32:
            break

video.release
cv.destroyAllWindows()