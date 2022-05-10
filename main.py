import cv2 as cv
import os

face_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')
video = cv.VideoCapture(0)

# train
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("training.yml")

names = []
for users in os.listdir("dataset"):
    names.append(users)

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
        
        # show the frame
        cv.imshow("Face Recognition", frame)
        
        
        key = cv.waitKey(1)
        
        # if space is pressed
        if key== 32:
            break

video.release
cv.destroyAllWindows()