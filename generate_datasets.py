import cv2
import os
import schedule

from pathlib import Path

# init
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
vc = cv2.VideoCapture(0)

# get name
print("Please enter your name: ")
uname = input()

# generate index
idx = 1
for users in os.listdir("dataset"):
    idx += 1

# flag = 0
    
# init count
count = 1

# capture then save the image
def captSaveImage(image, uname, idx, imgId):
    # create folder using name
    Path("dataset/{}".format(uname)).mkdir(parents=True, exist_ok=True)
    # save the image in folder
    cv2.imwrite("dataset/{}/{}_{}.jpg".format(uname, idx, imgId), image)
    print("[INFO] Image {} has been saved in folder : {}".format(
        imgId, uname))

def takeImage():
    flag = 1

print("Capturing video...")

while True:
    # capture the image
    _, img = vc.read()

    # create copy
    originalImg = img.copy()

    # grayscale the image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # create the coords of the face
    faces = faceCascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
                    img,
                    uname,
                    (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (147, 20, 255),
                    1,
                    cv2.LINE_AA,
                )
        coords = [x, y, w, h]
        found = 1

    cv2.imshow("Captured face", img)
    
    key = cv2.waitKey(1) & 0xFF
    
    # schedule.every(1).seconds.do(takeImage)
  
    # if flag == 1:
    #     roi_img = originalImg[coords[1] : coords[1] + coords[3], coords[0] : coords[0] + coords[2]]
    #     captSaveImage(roi_img, uname, idx, count)
    #     count += 1
    #     flag = 0
  
    # # If q is pressed break out of the loop
    # if key == ord('q'):
    #     break
    
    # take pict everytime space is pressed
    if key == 32:
        # take until 5 image
        if count <= 5:
            roi_img = originalImg[coords[1] : coords[1] + coords[3], coords[0] : coords[0] + coords[2]]
            captSaveImage(roi_img, uname, idx, count)
            count += 1
        else:
            break
    # if q pressed brake
    if key == ord('q'):
        break

print("{}'s dataset has been created".format(uname))

vc.release()
cv2.destroyAllWindows()