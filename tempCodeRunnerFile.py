cv.namedWindow("Params")
cv.resizeWindow("Params", 600, 300)
cv.createTrackbar("Threshold1", "Params", 128, 256, Empty)
cv.createTrackbar("Threshold2", "Params", 256, 256, Empty)
cv.createTrackbar("Threshold3", "Params", 160, 256, Empty)
cv.createTrackbar("Threshold4", "Params", 100, 256, Empty)