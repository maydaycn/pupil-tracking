import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('BehaviorData_video.avi')

aperture = 5
thres = 50

windowClose = np.ones((5, 5), np.uint8)
windowOpen = np.ones((2, 2), np.uint8)
windowErode = np.ones((2, 2), np.uint8)

while (cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.equalizeHist(gray, gray)
    # _, im_bw = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY )
    im_bw = gray
    # cv2.medianBlur(im_bw, aperture, im_bw)
    dev = cv2.Sobel(im_bw, cv2.CV_32F, 0, 2)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
    #----------------------------------
    # TODO: Remove this later
    from IPython import embed
    embed()
    exit()
    #----------------------------------

    cv2.imshow('frame', im_bw)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
