import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

# -----------------------------------------------------------
parser = argparse.ArgumentParser(description='Collect pupil training data from video')
parser.add_argument('-P', '--patch-size', type=int, default=500, help='patch size (default 400)')
parser.add_argument('videofile', metavar='videofile', type=str,
                    help='Videofile with showing the eye. ')

args = parser.parse_args()
# -----------------------------------------------------------


fig, ax = plt.subplots()
cap = cv2.VideoCapture(args.videofile)
leng = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
print("Number of frames=", leng)

ii=0
while cap.isOpened():
    ii+=1
    print(ii)
    ret, frame = cap.read()
    if ii >= (500):
        while(1):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ax.imshow(gray, cmap=plt.cm.gray)
            ax.set_title('Click on the center of the eye!')
            x = plt.ginput(1) #jugnu to take the input from mouse 1 time
            print("input_mouse_len=", len(x), x)
            psh = args.patch_size/2
            x = [x[0][1],x[0][0]]
            ax.fill_betweenx([x[0] - psh, x[0] + psh], [x[1] - psh, x[1] - psh], [x[1] + psh, x[1] + psh],
                            color='lime', alpha=.1)
            #print('Roi=', x[0], x[1])
        break






