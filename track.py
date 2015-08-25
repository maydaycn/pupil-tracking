from __future__ import print_function
import argparse

import h5py
import numpy as np
import cv2
from sklearn.externals import joblib

from tracklib import extract_patches, PatchSelector


# ----------------------------------
parser = argparse.ArgumentParser(description='Track pupil from given video and trained SVM.')
parser.add_argument('eyefile', metavar='eyefile', type=str,
                    help='File containing the training data for the eye.')
parser.add_argument('eye_svm', metavar='eye_svm', type=str,
                    help='File with the trained eye SVM.')
parser.add_argument('pupilfile', metavar='pupilfile', type=str,
                    help='File containing the training data for the pupil.')
parser.add_argument('pupil_svm', metavar='pupil_svm', type=str,
                    help='File with the trained pupil SVM.')
parser.add_argument('videofile', metavar='videofile', type=str,
                    help='Video of the mouse eye.')
parser.add_argument('-t', '--threshold', type=int, default=10, help='threshold for binarizing pupil image (in percent)')
parser.add_argument('-k', '--erosion-kernel-size', type=int, default=5, help='size of the erosion kernel')
args = parser.parse_args()
# ----------------------------------

eye_selector = PatchSelector(args.eye_svm, args.eyefile)
pupil_selector = PatchSelector(args.pupil_svm, args.pupilfile)


kernel = np.ones((args.erosion_kernel_size,args.erosion_kernel_size), dtype=np.uint8)

r = int(np.floor(args.erosion_kernel_size/2))
i,j = np.meshgrid(np.arange(-r,r+1), np.arange(-r,r+1))
n = np.sqrt(i**2 + j**2)
kernel[n > r] = 0
print(kernel)

cap = cv2.VideoCapture(args.videofile)
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    i += 1
    if i < 200:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_pos = eye_selector.get_pos(gray)

    if eye_pos is not None:
        pupil_pos = pupil_selector.get_pos(eye_selector(gray))
        if pupil_pos is not None:
            pupil_pos = eye_pos+pupil_pos
            cv2.rectangle(gray, tuple(pupil_pos[::-1]), tuple( pupil_pos[::-1]+pupil_selector.full_patch_size), (255,0,0), 3)

            small_gray = gray[pupil_pos[0]:pupil_pos[0]+pupil_selector.full_patch_size,
                                pupil_pos[1]:pupil_pos[1]+pupil_selector.full_patch_size]

            _, im_bw = cv2.threshold(small_gray, np.percentile(small_gray.ravel(), args.threshold), 255, cv2.THRESH_BINARY_INV )
            cv2.medianBlur(im_bw, 7, im_bw)
            im_bw = cv2.erode(im_bw,kernel,iterations = 1)
            cv2.medianBlur(im_bw, 3, im_bw)

            im_bw = cv2.dilate(im_bw,kernel,iterations = 1)
            moments = cv2.moments(im_bw)
            try:
                cog = np.array([moments['m10']/moments['m00'], moments['m01']/moments['m00']], dtype=int)
            except:
                continue


            cv2.circle(gray, tuple(pupil_pos[::-1] + cog), 5, (255,0,0), thickness=2, lineType=8, shift=0)
        cv2.rectangle(gray, tuple(eye_pos[::-1]), tuple( eye_pos[::-1]+eye_selector.full_patch_size), (255,0,0), 3)


    cv2.imshow('frame', gray)

    # cv2.imshow('frame', 255-im_bw)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
