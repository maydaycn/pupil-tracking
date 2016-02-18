from __future__ import print_function
import argparse
import math
import random
# import h5py
import numpy as np
import cv2
import pandas as pd
# from sklearn.externals import joblib

from tracklib import extract_patches, PatchSelector, ransac

# ----------------------------------
parser = argparse.ArgumentParser(description='Track pupil from given video and trained SVM.')
parser.add_argument('x_roi', metavar='eyefile', type=int,
                    help='x-roi for pupil')
parser.add_argument('y_roi', metavar='eye_svm', type=int,
                    help='y-roi for pupil')
parser.add_argument('videofile', metavar='videofile', type=str,
                    help='Video of the mouse eye.')
parser.add_argument('-t', '--threshold', type=int, default=20, help='threshold for binarizing pupil image (in percent)')
parser.add_argument('-k', '--erosion-kernel-size', type=int, default=5, help='size of the erosion kernel')
parser.add_argument('-s', '--stride', type=int, default=0, help='stride for rastering the image')
parser.add_argument('-T', '--start', type=int, default=0, help='starting frame')
parser.add_argument('-P', '--full_patch_size', type=int, default=400, help='patch size (default 10)')
parser.add_argument('-R', '--ransac_trials', type=int, default=100,
                    help='Number of times to pick points and calculate standard deviation')
args = parser.parse_args()
# ----------------------------------


print("Starting for video", args.videofile)
file1 = open("trace.txt", 'w')
cap = cv2.VideoCapture(args.videofile)
# trace=pd.DataFrame(columns=['x_pos','y_pos','rad_min','rad_maj','angle','x_roi','y_roi','std_x','std_y','std_rad_min','std_rad_min','std_angle'])
trace = pd.DataFrame(
    columns=['pupil_x', 'pupil_y', 'pupil_r_minor', 'pupil_r_major', 'pupil_angle', 'pupil_x_std', 'pupil_y_std',
             'pupil_r_minor_std', 'pupil_r_major_std', 'pupil_angle_std', 'intensity_std'])
#trace2 = pd.DataFrame(columns=['intensity_std'])
leng = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# use tt to indicate after how many image do you want to locally save the image
tt = 10
fr_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    fr_count += 1
    if fr_count%1000 is 0:
        print("fr_count=", fr_count)
        print("Total frames = ", leng)
    if fr_count >= (leng):
        print("Video: ", args.videofile, " is over")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # eye_pos is the eye centre which is converted to patch starting pixel index below
    full_patch_size = args.full_patch_size
    eye_pos = [args.y_roi - full_patch_size / 2, args.x_roi - full_patch_size / 2]

    if eye_pos is not None:
        # START HERE JUGNU... implementation 2

        small_gray = gray[eye_pos[0]:eye_pos[0] + full_patch_size,
                     eye_pos[1]:eye_pos[1] + full_patch_size]
        cv2.medianBlur(small_gray, 7, small_gray)
        variation = np.std(small_gray)
        #trace2.loc[len(trace2) +1] = (float(variation))

        th = 0.5 * (np.percentile(small_gray.ravel(), 99)) + 0.5 * (np.percentile(small_gray.ravel(), 1))

        flag = 0
        while (1):
            _, thres = cv2.threshold(small_gray, th, 255, cv2.THRESH_BINARY)
            thres_copy = thres.copy()
            _, contours1, hierarchy1 = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            maxc = None
            maxr = 0
            for j in xrange(len(contours1)):
                cnt1 = contours1[j]
                (x, y), radius1 = cv2.minEnclosingCircle(cnt1)
                center1 = (int(x), int(y))
                radius1 = int(radius1)
                draw = 0
                # r2centerx, r2centery, r2majrad, r2minrad, r2angle, small_gray = ransac(args.ransac_trials, cnt1, small_gray, draw)
                # if ((maxr < radius1 - r2majrad.std()/2) and (center1[1] > 0.20*full_patch_size) and (center1[1] < 0.80*full_patch_size) and (center1[0] > 0.20*full_patch_size) and (center1[0] < 0.80*full_patch_size) and (radius1 >5) and (radius1 <180) and len(contours1[j]) >= 5):
                if ((maxr < radius1 - 0.05 * (center1[0] - full_patch_size / 2) - 0.05 * (
                    center1[1] - full_patch_size / 2)) and (center1[1] > 0.20 * full_patch_size) and (
                    center1[1] < 0.80 * full_patch_size) and (center1[0] > 0.20 * full_patch_size) and (
                    center1[0] < 0.80 * full_patch_size) and (radius1 > 5) and (radius1 < 180) and len(
                        contours1[j]) >= 5):
                    maxr = radius1 - 0.05 * (center1[0] - full_patch_size / 2) - 0.05 * (
                    center1[1] - full_patch_size / 2)
                    maxc = center1
                    maxj = j

            break

        cv2.rectangle(gray, tuple(eye_pos[::-1]), tuple([x + full_patch_size for x in eye_pos[::-1]]), (255, 0, 0), 3)
        # cv2.circle(small_gray,maxc,maxr,(0,255,0),thickness=5)
        if maxc is not None:
            draw = 1
            r2centerx, r2centery, r2majrad, r2minrad, r2angle, small_gray = ransac(args.ransac_trials, contours1[maxj],
                                                                                   small_gray, draw)

            cv2.drawContours(small_gray, contours1, maxj, (255, 0, 0), 1)
            ellipse = cv2.fitEllipse(contours1[maxj])
            cv2.ellipse(small_gray, ellipse, (0, 0, 255), 2)

            # trace=pd.DataFrame(columns=['x_pos','y_pos','rad_min','rad_maj','angle','std_x','std_y','std_rad_min','std_rad_min','std_angle'])

            trace.loc[len(trace) + 1] = (
            float(ellipse[0][0] + eye_pos[1] + 1), float(ellipse[0][1] + eye_pos[0] + 1), float(ellipse[1][0]),
            float(ellipse[1][1]), float(ellipse[2]),
            float(r2centerx.std()), float(r2centery.std()), float(r2minrad.std()), float(r2majrad.std()),
            float(r2angle.std()), float(variation))

            # file1.write("\n")
        else:
            print("No ellipse found")
            # file1.write("NONE \n")
            trace.loc[len(trace) + 1] = 11 * (None,)

    else:
        # file1.write("NONE \n")
        trace.loc[len(trace) + 1] = 11 * (None,)

    re = fr_count % tt
    f_count = fr_count // tt
    if re == 0:
        name = "images/img%06d.bmp" % (fr_count,)
        #print("Writing file for ", fr_count)
        cv2.imwrite(name, gray)

    if (cv2.waitKey(1) & 0xFF == ord('q')) or fr_count >= 999999:
        break

cap.release()
file1.close()
trace.to_csv('trace.csv', index=False)
#trace2.to_csv('intensity_std.csv', index=False)
# pd.HDFStore('trace.h5')
cv2.destroyAllWindows()
