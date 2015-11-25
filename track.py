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
parser.add_argument('-t', '--threshold', type=int, default=20, help='threshold for binarizing pupil image (in percent)')
parser.add_argument('-k', '--erosion-kernel-size', type=int, default=5, help='size of the erosion kernel')
parser.add_argument('-s', '--stride', type=int, default=1, help='stride for rastering the image')
args = parser.parse_args()
# ----------------------------------

eye_selector = PatchSelector(args.eye_svm, args.eyefile)
pupil_selector = PatchSelector(args.pupil_svm, args.pupilfile, args.stride)


kernel = np.ones((args.erosion_kernel_size,args.erosion_kernel_size), dtype=np.uint8)

r = int(np.floor(args.erosion_kernel_size/2))
i,j = np.meshgrid(np.arange(-r,r+1), np.arange(-r,r+1))
n = np.sqrt(i**2 + j**2)
kernel[n > r] = 0
print(kernel)
#pipe_maxr = [0,0,0,0,0]
#pipe_maxc = [[0,0],[0,0],[0,0],[0,0],[0,0]]

# Define the codec and create VideoWriter object
#video = cv2.VideoWriter('video_output.avi',-1,1,(480,640))
cap = cv2.VideoCapture(args.videofile)
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
fourcc = cv2.cv.CV_FOURCC('P','I','M','1')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    i += 1
    if i < args.stride: # TODO: 2 B removed
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_pos = eye_selector.get_pos(gray)
    print(i)

    if eye_pos is not None:
        pupil_pos = pupil_selector.get_pos(eye_selector(gray)) # this is the patch selection with SVMs
        if pupil_pos is not None:

            # ------------ START HERE.... fabian implementation
            '''
            pupil_pos = eye_pos+pupil_pos
            cv2.rectangle(gray, tuple(pupil_pos[::-1]), tuple( pupil_pos[::-1]+pupil_selector.full_patch_size), (255,0,0), 3)

            small_gray = gray[pupil_pos[0]:pupil_pos[0]+pupil_selector.full_patch_size,
                                pupil_pos[1]:pupil_pos[1]+pupil_selector.full_patch_size]

            _, im_bw = cv2.threshold(small_gray, np.percentile(small_gray.ravel(), args.threshold), 255, cv2.THRESH_BINARY_INV )
            cv2.medianBlur(im_bw, 7, im_bw)
            im_bw = cv2.erode(im_bw,kernel,iterations = 1)
            cv2.medianBlur(im_bw, 3, im_bw)

            im_bw = cv2.dilate(im_bw,kernel,iterations = 1)
            moments = cv2.moments(im_bw) # moment
            try:
                cog = np.array([moments['m10']/moments['m00'], moments['m01']/moments['m00']], dtype=int)
            except:
                continue
            '''
            # ------------ END HERE
            '''
            #START HERE JUGNU... implementation 1
            pupil_pos = eye_pos+pupil_pos
            cv2.rectangle(gray, tuple(pupil_pos[::-1]), tuple( pupil_pos[::-1]+pupil_selector.full_patch_size), (255,0,0), 3)
            small_gray = gray[pupil_pos[0]:pupil_pos[0]+pupil_selector.full_patch_size,
                                pupil_pos[1]:pupil_pos[1]+pupil_selector.full_patch_size]
            cv2.medianBlur(small_gray, 7, small_gray)
            im_bw = cv2.adaptiveThreshold(small_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            cv2.medianBlur(im_bw, 7, im_bw)
            im_bw = cv2.erode(im_bw,kernel,iterations = 1)
            cv2.medianBlur(im_bw, 3, im_bw)
            im_bw = cv2.dilate(im_bw,kernel,iterations = 1)
            im_bw_inv = 255-im_bw
            moments = cv2.moments(im_bw_inv) # moment
            try:
                cog = np.array([moments['m10']/moments['m00'], moments['m01']/moments['m00']], dtype=int)
            except:
                continue

            contours, hierarchy = cv2.findContours(im_bw_inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnt=contours[0]
            #cv2.drawContours(im_bw6,cnt,-1,(0,255,0),-1)
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            radius = int(radius)
            #END HERE JUGNU..implementation 1
            '''
            #START HERE JUGNU... implementation 2

            pupil_pos = eye_pos+pupil_pos
            cv2.rectangle(gray, tuple(pupil_pos[::-1]), tuple( pupil_pos[::-1]+pupil_selector.full_patch_size), (255,0,0), 3)
            small_gray = gray[pupil_pos[0]:pupil_pos[0]+pupil_selector.full_patch_size,
                                pupil_pos[1]:pupil_pos[1]+pupil_selector.full_patch_size]
            cv2.medianBlur(small_gray, 7, small_gray)
            th=args.threshold
            #th=0.4*np.percentile(small_gray.ravel(), 50) -40
            flag=0
            while(1):
                _, thres = cv2.threshold(small_gray, np.percentile(small_gray.ravel(), th), 255, cv2.THRESH_BINARY_INV )
                thres_copy = thres.copy()
                contours1, hierarchy1 = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                thres1=cv2.cvtColor(thres_copy, cv2.COLOR_GRAY2BGR)
                maxr=0
                for j in xrange(len(contours1)):
                    cnt1=contours1[j]
                    #cv2.drawContours(thres1,cnt1,-1,(0,255,0),-1)
                    (x,y),radius1 = cv2.minEnclosingCircle(cnt1)
                    center1 = (int(x),int(y))
                    radius1 = int(radius1)
                    if maxr < radius1:
                        maxr=radius1;
                        maxc=center1;
                #print("Radius= ",maxr)
                if (maxr>38 and flag==0):
                    th= th-10
                    #print("th=",th)
                    flag=1
                else:
                    #print("Entered break")
                    break

            #END HERE JUGNU..implementation 2




            #for k in range(4):
            #    pipe_maxc[k]=pipe_maxc[k+1]
            #    pipe_maxr[k]=pipe_maxr[k+1]
            #pipe_maxc[4]=maxc
            #pipe_maxr[4]=maxr
            #piper_avg=np.average(pipe_maxr)
            #pipec_avg0=pipe.
            #print("Average maxr and maxc are", piper_avg, pipec_avg)
            cv2.circle(gray, tuple(pupil_pos[::-1] + maxc), maxr, (255,0,0), thickness=2, lineType=8, shift=0)
        cv2.rectangle(gray, tuple(eye_pos[::-1]), tuple( eye_pos[::-1]+eye_selector.full_patch_size), (255,0,0), 3)


    cv2.imshow('frame', gray)
    out.write(gray)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()