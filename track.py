from __future__ import print_function
import argparse
import math
import random
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
parser.add_argument('videofile', metavar='videofile', type=str,
                    help='Video of the mouse eye.')
parser.add_argument('-t', '--threshold', type=int, default=20, help='threshold for binarizing pupil image (in percent)')
parser.add_argument('-k', '--erosion-kernel-size', type=int, default=5, help='size of the erosion kernel')
parser.add_argument('-s', '--stride', type=int, default=0, help='stride for rastering the image')
parser.add_argument('-T', '--start', type=int, default=0, help='starting frame')
args = parser.parse_args()
# ----------------------------------

eye_selector = PatchSelector(args.eye_svm, args.eyefile)

file1=open("trace.txt",'w')
cap = cv2.VideoCapture(args.videofile)

leng = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

tt=1
fr_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    fr_count+=1

    print("fr_count=",fr_count)
    print("Total frames = ",leng)
    if fr_count >= (leng):
        print("Video: ",args.videofile," is over")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_pos = eye_selector.get_pos(gray)


    if eye_pos is not None:
        #START HERE JUGNU... implementation 2

        #cv2.rectangle(gray, tuple(eye_pos[::-1]), tuple( eye_pos[::-1]+eye_selector.full_patch_size), (255,0,0), 3)
        small_gray = gray[eye_pos[0]:eye_pos[0]+eye_selector.full_patch_size,
                            eye_pos[1]:eye_pos[1]+eye_selector.full_patch_size]
        cv2.medianBlur(small_gray, 7, small_gray)
        th=args.threshold
        th=0.5*(np.percentile(small_gray.ravel(),99)) + 0.5 *(np.percentile(small_gray.ravel(),1))
        #th=0.4*np.percentile(small_gray.ravel(), 50) -40
        flag=0
        while(1):
            _, thres = cv2.threshold(small_gray, th, 255, cv2.THRESH_BINARY )
            thres_copy = thres.copy()
            contours1, hierarchy1 = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #thres1=cv2.cvtColor(thres_copy, cv2.COLOR_GRAY2BGR)
            maxc=None
            maxr=0
            for j in xrange(len(contours1)):
                cnt1=contours1[j]
                (x,y),radius1 = cv2.minEnclosingCircle(cnt1)
                center1 = (int(x),int(y))
                radius1 = int(radius1)
                if ((maxr < radius1) and (center1[1] > 70) and (center1[1] < 230) and (center1[0] > 70) and (center1[0] < 230) and (radius1 >5) and (radius1 <180)):
                    maxr=radius1;
                    maxc=center1;
                    maxj=j
                #cv2.circle(thres1,center1,radius1,(0,255,0),thickness=5)
            #print("Radius= ",maxr)
            break

        cv2.rectangle(gray, tuple(eye_pos[::-1]), tuple( eye_pos[::-1]+eye_selector.full_patch_size), (255,0,0), 3)
        #cv2.circle(small_gray,maxc,maxr,(0,255,0),thickness=5)
        if maxc is not None:
            #RANSAC1 implementation starts
            angles=[]
            for pts in contours1[maxj]:
                x=(pts[0][0]-maxc[0])
                if x == 0:
                    alpt=float(pts[0][1]-maxc[1])/0.001
                else:
                    alpt=float(pts[0][1]-maxc[1])/x
                alp=math.atan(alpt)
                angles.append(alp*180/3.14) 
            centerx=[]
            centery=[]
            ang=np.asarray(angles)
            for i in np.random.uniform(-80,80,30):
                #print("Doing for angle=",i)
                idx1=np.asarray(np.where(ang>(i-10)))
                idx2=np.asarray(np.where(ang<(i+10)))
                res=np.intersect1d(idx1,idx2)
                samples=contours1[maxj][res]
                #print("size samples=",len(samples))
                #from IPython import embed
                #embed()
                centerx.append(samples.mean(axis=0)[0][0])
                centery.append(samples.mean(axis=0)[0][1])
                #if len(samples)>5:
                    #ellipse=cv2.fitEllipse(samples)
                    
                    #cv2.ellipse(small_gray,ellipse,(0,0,255),2)
                    #centerx.append(ellipse[0][0])
                    #centery.append(ellipse[0][1])
            centerx=np.asarray(centerx)
            centery=np.asarray(centery)
            #RANSAC1 implementation ends


            #from IPython import embed
            #embed()
            #RANSAC2 implementation starts
            r2centerx=[]
            r2centery=[]
            r2majrad=[]
            r2minrad=[]
            r2angle=[]
            for i in range(100):
                if len(contours1[maxj])>60:
                    samples=np.asarray(random.sample(contours1[maxj],len(contours1[maxj])/10))
                    ellipse=cv2.fitEllipse(samples)
                    cv2.ellipse(small_gray,ellipse,(0,0,255),2)
                    r2centerx.append(ellipse[0][0])
                    r2centery.append(ellipse[0][1])
                    r2majrad.append(ellipse[1][1])
                    r2minrad.append(ellipse[1][0])
                    r2angle.append(ellipse[2])
                else:
                    r2centerx.append(0)
                    r2centery.append(0)
                    r2majrad.append(0)
                    r2minrad.append(0)
                    r2angle.append(0)
            r2centerx=np.asarray(r2centerx)
            r2centery=np.asarray(r2centery)
            r2majrad=np.asarray(r2majrad)
            r2minrad=np.asarray(r2minrad)
            r2angle=np.asarray(r2angle)

            #from IPython import embed
            #embed()







            #RANSAC2 implementation ends

            cv2.drawContours(small_gray,contours1,maxj,(255,0,0),1)
            ellipse = cv2.fitEllipse(contours1[maxj])
            cv2.ellipse(small_gray,ellipse,(0,0,255),2)
            stx=100.0*centerx.std()/min(ellipse[1][0],ellipse[1][1])
            sty=float(100.0*centery.std()/(ellipse[1][0]+ellipse[1][1]))
            file1.write("     X-Position=%s"%(int(ellipse[0][0]+eye_pos[1])))
            file1.write("     Y-Position=%s"%(int(ellipse[0][1]+eye_pos[0])))
            file1.write("     Radius1=%s"%(ellipse[1][0]))
            file1.write("     Radius2=%s"%(ellipse[1][1]))
            file1.write("     Angle_pupil=%s"%(ellipse[2]))
            file1.write("     X_ROI=%s"%(eye_pos[1]))
            file1.write("     Y_ROI=%s"%(eye_pos[0]))
            file1.write("     STD_X=%s"%(centerx.std()))
            file1.write("     STD_Y=%s"%(centery.std()))
            file1.write("     R2STD_X=%s"%(r2centerx.std()))
            file1.write("     R2STD_Y=%s"%(r2centery.std()))
            file1.write("     R2STD_rad_min=%s"%(r2minrad.std()))
            file1.write("     R2STD_rad_maj=%s"%(r2majrad.std()))
            file1.write("     R2STD_angle=%s"%(r2angle.std()))
            file1.write("\n")
        else:
            print("No ellipse found")
            file1.write("NONE \n")

    else:
        file1.write("NONE \n")

    #cv2.imshow('frame', gray)
    re = fr_count%tt
    f_count = fr_count//tt
    if re == 0:
        name = "images/img%06d.bmp" % (f_count,)
        print("Writing file for ", f_count)
        cv2.imwrite(name,gray)
    #cv2.imshow('frame2', thres1)
    #out.write(gray)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
file1.close()
cv2.destroyAllWindows()
