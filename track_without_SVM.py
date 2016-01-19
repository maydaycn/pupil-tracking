from __future__ import print_function
import argparse
import math
import random
import h5py
import numpy as np
import cv2
#import pandas as pd
from sklearn.externals import joblib

from tracklib import extract_patches, PatchSelector


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
parser.add_argument('-R', '--ransac_trials', type=int, default=100, help='Number of times to pick points and calculate standard deviation')
args = parser.parse_args()
# ----------------------------------

#eye_selector = PatchSelector(args.eye_svm, args.eyefile)

file1=open("trace.txt",'w')
cap = cv2.VideoCapture(args.videofile)
#trace=pd.DataFrame(columns=['x_pos','y_pos','rad_min','rad_maj','angle','x_roi','y_roi','std_x','std_y','std_rad_min','std_rad_min','std_angle'])
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
    #eye_pos = eye_selector.get_pos(gray)
    #eye_pos=[270,550] #x and y coordinate are given in reverse order this is for video 151230
    #eye_pos=[420,900]
    eye_pos=[args.y_roi, args.x_roi]
    full_patch_size=args.full_patch_size


    if eye_pos is not None:
        #START HERE JUGNU... implementation 2

        #cv2.rectangle(gray, tuple(eye_pos[::-1]), tuple( eye_pos[::-1]+full_patch_size), (255,0,0), 3)
        small_gray = gray[eye_pos[0]:eye_pos[0]+full_patch_size,
                            eye_pos[1]:eye_pos[1]+full_patch_size]
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
                if ((maxr < radius1) and (center1[1] > 0.25*full_patch_size) and (center1[1] < 0.75*full_patch_size) and (center1[0] > 0.25*full_patch_size) and (center1[0] < 0.75*full_patch_size) and (radius1 >5) and (radius1 <180) and len(contours1[j]) >= 5):
                    maxr=radius1
                    maxc=center1
                    maxj=j
                #cv2.circle(thres1,center1,radius1,(0,255,0),thickness=5)
            #print("Radius= ",maxr)
            break
        #from IPython import embed
        #embed()
        cv2.rectangle(gray, tuple(eye_pos[::-1]), tuple( [x+full_patch_size for x in eye_pos[::-1]]), (255,0,0), 3)
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

            #RANSAC2 implementation starts
            r2centerx=[]
            r2centery=[]
            r2majrad=[]
            r2minrad=[]
            r2angle=[]
            for i in range(args.ransac_trials):
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
                    r2centerx.append(100*(i%2))
                    r2centery.append(100*(i%2))
                    r2majrad.append(100*(i%2))
                    r2minrad.append(100*(i%2))
                    r2angle.append(100*(i%2))
            r2centerx=np.asarray(r2centerx)
            r2centery=np.asarray(r2centery)
            r2majrad=np.asarray(r2majrad)
            r2minrad=np.asarray(r2minrad)
            r2angle=np.asarray(r2angle)
            #RANSAC2 implementation ends

            #RANSAC 3
            '''
            ellipse = cv2.fitEllipse(contours1[maxj])
            centx=ellipse[0][0]
            centy=ellipse[0][1]
            angle=ellipse[2]
            i=0
            g_GOF=0
            for coord in contours1[maxj]:
                posx = (coord[0][0] - centx) * math.cos(-angle) - (coord[0][1]- centy) * math.sin(-angle)
                posy = (coord[0][0] - centx) * math.sin(-angle) + (coord[0][1]- centy) * math.cos(-angle)
                #embed()
                temp = abs(pow(pow(posx/ellipse[1][0],2) + pow(posy/ellipse[1][1],2) - 0.25,2))
                g_GOF += temp
                print("Error of point=",temp)
            g_GOF=g_GOF/len(contours[maxj])
            '''
            #RANSAC3 ends


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
            #file1.write("     g_GOF=%s"%(g_GOF))
            ##trace=pd.DataFrame(columns=['x_pos','y_pos','rad_min','rad_maj','angle','x_roi','y_roi','std_x','std_y','std_rad_min','std_rad_min','std_angle'])
            #trace.loc[len(trace)+1] = (int(ellipse[0][0]+eye_pos[1]),int(ellipse[0][1]+eye_pos[0]),ellipse[1][0],ellipse[1][1],ellipse[2],
            #                        eye_pos[1],eye_pos[0],r2centerx.std(),r2centery.std(),r2minrad.std(),r2majrad.std(),r2angle.std())
            file1.write("\n")
        else:
            print("No ellipse found")
            file1.write("NONE \n")
            #trace.loc[len(trace)+1] = 12*(NONE,)

    else:
        file1.write("NONE \n")
        #trace.loc[len(trace)+1] = 12*(NONE,)

    re = fr_count%tt
    f_count = fr_count//tt
    if re == 0:
        name = "images/img%06d.bmp" % (f_count,)
        print("Writing file for ", f_count)
        cv2.imwrite(name,gray)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
file1.close()
trace.to_csv('trace.csv')
cv2.destroyAllWindows()
