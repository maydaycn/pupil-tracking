import argparse
import numpy as np
import cv2
import pandas as pd
from IPython import embed


class PupilTracker:
    def ransac(self, ntrials, contour, small_gray, draw):
        # RANSAC2 implementation starts
        r2centerx = []
        r2centery = []
        r2majrad = []
        r2minrad = []
        r2angle = []
        for i in range(ntrials):
            if len(contour) > 60:
                # embed()
                samples = contour[np.random.choice(len(contour), int(len(contour) / 10))]
                ellipse = cv2.fitEllipse(samples)
                if draw:
                    cv2.ellipse(small_gray, ellipse, (0, 0, 255), 2)
                r2centerx.append(ellipse[0][0])
                r2centery.append(ellipse[0][1])
                r2majrad.append(ellipse[1][1])
                r2minrad.append(ellipse[1][0])
                r2angle.append(ellipse[2])
            else:
                r2centerx.append(100 * (i % 2))
                r2centery.append(100 * (i % 2))
                r2majrad.append(100 * (i % 2))
                r2minrad.append(100 * (i % 2))
                r2angle.append(100 * (i % 2))
        r2centerx = np.asarray(r2centerx)
        r2centery = np.asarray(r2centery)
        r2majrad = np.asarray(r2majrad)
        r2minrad = np.asarray(r2minrad)
        r2angle = np.asarray(r2angle)
        return r2centerx, r2centery, r2majrad, r2minrad, r2angle, small_gray
        # RANSAC2 implementation ends

    def track_without_svm(self, videofile, eye_roi, ransac_trials = 100):

        # x_roi, y_roi, full_patch_size=350, ransac_trials=100):

        draw_image = 0
        print("Starting for video", videofile)
        cap = cv2.VideoCapture(videofile)
        trace = pd.DataFrame(
            columns=['pupil_x', 'pupil_y', 'pupil_r_minor', 'pupil_r_major', 'pupil_angle', 'pupil_x_std',
                     'pupil_y_std', 'pupil_r_minor_std', 'pupil_r_major_std', 'pupil_angle_std', 'intensity_std'])
        leng = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # use tt to indicate AFTER how many image do you want to locally save the image
        tt = 9999999
        fr_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            fr_count += 1
            if fr_count % 500 == 0:
                print("fr_count=", fr_count)
                print("Total frames = ", leng)
            if fr_count >= (leng):
                print("Video: ", videofile, " is over")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # eye_pos is the eye centre which is converted to patch starting pixel index below
            # full_patch_size = full_patch_size
            #eye_pos = [y_roi - full_patch_size / 2, x_roi - full_patch_size / 2]
            eye_pos = [eye_roi[1], eye_roi[0]]
            full_patch_size = max(eye_roi[2]-eye_roi[0], eye_roi[3]-eye_roi[1])
            # print("Eye_pos =", eye_pos, "full_patch_size=", full_patch_size)


            #small_gray = gray[eye_pos[0]:eye_pos[0] + full_patch_size,
            #             eye_pos[1]:eye_pos[1] + full_patch_size]

            small_gray = gray[eye_roi[1]:eye_roi[3], eye_roi[0]:eye_roi[2]]
            cv2.medianBlur(small_gray, 7, small_gray)
            variation = np.std(small_gray)
            th = 0.5 * (np.percentile(small_gray.ravel(), 99)) + 0.5 * (np.percentile(small_gray.ravel(), 1))
            # flag = 0

            _, thres = cv2.threshold(small_gray, th, 255, cv2.THRESH_BINARY)
            _, contours1, hierarchy1 = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            maxc = None
            maxr = 0
            draw = 0
            for j in range(len(contours1)):
                cnt1 = contours1[j]
                (x, y), radius1 = cv2.minEnclosingCircle(cnt1)
                center1 = (int(x), int(y))
                radius1 = int(radius1)
                if ((maxr < radius1 - 0.05 * (center1[0] - full_patch_size / 2) - 0.05 * (
                            center1[1] - full_patch_size / 2)) and (center1[1] > 0.20 * full_patch_size) and (
                            center1[1] < 0.80 * full_patch_size) and (center1[0] > 0.20 * full_patch_size) and (
                            center1[0] < 0.80 * full_patch_size) and (radius1 > 5) and (radius1 < 180) and len(
                    contours1[j]) >= 5):
                    maxr = radius1 - 0.05 * (center1[0] - full_patch_size / 2) - 0.05 * (
                        center1[1] - full_patch_size / 2)
                    maxc = center1
                    maxj = j
            if draw:
                cv2.rectangle(gray, tuple([int(float(x)) for x in eye_pos[::-1]]),
                              tuple([int(float(x)) + full_patch_size for x in eye_pos[::-1]]), (255, 0, 0), 3)
            if maxc is not None:
                draw = 1 * draw_image
                r2centerx, r2centery, r2majrad, r2minrad, r2angle, small_gray = self.ransac(ransac_trials,
                                                                                            contours1[maxj],
                                                                                            small_gray, draw)
                cv2.drawContours(small_gray, contours1, maxj, (255, 0, 0), 1)
                ellipse = cv2.fitEllipse(contours1[maxj])
                cv2.ellipse(small_gray, ellipse, (0, 0, 255), 2)

                trace.loc[len(trace) + 1] = (
                    float(ellipse[0][0] + eye_pos[1] + 1), float(ellipse[0][1] + eye_pos[0] + 1),
                    float(ellipse[1][0]),
                    float(ellipse[1][1]), float(ellipse[2]),
                    float(r2centerx.std()), float(r2centery.std()), float(r2minrad.std()), float(r2majrad.std()),
                    float(r2angle.std()), float(variation))
            else:
                print("No ellipse found")
                trace.loc[len(trace) + 1] = 11 * (None,)

            re = fr_count % tt
            if re == 0 and draw_image:
                name = "images/img%06d.png" % (fr_count,)
                cv2.imwrite(name, gray)
            if (cv2.waitKey(1) & 0xFF == ord('q')) or fr_count >= 999999:
                break

        cap.release()
        trace.to_csv('trace.csv', index=False)
        cv2.destroyAllWindows()
        return trace
