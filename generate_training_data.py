from __future__ import print_function
import argparse
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from tracklib import sample_random_patches, extract_patch, PatchSelector


# -----------------------------------------------------------

parser = argparse.ArgumentParser(description='Collect pupil training data from video')
parser.add_argument('videofile', metavar='videofile', type=str,
                    help='Videofile with showing the eye. ')
parser.add_argument('outfile', metavar='outfile', type=str,
                    help='Output file for the training dataset.')
parser.add_argument('-q', '--downsample-factor', type=int, default=30, help='downsample factor (default 30)')
parser.add_argument('-P', '--patch-size', type=int, default=10, help='patch size (default 10)')

parser.add_argument('-p', '--positives', type=int, default=200,
                    help='numper of positive examples to collect (default 200)')
parser.add_argument('-n', '--negative-factor', type=int, default=5,
                    help='factor of negatives per positive example (default 5)')
parser.add_argument('-s', '--stride', type=int, default=50, help='look at every stride frame (default 50)')
parser.add_argument('-T', '--start', type=int, default=1, help='starting frame')

parser.add_argument('-S', '--svm-file', type=str, default=None, help='SVM file for preselection')
parser.add_argument('-D', '--svm-data', type=str, default=None, help='Datafile for the SVM.')

args = parser.parse_args()
# -----------------------------------------------------------

assert args.svm_file is not None or args.svm_data is None, 'If --svm-file is set, you must set --svm-data as well.'


if args.svm_file is not None:
    patch_selector = PatchSelector(args.svm_file, args.svm_data)
else:
    patch_selector = lambda img: img

X = []
y = []


q = args.downsample_factor
patch_size = args.patch_size

fig, ax = plt.subplots() #jugnu what is this for

cap = cv2.VideoCapture(args.videofile)
leng = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
print("Number of frames=",leng)

stride = args.stride
if (args.stride==0):
    stride= leng/args.positives;
    print("Auto Stride =",stride)

#quit()
i = 0
ii=0
positives = 0
x = None

while (cap.isOpened()) and positives < args.positives:
    ret, frame = cap.read()
    #print("Entered loop and ii is=",ii)
    i += 1
    ii+= 1
    if ii >= (leng-1):
        break

    if ii < args.start:
        continue

    if i < stride:
        continue
    else:
        i = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("Frame number is=",ii)
    #print("Debug2: Entered loop and i is=",i)
    #print("Calculating patch for i= ", i)
    gray = patch_selector(gray) # TODO this doesn't make any sense at all
    if gray is None:
        continue
    else:
        gray = gray[::q, ::q]
    #print("Calculated done for i=" ,i)
    ax.imshow(gray, cmap=plt.cm.gray)
    if x is not None:
        psh = patch_size / 2
        ax.fill_between([x[0] - psh, x[0] + psh], [x[1] - psh, x[1] - psh], [x[1] + psh, x[1] + psh],
                        color='lime', alpha=.1)

    ax.set_title('Click on the center of the eye! Middle mouse button for skip. %i/%i' % (positives, args.positives))
    plt.draw()

    x = plt.ginput(1) #jugnu to take the input from mouse 1 time
    print("input_mouse_len=",len(x))
    if len(x) == 0:
        print ("No input found.")
	x=None
    else:
        print("Input recorded for i=",i)
        x = x[0]
        xp = extract_patch(x, gray, patch_size)

        if xp is None:
            print ("""Could not extract patch because it is too close to the boundary.
                    Consider decreasing the downsample factor or decreasing the patch size""")
            continue
        else:
            print("Patch extracted for i=",i)
            X.append(xp)
            y.append(1.)
            positives += 1

    X.extend(sample_random_patches(gray, patch_size, args.negative_factor)) #jugnu how is negative factor used?
    y.extend(args.negative_factor * [-1.])

    plt.draw()
    ax.clear()
    print("data appended and now going for next image for i=",i)

with h5py.File(args.outfile, 'w') as fid:
    X = np.vstack(X)
    y = np.asarray(y)
    fid.create_dataset('X', X.shape, data=X, compression='gzip')
    fid.create_dataset('y', y.shape, data=y, compression='gzip')

    for key in [k for k in dir(args) if not k.startswith('_')]:
        if getattr(args, key) is not None:
            fid.attrs[key] = getattr(args, key)
