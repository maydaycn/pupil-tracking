from __future__ import print_function
import argparse
import warnings
from scipy.spatial.distance import pdist

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
import h5py
import numpy as np
from sklearn.externals import joblib

def compute_crange(K, basefactors=2 ** np.arange(-3, 4.)):
    """
    Estimates a good range for C based on the inverse variance in features space
    of the kernel with kernel matrix K.

    See also:
    Chapelle, O., & Zien, A. (2005). Semi-Supervised Classification by Low Density Separation.

    :param K: kernel matrix
    :param basefactors: factors that get multiplied with the inverse variance to get a good range for C
    :returns: basefactors/estimated variance in feature space

    """
    s2 = np.mean(np.diag(K)) - np.mean(K.ravel())
    if s2 == 0.:
        warnings.warn("Variance in feature space is 0. Using 1!")
    return basefactors / s2


# ----------------------------------
parser = argparse.ArgumentParser(description='Train SVM for pupil recognition')
parser.add_argument('datafile', metavar='datafile', type=str,
                    help='Datafile collected with generate_training_data.py ')
parser.add_argument('svmfile', metavar='outfile', type=str,
                    help='File where joblib saves the SVM to (*.pkl)')

args = parser.parse_args()
# ----------------------------------

with h5py.File(args.datafile, 'r') as fid:
    X_train = np.asarray(fid['X'], dtype=float)
    X_train /= 255
    y_train = np.asarray(fid['y'], dtype=float)

gamma = 1 / np.median(pdist(X_train, 'euclidean'))
C = compute_crange(rbf_kernel(X_train, gamma=gamma))

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': list(2 ** np.arange(-3, 4.) * gamma), 'C': list(C),
                     'class_weight': [{1: int((y_train == -1).sum() / (y_train == 1).sum())}]},
                    ]

clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='accuracy', n_jobs=10, verbose=True)
clf.fit(X_train, y_train)

best = clf.best_estimator_
print("Best estimator has training accuracy of %.4g" % clf.best_score_)
joblib.dump(best, args.svmfile)
