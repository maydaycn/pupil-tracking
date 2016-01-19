import itertools
import warnings
import cv2
import h5py
import numpy as np
import scipy
from sklearn import preprocessing
from sklearn.externals import joblib


class PatchSelector:
    def __init__(self, svmfile, datafile, thin=1):
        self.svm = joblib.load(svmfile)
        with h5py.File(datafile, 'r') as fid:
            self.q = fid.attrs['downsample_factor'] #jugnu, when is this called and how this works
            self.patch_size = fid.attrs['patch_size']
        self.thin = thin

    def __call__(self, img):
        # q = self.q
        # patch_size = self.patch_size
        img_pos = self.get_pos(img)
        retval = None
        fps = self.full_patch_size #jugnu actual number of pixels without downsampling
        if img_pos is not None:
            start_i, start_j = img_pos
            retval = img[start_i:start_i + fps, start_j:start_j + fps]
        return retval

    def get_pos(self, img):
        q = self.q
        gray_small = img[::q, ::q]
        X, pos = extract_patches(gray_small, self.patch_size) #jugnu we get a column vector with elements as different box/ patch
        y = self.svm.decision_function(X) #jugnu will this work if svm is untrained? what values y can take?
        if np.any(y > 0):
            return pos[np.argmax(y), :]*q #jugnu is argmax taken to get the lowest  and most right image??
        else:
            return None

    @property
    def full_patch_size(self):
        return self.patch_size * self.q

def center_patches(X):
    return X - X.mean(axis=1)[:, np.newaxis]

def thresholding(X):
    for i, X_temp in enumerate(X):
        th=0.5*(np.percentile(X_temp,99)) + 0.5 *(np.percentile(X_temp,1))
        X_temp = scipy.stats.threshold(X_temp,threshmax=th-0.1, newval=1 )
        X_temp = scipy.stats.threshold(X_temp,threshmin=th, newval=0 )
        X[i]=X_temp
        #print(i)
    return X

def center_scale (X):
    Y = X - X.mean(axis=1)[:, np.newaxis]
    return preprocessing.scale(Y)



def extract_patches(img, patch_size, normalize=True, thin=1, preprocess=True):
    X = []
    pos = list(itertools.product(range(img.shape[0] - patch_size), range(img.shape[1] - patch_size))) #jugnu img.shape[1] for y cordinate??
    for i, j in pos:
        X.append(img[i:i + patch_size, j:j + patch_size].ravel())
    #return np.vstack(X) / (255. if normalize else 1.), np.vstack(pos)
    if preprocess:
        X_st = np.vstack(X) / (255. if normalize else 1.)
        #preprocessing the data before prediction by SVM
        X_pp = center_patches(X_st)
        #X_pp = center_scale(X_st)
        return np.vstack(X_pp), np.vstack(pos)
    else:
        return np.vstack(X) / (255. if normalize else 1.), np.vstack(pos)



def frst(img, radii, alpha=2., std_factor=0.25, k_n=9.9, orientation_based=False, beta=2):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    g = np.concatenate((sobelx[..., None], sobely[..., None]), axis=2)
    g_norm = np.sqrt((g ** 2).sum(axis=2))

    gn_thres = np.percentile(g_norm.ravel(), beta)

    max_radius = int(np.ceil(radii.max()))
    filtered = np.zeros((img.shape[0] + 2 * max_radius, img.shape[1] + 2 * max_radius))
    S = np.zeros_like(filtered)
    O = np.zeros_like(filtered)
    M = np.zeros_like(filtered)

    idx = g_norm > np.max([gn_thres, 0])
    g_dir = g / g_norm[..., None]

    I, J = np.meshgrid(
        np.arange(img.shape[1], dtype=int),
        np.arange(img.shape[0], dtype=int))
    p = np.concatenate((I[..., None], J[..., None]), axis=2)

    n_inv = 1. / len(radii)
    for r_idx, n in enumerate(radii):
        O *= 0
        M *= 0

        gp = np.round(g_dir * n).astype(int)

        for sgn in [-1, 1]:  # TODO make that a swtich
            ppve = p + sgn * gp + max_radius
            O[ppve[idx, 1], ppve[idx, 0]] += sgn
            M[ppve[idx, 1], ppve[idx, 0]] += sgn * g_norm[idx]
        # k_n = np.abs(O).max()/2
        k_idx = O >= k_n
        O[k_idx] = k_n
        if not orientation_based:
            F = M / k_n * (np.abs(O) / k_n) ** alpha
        else:
            F = np.sign(O) * (np.abs(O) / k_n) ** alpha

        ksize = 2 * (int(np.ceil(n / 2) + (np.ceil(n / 2) % 2 == 0)),)
        S += n_inv * cv2.GaussianBlur(F, ksize, n * std_factor)
    return S[max_radius:-max_radius, max_radius:-max_radius]


def sample_random_patches(img, patch_size, n):
    ret = []
    for n in range(n): # TODO: possibly exclude the eye
        i = np.random.randint(patch_size // 2 + 1, img.shape[1] - patch_size // 2 - 1)
        j = np.random.randint(patch_size // 2 + 1, img.shape[0] - patch_size // 2 - 1)
        ret.append(extract_patch((i, j), img, patch_size))
    return ret


def extract_patch(x, img, patch_size):
    start = np.round(np.asarray(x) - patch_size / 2).astype(int)

    x_sl, y_sl = slice(start[0], start[0] + patch_size), slice(start[1], start[1] + patch_size)

    if x_sl.start < 0 or x_sl.stop >= img.shape[1] or y_sl.start < 0 or y_sl.stop >= img.shape[0]: #jugnu is 0 and 1 not interchanged?
        warnings.warn("Cannot extract patch. Returning None.")
        return None
    else:
        return img[y_sl, x_sl].ravel()
