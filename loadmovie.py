import numpy as np
import cv2
import matplotlib.pyplot as plt


def frst(img, radii, alpha=2., std_factor=0.25, k_n = 9.9, orientation_based=False, beta=2):
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

    n_inv = 1./len(radii)
    for r_idx, n in enumerate(radii):
        O *= 0
        M *= 0

        gp = np.round(g_dir * n).astype(int)

        for sgn in [-1, 1]: # TODO make that a swtich
            ppve = p + sgn * gp + max_radius
            O[ppve[idx, 1], ppve[idx, 0]] += sgn
            M[ppve[idx, 1], ppve[idx, 0]] += sgn*g_norm[idx]
        # k_n = np.abs(O).max()/2
        k_idx = O >= k_n
        O[k_idx] = k_n
        if not orientation_based:
            F = M/k_n * (np.abs(O)/k_n)**alpha
        else:
            F = np.sign(O) * (np.abs(O)/k_n)**alpha

        ksize = 2 * (int(np.ceil(n / 2) + (np.ceil(n / 2) % 2 == 0)),)
        S += n_inv * cv2.GaussianBlur(F, ksize, n * std_factor)
    return S[max_radius:-max_radius, max_radius:-max_radius]


cap = cv2.VideoCapture('demo.avi')

thres = 10
kernel = np.ones((5,5))
I, J = np.meshgrid(
    np.arange(640),
    np.arange(480))


while (cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)
    _, im_bw = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY_INV )
    # _, im_bw = cv2.threshold(gray, thres, 255, cv2.THRESH_TRUNC )

    # cv2.medianBlur(im_bw, 3, im_bw)
    # im_bw = gray

    S = frst(im_bw, np.arange(5, 30, 2), alpha=2.8, std_factor=.8, beta=80)
    p = np.percentile(S.ravel(), 99.9)
    S[S<p] = 0
    S = cv2.erode(S,kernel,iterations = 1)
    S = S/S.sum()
    x = int((I*S).sum())
    y = int((J*S).sum())
    cv2.circle(gray, (x,y), 5, (255,0,0), thickness=2, lineType=8, shift=0)
    cv2.imshow('frame', gray)
    # cv2.imshow('frame', 255-im_bw)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
