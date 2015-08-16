import numpy as np
import cv2
import matplotlib.pyplot as plt


def frst(img, radii, bright=True, alpha=2., std_factor=0.1):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    g = np.concatenate((sobelx[..., None], sobely[..., None]), axis=2)
    g_norm = np.sqrt((g ** 2).sum(axis=2))

    max_radius = int(np.ceil(radii.max()))
    filtered = np.zeros((img.shape[0] + 2 * max_radius, img.shape[1] + 2 * max_radius))
    S = np.zeros_like(filtered)
    O = np.zeros_like(filtered)
    M = np.zeros_like(filtered)

    idx = g_norm > 0
    gp = np.zeros_like(g)
    g_dir = g / g_norm[..., None]
    idx = ~np.isnan(g_dir[..., 0]) & ~np.isnan(g_dir[..., 1])
    I, J = np.meshgrid(
        np.arange(img.shape[1], dtype=int),
        np.arange(img.shape[0], dtype=int))
    p = np.concatenate((I[..., None], J[..., None]), axis=2)
    for r_idx, n in enumerate(radii):
        O *= 0
        M *= 0

        gp = np.round(g_dir * n).astype(int)

        for sgn in [-1,1]:
            ppve = p + sgn * gp + max_radius
            O[ppve[idx, 1], ppve[idx, 0]] += sgn
            M[ppve[idx, 1], ppve[idx, 0]] += sgn * g_norm[idx]

        O = np.abs(O)
        O /= O.max()

        M = np.abs(M)
        M /= M.max()
        ksize = 2 * (int(np.ceil(n / 2) + (np.ceil(n / 2) % 2 == 0)),)
        S += cv2.GaussianBlur(O ** alpha * M, ksize, n * std_factor)
    return S[max_radius:-max_radius, max_radius:-max_radius]


cap = cv2.VideoCapture('demo.avi')

aperture = 5
thres = 50

windowClose = np.ones((5, 5), np.uint8)
windowOpen = np.ones((2, 2), np.uint8)
windowErode = np.ones((2, 2), np.uint8)

while (cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.equalizeHist(gray, gray)
    # _, im_bw = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY )
    # im_bw = gray
    # cv2.medianBlur(im_bw, aperture, im_bw)
    # dev = cv2.Sobel(im_bw, cv2.CV_32F, 0, 2)
    S = frst(gray, np.arange(100, 110, 10), bright=False)
    cv2.imshow('frame', S)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
