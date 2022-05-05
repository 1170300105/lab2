import math

import numpy as np
import OneDimSearch
import matplotlib.image as mpi
import cv2


def prox_2d(X, lmbd, loop_times=4):
    Y = X.copy()
    P = np.zeros(X.shape)
    Q = np.zeros(X.shape)
    for _ in range(loop_times):
        if _ % 2 == 0:
            Z = np.zeros(X.shape)
            for col_idx in range(X.shape[1]):
                Z[:, col_idx] = prox_1d(np.mat(Y[:, col_idx] + P[:, col_idx]).T, lmbd).T
            P = P + Y - Z
            for row_idx in range(X.shape[0]):
                Y[row_idx, :] = prox_1d(np.mat(Z[row_idx, :] + Q[row_idx, :]), lmbd)
            Q = Q + Z - Y
        else:
            for row_idx in range(X.shape[0]):
                Z[row_idx, :] = prox_1d(np.mat(Y[row_idx, :] + P[row_idx, :]), lmbd)
            P = P + Y - Z
            for col_idx in range(X.shape[1]):
                Y[:, col_idx] = prox_1d(np.mat(Z[:, col_idx] + Q[:, col_idx]).T, lmbd).T
            Q = Q + Z - Y
    return Y


def prox_1d(X, lmbd):
    Y = X.copy()
    f = f_1d(X, lmbd)
    for i in range(100):
        d = f.g(Y) * -1
        f_lmbd = fun_lmbd(f, Y, d)
        area = OneDimSearch.search_area_get(f_lmbd, 100, 0.5, 0)
        step = OneDimSearch.golden_partition_search(f_lmbd, area, 0.001)
        if abs(np.linalg.norm(d) * step) < 1:
            break
        Y = Y + step * d
    return Y


class f_1d:
    def __init__(self, x0, lmbd):
        self.x0 = x0
        self.lmbd = lmbd
        self.d_n = np.diag([1] * x0.shape[1])
        for i in range(x0.shape[1] - 1):
            self.d_n[i][i + 1] = -1

    def v(self, y):
        return np.linalg.norm(y - self.x0) ** 2 / 2 + self.lmbd * np.linalg.norm((self.d_n * y.T), ord=1)

    def g(self, y):
        g = y - self.x0
        g2 =[]
        for i in range(y.shape[1]):
            gt = 0
            if i > 0:
                gt += self.lmbd if y[0,i] - y[0,i - 1] > 0 else -1 * self.lmbd
            if i < y.shape[1] - 1:
                gt += self.lmbd if y[0,i] - y[0,i + 1] > 0 else -1 * self.lmbd
            g2.append(gt)
        g += np.mat(g2)
        return g


class fun_lmbd:
    def __init__(self, f, x0, d):
        self.f = f
        self.x0 = x0
        self.d = d

    def v(self, lmbd):
        return self.f.v(self.x0 + lmbd * self.d)


class TVLayer:
    def __init__(self, lmbd):
        self.lmbd = lmbd

    def forward(self, X):
        return prox_2d(X, self.lmbd)


def read_gray_img(img_path):
    image_mat = cv2.imread(img_path, 0).astype(np.float)
    return image_mat


def save_img(img_path, mat):
    cv2.imwrite(img_path, mat)


def add_noise(mat, sigma=20, mean=0):
    return mat + np.random.normal(mean, sigma, size=mat.shape)


if __name__ == '__main__':
    origin_mat = read_gray_img('test.jpg').T
    save_img('origin.jpg', origin_mat)
    print(origin_mat.shape)
    noise_mat = add_noise(origin_mat)
    print(np.linalg.norm(noise_mat - origin_mat))
    save_img('noise.jpg', noise_mat)
    itr_mat = noise_mat.copy()
    # itr_mat = origin_mat.copy()
    lmbd = 0.1
    offset_lmbd = 0.1
    for _ in range(50):
        tvl = TVLayer(math.log(1+math.exp(lmbd)))
        itr_mat = tvl.forward(itr_mat)
        print(np.linalg.norm(itr_mat - origin_mat))
        save_img('itr/itr'+str(_)+'.jpg', itr_mat.T)
        lmbd += offset_lmbd
