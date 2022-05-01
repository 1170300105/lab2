import matplotlib.image as mpi
import random
import numpy as np


def make_jigsaw_puzzles(image_path):
    pieces, shape, K = read_pieces(image_path)
    random.shuffle(pieces)
    mat = []
    for i in range(int(shape[0] / K)):
        mat.append(np.concatenate(pieces[i * int(shape[1] / K):int((i + 1) * (shape[1] / K))], axis=1))
    mat = np.concatenate(mat, axis=0)
    print(mat.shape)
    mpi.imsave('piece.jpg', mat)
    return pieces


def read_pieces(pieces_path):
    image_mat = mpi.imread(pieces_path)
    print(image_mat.shape)
    total_shape = image_mat.shape
    K = 25
    pieces = []
    for i in range(int(total_shape[0] / K)):
        for j in range(int(total_shape[1] / K)):
            pieces.append(image_mat[i * K:(i + 1) * K, j * K:(j + 1) * K, :])
    return pieces, total_shape, K


def get_MN(pieces, K):
    total_size = len(pieces)
    mn_r = np.arange(total_size ** 2).reshape(total_size, total_size)
    mn_d = np.arange(total_size ** 2).reshape(total_size, total_size)
    for i in range(total_size):
        for j in range(total_size):
            p1, p2 = pieces[i], pieces[j]
            mn_r[i][j] = np.linalg.norm(p1[:, K - 1, :] - p2[:, 0, :])
            mn_d[i][j] = np.linalg.norm(p1[K - 1, :, :] - p2[0, :, :])
    return mn_r, mn_d


def get_score(mn_r, mn_d, ans, total_shape):
    score = 0
    for i in range(ans.shape[0] - 1):
        for j in range(ans.shape[1] - 1):
            score += mn_r[i][j]
            score += mn_d[i][j]
    return score


if __name__ == '__main__':
    pieces, shape, K = read_pieces('piece.jpg')
    r, d = get_MN(pieces, K)
