import math

import matplotlib.image as mpi
import random
import numpy as np


def make_jigsaw_puzzles(image_path):
    pieces, shape, K = read_pieces(image_path)
    mn_r, mn_d, best_r, best_d = get_MN(pieces, K)
    no_shuffle = np.arange(shape[0] * shape[1]).reshape(shape)
    for i in range(len(pieces)):
        mpi.imsave('piece/piece' + str(i) + '.jpg', pieces[i].astype(np.uint8))
    random.shuffle(pieces)
    mat = []
    for i in range(shape[0]):
        mat.append(np.concatenate(pieces[i * shape[1]:int((i + 1) * shape[1])], axis=1))
    mat = np.concatenate(mat, axis=0).astype(np.uint8)
    print(mat.shape)
    mpi.imsave('piece.jpg', mat)
    return pieces


def read_pieces(pieces_path):
    image_mat = mpi.imread(pieces_path).astype(np.int)
    print(image_mat.shape)
    total_shape = image_mat.shape
    K = 100
    pieces = []
    for i in range(int(total_shape[0] / K)):
        for j in range(int(total_shape[1] / K)):
            pieces.append(image_mat[i * K:(i + 1) * K, j * K:(j + 1) * K, :])
    return pieces, (int(total_shape[0] / K), int(total_shape[1] / K)), K


def get_MN(pieces, K):
    total_size = len(pieces)
    mn_r = np.arange(total_size ** 2).reshape(total_size, total_size)
    mn_d = np.arange(total_size ** 2).reshape(total_size, total_size)
    for i in range(total_size):
        for j in range(total_size):
            p1, p2 = pieces[i], pieces[j]
            mn_r[i][j] = np.linalg.norm(p1[:, K - 1, :] - p2[:, 0, :])
            mn_d[i][j] = np.linalg.norm(p1[K - 1, :, :] - p2[0, :, :])
    best_match_r = mn_r.argsort(axis=1)
    best_match_d = mn_d.argsort(axis=1)
    return mn_r, mn_d, best_match_r, best_match_d


def get_score(mn_r, mn_d, ans):
    score = 0
    for i in range(ans.shape[0] - 1):
        for j in range(ans.shape[1] - 1):
            score += mn_r[ans[i][j]][ans[i][j + 1]]
            score += mn_d[ans[i][j]][ans[i + 1][j]]
    return score


def GA(mn_r, mn_d, total_shape, best_match_r, best_match_d):
    population_limit = 1500
    generation_num = 1500
    child_num = 1000
    new_random = 100  # 完全随机生成
    copy_num = 100  # 自我不完全复制
    save_num = 900  # 每次保存的数量
    population = []
    for i in range(population_limit):
        p = list(range(total_shape[0] * total_shape[1]))
        random.shuffle(p)
        p = np.array(p).reshape(total_shape[0], total_shape[1])
        population.append(p)
    for g in range(1, generation_num):
        print('gen' + str(g))
        population.sort(key=lambda x: get_score(mn_r, mn_r, x))
        print(get_score(mn_r, mn_r, population[0]))
        new_population = population[0: save_num - 1]
        for i in range(new_random):
            p = list(range(total_shape[0] * total_shape[1]))
            random.shuffle(p)
            p = np.array(p).reshape(total_shape[0], total_shape[1])
            new_population.append(p)
        children = []
        while len(children) < child_num:
            parent1 = random.randint(0, len(population) - 1)
            parent2 = random.randint(0, len(population) - 1)
            if random.randint(1, 10) > 8 - int(math.log(i)):
                parent1 = random.randint(0, 10)
            if random.randint(1, 10) > 8 - int(math.log(i)):
                parent2 = random.randint(0, 10)
            if parent1 == parent2:
                continue
            else:
                parent1 = population[parent1]
                parent2 = population[parent2]
                child = cross_over(parent1, parent2, mn_r, mn_d, best_match_r, best_match_d)
                children.append(child)
        children.sort(key=lambda x: get_score(mn_r, mn_r, x))
        new_population.extend(children[0:population_limit - copy_num - save_num])
        for i in range(copy_num):
            parent = population[random.randint(0, 10)]
            child = cross_over_copy(parent, g, best_match_r, best_match_d)
            new_population.append(child)
        population = new_population
    population.sort(key=lambda x: get_score(mn_r, mn_r, x))
    return population[0:3]


def cross_over_copy(parent, g, best_match_r, best_match_d):
    is_usd_piece = [False] * (parent.shape[0] * parent.shape[1])
    child = np.ones((parent.shape[0], parent.shape[1]), dtype=int)*-1
    offset_r = random.randint(0, min(1 + 2 * int(math.log(g)), parent.shape[0] - 1))
    offset_l = random.randint(0, min(1 + 2 * int(math.log(g)), parent.shape[1] - 1))
    random_copy_loss = random.randint(0, 2)
    for i in range(parent.shape[0]):
        for j in range(parent.shape[1]):
            off_i = (i + offset_r) % parent.shape[0]
            off_j = (j + offset_l) % parent.shape[1]
            if i != 0 and j != 0 and ((i < random_copy_loss) or (j < random_copy_loss)):
                continue
            child[i][j] = parent[off_i][off_j]
            is_usd_piece[child[i][j]] = True
    for i in range(parent.shape[0]):
        for j in range(parent.shape[1]):
            if child[i][j] == -1:
                choices = []
                if i >= 1:
                    if child[i - 1][j] != -1:
                        for p in best_match_d[child[i - 1][j], :]:
                            if not is_usd_piece[p]:
                                choices.append(p)
                                break
                if j >= 1:
                    if child[i][j - 1] != -1:
                        for p in best_match_r[child[i][j - 1], :]:
                            if not is_usd_piece[p]:
                                choices.append(p)
                                break
                if len(choices) == 1:
                    child[i][j] = choices[0]
                    is_usd_piece[child[i][j]] = True
                elif len(choices) == 2:
                    s1, s2 = 0, 0
                    if i >= 1:
                        if child[i - 1][j] != -1:
                            s1 += mn_d[child[i - 1][j]][choices[0]]
                            s2 += mn_d[child[i - 1][j]][choices[1]]
                    if j >= 1:
                        if child[i][j - 1] != -1:
                            s1 += mn_r[child[i][j - 1]][choices[0]]
                            s2 += mn_r[child[i][j - 1]][choices[1]]
                    if s1 >= s2:
                        child[i][j] = choices[1]
                        is_usd_piece[child[i][j]] = True
                    else:
                        child[i][j] = choices[0]
                        is_usd_piece[child[i][j]] = True
    rest = []
    for i in range(len(is_usd_piece)):
        if not is_usd_piece[i]:
            rest.append(i)
    random.shuffle(rest)
    count = 0
    for i in range(parent.shape[0]):
        for j in range(parent.shape[1]):
            if child[i][j] == -1:
                child[i][j] = rest[count]
                count += 1
    if count != len(rest):
        print("not valid")
        print(rest)
        print(count)
    return child


def cross_over(parent1, parent2, mn_r, mn_d, best_match_r, best_match_d):
    is_usd_piece = [False] * parent1.shape[0] * parent1.shape[1]
    child = np.ones((parent1.shape[0], parent1.shape[1]), dtype=int) * -1
    random_int = random.randint(0, 100)
    if random_int >= 80:
        offset_r = random.randint(0, 1)
        offset_l = random.randint(0, 1)
        for i in range(parent1.shape[0]):
            for j in range(parent1.shape[1]):
                child[i][j] = parent1[(i + offset_r) % parent1.shape[0]][(j + offset_l) % parent1.shape[1]]
                is_usd_piece[child[i][j]] = True
    elif random_int <= 20:
        offset_r = random.randint(0, 1)
        offset_l = random.randint(0, 1)
        for i in range(parent1.shape[0]):
            for j in range(parent1.shape[1]):
                child[i][j] = parent2[(i + offset_r) % parent2.shape[0]][(j + offset_l) % parent2.shape[1]]
                is_usd_piece[child[i][j]] = True
    else:
        for i in range(parent1.shape[0]):
            for j in range(parent1.shape[1]):
                if child[i][j] == -1 and parent1[i][j] == parent2[i][j]:
                    child[i][j] = parent1[i][j]
                    is_usd_piece[child[i][j]] = True
    for i in range(parent1.shape[0]):
        for j in range(parent1.shape[1]):
            if child[i][j] == -1:
                choices = []
                if i >= 1:
                    if child[i - 1][j] != -1:
                        for p in best_match_d[child[i - 1][j], :]:
                            if not is_usd_piece[p]:
                                choices.append(p)
                                break
                if j >= 1:
                    if child[i][j - 1] != -1:
                        for p in best_match_r[child[i][j - 1], :]:
                            if not is_usd_piece[p]:
                                choices.append(p)
                                break
                if len(choices) == 1:
                    child[i][j] = choices[0]
                    is_usd_piece[child[i][j]] = True
                elif len(choices) == 2:
                    s1, s2 = 0, 0
                    if i >= 1:
                        if child[i - 1][j] != -1:
                            s1 += mn_d[child[i - 1][j]][choices[0]]
                            s2 += mn_d[child[i - 1][j]][choices[1]]
                    if j >= 1:
                        if child[i][j - 1] != -1:
                            s1 += mn_r[child[i][j - 1]][choices[0]]
                            s2 += mn_r[child[i][j - 1]][choices[1]]
                    if s1 >= s2:
                        child[i][j] = choices[1]
                        is_usd_piece[child[i][j]] = True
                    else:
                        child[i][j] = choices[0]
                        is_usd_piece[child[i][j]] = True
    rest = []
    for i in range(len(is_usd_piece)):
        if not is_usd_piece[i]:
            rest.append(i)
    random.shuffle(rest)
    count = 0
    for i in range(parent1.shape[0]):
        for j in range(parent1.shape[1]):
            if child[i][j] == -1:
                child[i][j] = rest[count]
                count += 1
    if count != len(rest):
        print("not valid")
    return child


def save_ans(pieces, ans, shape, name):
    pic = []
    for i in range(shape[0]):
        p = pieces[ans[i][0]]
        for j in range(1, shape[1]):
            p = np.concatenate([p, pieces[ans[i][j]]], axis=1)
        pic.append(p)
    pic = np.concatenate(pic, axis=0).astype(np.uint8)
    print(pic.shape)
    mpi.imsave('ans' + name + '.jpg', pic)


if __name__ == '__main__':
    # make_jigsaw_puzzles('1.jpg')
    # pieces, shape, K = read_pieces('piece.jpg')
    # mn_r, mn_d, best_r, best_d = get_MN(pieces, K)
    # print(np.min(mn_r))
    # print(np.min(mn_d))
    # ans = GA(mn_r=mn_r, mn_d=mn_d, total_shape=shape, best_match_r=best_r, best_match_d=best_d)
    # for i in range(len(ans)):
    #     print(ans[i])
    #     save_ans(pieces, ans[i], shape, str(i))#
    pieces,shape,K = read_pieces('ans0.jpg')
    mn_r, mn_d, best_r, best_d = get_MN(pieces, K)
    no_shuffle = np.arange(shape[0] * shape[1]).reshape(shape)
    print(get_score(mn_r, mn_d, no_shuffle))
    pieces, shape, K = read_pieces('1.jpg')
    mn_r, mn_d, best_r, best_d = get_MN(pieces, K)
    no_shuffle = np.arange(shape[0] * shape[1]).reshape(shape)
    print(get_score(mn_r, mn_d, no_shuffle))

