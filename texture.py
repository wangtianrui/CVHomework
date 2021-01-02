# coding: utf-8
# Author：WangTianRui
# Date ：2020/12/3 20:47
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import copy, os
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置


def plot_3d(x, y, z, img, title):
    fig = plt.figure()
    ax = Axes3D(fig)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
    # zdir 表示向那个轴投影
    ax.contourf(x, y, img, zdir='z', offset=np.max(z) + 0.5, cmap='gray')
    ax.set_title(title)
    plt.show()


def load_img(path):
    if os.path.exists(path):
        return np.array(mpimg.imread(path), dtype="float")
    else:
        print("图片路径错误")
        return


def show_gray_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def show_color_img(img):
    plt.imshow(img)
    plt.show()


def glcm(input_win, step_x, step_y, gray_level):
    w, h = input_win.shape
    ret = np.zeros([gray_level, gray_level])
    count = 0
    for index_h in range(h):
        for index_w in range(w):
            try:
                row = int(input_win[index_h][index_w])
                col = int(input_win[index_h + step_y][index_w + step_x])
                ret[row, col] += 1
                count += 1
            except Exception as e:
                continue
    return ret / count


def conv_smooth(array, ker_s=10):
    re = np.zeros_like(array)
    W, H, f_dim = array.shape
    padding = (ker_s - 1) // 2
    array = np.pad(array, ((padding, padding), (padding, padding), (0, 0)), "edge")
    for w in range(padding, W + padding):
        for h in range(padding, H + padding):
            re[w - padding][h - padding] = np.mean(
                array[w - padding:w + padding, h - padding:h + padding, :].reshape((-1, f_dim)), axis=0)
    return re


def glcm_conv(win_len, img, stride=(1, 1), gray_level=25, ker_s=5):
    f_dim = 6
    eps = np.array(1e-12)
    W, H = img.shape
    new_W = (W - (win_len - 1)) // stride[0]
    new_H = (H - (win_len - 1)) // stride[1]
    print(W, H)
    strides = img.itemsize * np.array([W * stride[1], stride[0], W, 1])
    print(strides)
    win_wrap_img = np.lib.stride_tricks.as_strided(img, shape=(new_W, new_H, win_len, win_len),
                                                   strides=strides)
    print("sliding window:", win_wrap_img.shape)
    # 灰度级调整
    max_gray = np.max(win_wrap_img) + 1
    if max_gray != max_gray:
        print(np.max(win_wrap_img))
    win_wrap_img = np.array(win_wrap_img * gray_level / max_gray, dtype="int")
    glcm_result = np.zeros((new_W, new_H, f_dim))
    # glcm_result = []

    for w in range(new_W):
        for h in range(new_H):
            glcm_4_direction = (glcm(input_win=win_wrap_img[w][h], step_x=1, step_y=0, gray_level=gray_level) +
                                glcm(input_win=win_wrap_img[w][h], step_x=0, step_y=1, gray_level=gray_level) +
                                glcm(input_win=win_wrap_img[w][h], step_x=1, step_y=1, gray_level=gray_level) +
                                glcm(input_win=win_wrap_img[w][h], step_x=-1, step_y=1, gray_level=gray_level)) / 4
            glcm_4_direction = glcm_4_direction / np.sum(glcm_4_direction)

            entropy = -np.sum(glcm_4_direction * np.log(glcm_4_direction + eps))
            energy = np.sum(glcm_4_direction ** 2)

            i_array = np.arange(gray_level).reshape((1, gray_level)).repeat(gray_level, 0)
            j_array = np.arange(gray_level).reshape((gray_level, 1)).repeat(gray_level, 1)
            inertia = np.sum((i_array - j_array) ** 2 * glcm_4_direction)
            mu_x = np.sum(i_array * glcm_4_direction)
            mu_y = np.sum(j_array * glcm_4_direction)
            delta_x = np.sum((i_array - mu_x) ** 2 * glcm_4_direction)
            delta_y = np.sum((j_array - mu_y) ** 2 * glcm_4_direction)
            corr = np.sum((i_array * j_array * glcm_4_direction - mu_x * mu_y) / (delta_x * delta_y))
            std_ = np.std(glcm_4_direction)

            homogeneity = np.sum(glcm_4_direction / (1 + (i_array - j_array) ** 2))

            # glcm_result[w][h] = np.array([corr, corr, corr, corr, corr, homogeneity], dtype="float")
            glcm_result[w][h] = np.array([entropy, energy, inertia, corr, std_, homogeneity], dtype="float")
    glcm_result = conv_smooth(glcm_result, ker_s=ker_s)
    print("glcm_result:", glcm_result.shape)
    means = np.mean(glcm_result.reshape((-1, f_dim)), axis=0).reshape((1, 1, f_dim)).repeat(new_W, 0).repeat(new_H, 1)
    stds = np.std(glcm_result.reshape((-1, f_dim)), axis=0).reshape((1, 1, f_dim)).repeat(new_W, 0).repeat(new_H, 1)
    print(means.shape)
    print(stds.shape)
    # plt.plot(np.arange(f_dim), np.var(glcm_result.reshape((-1, f_dim)), axis=0).flatten())
    # plt.show()
    glcm_result = (glcm_result - means) / stds
    # plt.plot(np.arange(f_dim), np.var(glcm_result.reshape((-1, f_dim)), axis=0).flatten())
    # plt.show()
    names = ["熵(entropy)", "能量(energy)", "惯性(inertia)", "相关性(correlation)", "标准差(std)", "同质性(homogeneity)"]
    for i in range(f_dim):
        plot_3d(np.arange(new_W), np.arange(new_H), glcm_result[:, :, i],
                np.var(win_wrap_img.reshape((new_W, new_H, -1)) / np.max(win_wrap_img), axis=-1), title=names[i])
    return glcm_result


def dist(a, b, img):
    now_win = img[a[0]][a[1]]
    distance = []
    for index in range(len(b)):
        diff = (now_win - b[index]) ** 2
        distance.append(np.sum(diff))
    return np.array(distance)


def point_change(old, new):
    return np.sum((old - new) ** 2)


def k_means(img, K):
    W, H, f_dim = img.shape
    C = np.array([img[np.random.randint(low=0, high=W)][[np.random.randint(low=0, high=H)]][0] for i in range(K)])
    old = np.zeros_like(C)
    clusters = np.zeros((W, H))
    while point_change(C, old) != 0:
        for i in range(W):
            for j in range(H):
                distance = dist(a=[i, j], b=C, img=img)
                cluster = np.argmin(distance)
                clusters[i][j] = cluster
        old = copy.deepcopy(C)
        for k in range(K):
            points = []
            for i in range(W):
                for j in range(H):
                    if clusters[i][j] == k:
                        points.append(img[i][j])
            C[k] = np.mean(np.array(points), axis=0)
    return clusters


if __name__ == '__main__':
    np.random.seed(1)
    test_img = load_img(r"./data/texture_mosaic/Texture_mosaic_2.jpg")
    show_gray_img(test_img)
    glcm_result_ = glcm_conv(win_len=19, img=test_img, stride=(1, 1), gray_level=16, ker_s=10)
    clusters = k_means(glcm_result_, K=3)
    show_color_img(clusters)
