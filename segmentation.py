# coding: utf-8
# Author：WangTianRui
# Date ：2020/12/2 21:33
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置


def load_img(path):
    if os.path.exists(path):
        return np.array(mpimg.imread(path), dtype="float")
    else:
        print("图片路径错误")
        return


def show_gray_img(img, title=""):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


def smooth(x, win_len=31):
    pad_len = (win_len - 1) // 2
    padded = np.pad(x, (pad_len, pad_len), "constant")
    n_frame = len(padded) - (win_len - 1)
    strides = padded.itemsize * np.array([1, 1])
    win_wrapped = np.lib.stride_tricks.as_strided(padded, shape=(n_frame, win_len), strides=strides)
    return np.mean(win_wrapped, axis=1)


def get_histogram(img, bins=200, smooth_win=31):
    n, bins, patches = plt.hist(img.flatten(), bins=bins, density=True)
    plt.title("原始灰度值分布直方图")
    plt.show()
    bin_center = []
    for index in range(len(bins) - 1):
        bin_center.append((bins[index] + bins[index + 1]) / 2)
    return smooth(n / np.sum(n), win_len=smooth_win), np.array(bin_center)


def gaussian(mu, theta, x):
    theta = theta + np.array(1e-12)
    return (1 / ((2 * np.pi) ** 0.5 * theta)) * np.exp(-(x - mu) ** 2 / (2 * theta ** 2))


def gaussian_mixture_by_threshold(threshold, p_of_bin, bin_center, smooth_win=31):
    if threshold > np.max(bin_center) or threshold < 0:
        print("阈值设置越界")
        return None
    threshold_left_bins = bin_center[bin_center <= threshold]
    left_p = p_of_bin[bin_center <= threshold] / np.sum(p_of_bin[bin_center <= threshold])
    left_gaussian_mu = np.sum(threshold_left_bins * left_p)
    left_gaussian_theta = np.mean((threshold_left_bins - left_gaussian_mu) ** 2 * left_p)

    threshold_right_bins = bin_center[bin_center > threshold]
    right_p = p_of_bin[bin_center > threshold] / np.sum(p_of_bin[bin_center > threshold])
    right_gaussian_mu = np.sum(threshold_right_bins * right_p)
    right_gaussian_theta = np.mean((threshold_right_bins - right_gaussian_mu) ** 2 * right_p)

    p_l = np.sum(p_of_bin[bin_center <= threshold])
    p_r = 1 - p_l
    P = []
    for inx, bin_value in enumerate(bin_center):
        P.append(p_l * gaussian(left_gaussian_mu, left_gaussian_theta, bin_value)
                 + p_r * gaussian(right_gaussian_mu, right_gaussian_theta, bin_value))
    return smooth(P, win_len=smooth_win)


def divergence(f, p):
    eps = np.array(1e-9)
    p = p + eps
    return np.mean(f * np.log(f / p + eps))


def get_best_threshold(p_of_bin, bin_center, section, smooth_win=31):
    mus = []
    divers = []
    Ps = []
    for i in range(section[0], section[1]):
        mus.append(i)
        P = gaussian_mixture_by_threshold(i, p_of_bin, bin_center, smooth_win)
        Ps.append(P)
        divers.append(divergence(p_of_bin, P))
    plt.plot(range(*section), divers)
    plt.title("阈值与散度曲线")
    plt.show()
    plt.bar(bin_center, Ps[int(np.argmin(divers))])
    plt.title("最好阈值对应的分布")
    plt.show()
    return np.argmin(divers) + section[0]


def process_img(threshold, img):
    image = img.copy()
    image[img > threshold] = 1
    image[img <= threshold] = 0
    return image


if __name__ == '__main__':
    test_pic = r"./data/segmenttation_data/Test_Img_2.jpg"
    test_pic = load_img(test_pic)
    # 设置三个阈值观察结果
    img = process_img(50, test_pic)
    show_gray_img(img, title="阈值为50")
    img = process_img(120, test_pic)
    show_gray_img(img, title="阈值为120")
    img = process_img(200, test_pic)
    show_gray_img(img, title="阈值为200")
    ######################################
    p_of_bin, bin_center = get_histogram(test_pic, bins=200, smooth_win=71)
    plt.plot(p_of_bin)
    plt.title("平滑后灰度值分布")
    plt.show()
    threshold = get_best_threshold(p_of_bin, bin_center, [50, 200], smooth_win=71)
    print("自适应计算最佳阈值:", threshold)
    img = process_img(threshold, test_pic)
    show_gray_img(img, "自适应最佳阈值")
