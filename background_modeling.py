# coding: utf-8
# Author：WangTianRui
# Date ：2021/1/2 19:07
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time as T

home = r'./data/Scene_Data/'


def load_img(path):
    if os.path.exists(path):
        gray = cv2.cvtColor(mpimg.imread(path), cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        return cv2.resize(gray, (width // 5, height // 5))
    else:
        print("图片路径错误")
        return


def save_result(img, pic_name):
    # pic_name = pic_name.replace('jpg', 'svg')
    plt.imshow(img, interpolation='none')
    plt.title(pic_name)
    plt.savefig(os.path.join(home, 'save_3', pic_name))
    plt.show()


num_gauss = 3  # 高斯的数量
D = 2.5  # 阈值系数
sd_init = 6  # 标准差初始化
num_pic = 201  # 200张图片
lr = 0.01  # 学习率
threshold = 0.7
first = load_img(os.path.join(home, '0000.jpg'))
n_row, n_col = first.shape  # H,W
weights = np.zeros((n_row, n_col, num_gauss))  # 用来存混合高斯的参数
weights[:, :, 0] = 1
mean = np.zeros((n_row, n_col, num_gauss))  # 混合高斯的均值
mean[:, :, 0] += first[:, :]  # 将第一个初始化为第一帧的像素值
sd = np.ones_like(mean) * sd_init
mask = np.ones((num_pic, n_row, n_col))
n = 0
for i in os.walk(home):
    for name in i[2]:
        start = T.time()
        if n == 0:
            n += 1
            continue
        if name.endswith("jpg"):
            frame = load_img(os.path.join(home, name))
            print(name)
            for row in range(n_row):
                for col in range(n_col):
                    match = False
                    # 更新参数
                    for k in range(num_gauss):
                        diff = abs(frame[row, col] - mean[row, col, k])
                        if diff <= D * sd[row, col, k] and not match:
                            match = True
                            weights[row, col, k] = (1 - lr) * weights[row, col, k] + lr
                            p = lr / weights[row, col, k]
                            mean[row, col, k] = (1 - p) * mean[row, col, k] + p * frame[row, col]
                            sd[row, col, k] = np.sqrt((1 - p) * (sd[row, col, k] ** 2) + p * (diff ** 2))
                        else:
                            weights[row, col, k] = (1 - lr) * weights[row, col, k]
                    # 如果没有匹配上，则用新模型
                    if not match:
                        min_arg = np.argmin(weights[row, col, :])
                        mean[row, col, min_arg] = frame[row, col]
                        sd[row, col, min_arg] = sd_init
                    # 对于各个高斯进行weight/std排序。找出前k个，作为背景模型，其余为前景
                    w_sum = np.sum(weights[row, col, :])
                    weights[row, col, :] = weights[row, col, :] / w_sum  # 归一化
                    for p in range(num_gauss):
                        if np.sum(weights[row, col, :(p + 1)]) >= threshold:
                            if abs(frame[row, col] - mean[row, col, p]) <= D * sd[row, col, p]:
                                mask[n, row, col] = 255
                                break
                            else:
                                mask[n, row, col] = 1
                                break
            mask[n, 0, 0] = 1
            save_result(mask[n], name)
            cost = T.time() - start
            print("单张耗时:", cost, "每秒处理:", 1 / cost)
            n += 1
    break
# videowrite = cv2.VideoWriter(r'test.mp4', -1, 20, first.shape)  # 20是帧数，size是图片尺寸
# for i in range(len(mask)):
#     videowrite.write(mask[i])
