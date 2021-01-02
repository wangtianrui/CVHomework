# coding: utf-8
# Author：WangTianRui
# Date ：2020/12/18 8:09
import cv2, os
import numpy as np
import matplotlib.pyplot as plt

global img, cut_img
global point1, point2
global local


def on_mouse(event, x, y, flags, param):
    global img, point1, point2, local, cut_img
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 5)
        cv2.imshow('image', img2)
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        cut_img = img[min_y:min_y + height, min_x:min_x + width]
        # print(min_x, min_y, width, height)
        local = (min_y, min_x, height, width)
        cv2.imshow('result', cut_img)
        cv2.imshow('note', cv2.imread("./data/tracking/note.png"))


def split_img():
    global img
    img = cv2.imread('./data/tracking/Car_Data/car001.bmp')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)


def gauss(x):
    return (1 / ((2 * np.pi) ** 0.5 * 1)) * np.exp(-(x - 0) ** 2 / (2 * 1 ** 2))


def mean_shift(car_frame, top_y, top_x, n_rows, n_cols):
    nbin = 32
    h = (n_cols // 2) * 2 + (n_rows // 2) * 2  # 带宽
    center = [n_rows // 2, n_cols // 2]  # 中心点位置
    dis_k = np.zeros((n_rows, n_cols))  # 距离权重矩阵k
    dis_g = np.zeros((n_rows, n_cols))  # 距离权重g
    C = 0  # 权值归一化系数
    points_top = np.zeros((101, 2))  # 每个图片目标对应的左上角的坐标
    points_top[0] += np.array([top_x, top_y])  # 将第一帧我们画的放进去
    for i in range(n_rows):
        for j in range(n_cols):
            dist = ((i - center[0]) ** 2 + (j - center[1]) ** 2) / h  # 计算该点离目标框中心的距离
            dis_k[i, j] = gauss(dist)
            dis_g[i, j] = dis_k[i, j] * dist  # 高斯平滑核
            C = C + dis_k[i, j]

    frame_hist = np.zeros(nbin)
    frame_bin = np.zeros((n_rows, n_cols))
    error = 0
    for i in range(n_rows):
        for j in range(n_cols):
            frame_bin[i, j] = (car_frame[i, j]) // nbin
            frame_hist[int(frame_bin[i, j])] += dis_k[i, j]
    frame_hist = frame_hist / C

    # 迭代处理图
    for pic_i in range(2, 101):
        if pic_i < 10:
            pic_name = "car00%d.bmp" % pic_i
        elif 100 > pic_i >= 10:
            pic_name = "car0%d.bmp" % pic_i
        else:
            pic_name = "car%d.bmp" % pic_i
        path = os.path.join(r"./data/tracking/Car_Data/", pic_name)
        im_i = cv2.imread(path)
        im_i = cv2.cvtColor(im_i, cv2.COLOR_BGR2GRAY)
        # 设置阈值
        time = 0
        while time < 10:
            time += 1
            # 拿出上一次画出的框
            car_frame = im_i[top_y:top_y + n_rows + 1, top_x:top_x + n_cols + 1]
            frame_bin_now = np.zeros((n_rows, n_cols))
            frame_hist_now = np.zeros(nbin)
            for i in range(n_rows):
                for j in range(n_cols):
                    frame_bin_now[i, j] = (car_frame[i, j]) // nbin
                    frame_hist_now[int(frame_bin_now[i, j])] += dis_k[i, j]
            frame_hist_now = frame_hist_now / C

            w = np.zeros(nbin)
            for i in range(nbin):
                if frame_hist_now[i] != 0:
                    w[i] = np.sqrt(frame_hist[i] / frame_hist_now[i])
                else:
                    w[i] = 0
            w = w / 2
            coeff = 0
            wx = [0, 0]
            for i in range(n_rows):
                for j in range(n_cols):
                    coeff = coeff + dis_g[i, j] * w[int(frame_bin_now[i, j])]
                    wx = wx + dis_g[i, j] * w[int(frame_bin_now[i, j])] * np.array([i - center[0], j - center[1]])
            delta = wx / coeff
            top_x = int(top_x + delta[1])
            top_y = int(top_y + delta[0])
            if delta[0] ** 2 + delta[1] ** 2 == error:
                break
            else:
                error = delta[0] ** 2 + delta[1] ** 2
        show_result(im_i, [top_x, top_y], n_rows, n_cols,
                    os.path.join(r"./data/tracking/save/", '%s.jpg' % pic_name.split('.')[0]))
        points_top[pic_i - 1] += np.array([top_x, top_y])
    print(points_top)


def show_result(img, point_top, n_rows, n_cols, save):
    plt.imshow(img, cmap='gray')
    ax = plt.gca()
    # 默认框的颜色是黑色，第一个参数是左上角的点坐标
    # 第二个参数是宽，第三个参数是长
    ax.add_patch(plt.Rectangle((point_top[0], point_top[1]), n_cols, n_rows, color="blue", fill=False, linewidth=1))
    plt.savefig(save)
    plt.show()
    plt.pause(0.2)
    plt.close()


if __name__ == '__main__':
    split_img()
    print("画完了", local)
    mean_shift(cut_img, *local)
