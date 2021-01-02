# coding: utf-8
# Author：WangTianRui
# Date ：2020/12/14 9:20
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os, cv2, time
from mpl_toolkits.mplot3d import Axes3D


def load_img(path):
    if os.path.exists(path):
        return np.array(mpimg.imread(path), dtype="float32").astype(np.uint8)
    else:
        print("图片路径错误")
        return


def plot_3d(x, y, z, title):
    fig = plt.figure()
    ax = Axes3D(fig)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
    # zdir 表示向那个轴投影
    # ax.contourf(x, y, img, zdir='z', offset=np.max(z) + 0.5, cmap='gray')
    ax.set_title(title)
    plt.show()


def show_gray_img(img, title=""):
    plt.imshow(img)
    plt.title(title)
    plt.show()


def corr(frame1, frame2):
    return np.sum(frame1 * frame2) / np.sqrt(np.sum(frame1 ** 2) * np.sum(frame2 ** 2))


def argmax_2d(array):
    h, w = array.shape
    max_idx = np.argmax(array)
    return max_idx // w, max_idx % w


def argmin_2d(array):
    h, w = array.shape
    min_dix = np.argmin(array.flatten())
    return min_dix // w, min_dix % w


def ncc(template, scene):
    # 中心化
    template = template - np.mean(template)
    scene = scene - np.mean(scene)
    template_h, template_w = template.shape
    scene_h, scene_w = scene.shape
    print(template_h, template_w, scene_h, scene_w)
    # 分窗
    strides = scene.itemsize * np.array([scene_w, 1, scene_w, 1])
    # print(strides)
    new_h = scene_h - (template_h - 1)
    new_w = scene_w - (template_w - 1)
    win_wrap_img = np.lib.stride_tricks.as_strided(scene, shape=(new_h, new_w, template_h, template_w),
                                                   strides=strides)
    print(win_wrap_img.shape)
    corr_result = np.zeros((new_h, new_w))
    for h in range(new_h):
        for w in range(new_w):
            corr_result[h][w] += corr(template, win_wrap_img[h][w])
    # show_gray_img(corr_result)
    plot_3d(np.arange(new_w), np.arange(new_h), corr_result, "ncc corr result")
    max_local = argmax_2d(corr_result)
    return [*max_local, *template.shape]


def hausdorff_dis(frame1, frame2):
    frame1_2 = np.min(
        (
            abs(frame1.reshape([-1, 1]).repeat(frame2.shape[0], 1) - frame2)
        ).reshape([len(frame1.flatten()), -1]), axis=-1
    )
    frame2_1 = np.min(
        (
            abs(frame2.reshape([-1, 1]).repeat(frame1.shape[0], 1) - frame1)
        ).reshape([len(frame2.flatten()), -1]), axis=-1
    )
    return np.max((np.max(frame1_2), np.max(frame2_1)))


def hausdorff(template, scene):
    template = cv2.Canny(template, 150, 250)
    scene = cv2.Canny(scene, 150, 250)
    show_gray_img(scene)
    show_gray_img(template)
    stride = 3
    template_h, template_w = template.shape
    scene_h, scene_w = scene.shape
    strides = scene.itemsize * np.array([scene_w * stride, 1 * stride, scene_w, 1])
    new_h = (scene_h - template_h) // stride + 1
    new_w = (scene_w - template_w) // stride + 1
    win_wrap_img = np.lib.stride_tricks.as_strided(scene, shape=(new_h, new_w, template_h, template_w),
                                                   strides=strides)
    temp_x, temp_y = np.where(template > 0)
    dis_result = np.zeros((new_h, new_w))
    print(win_wrap_img.shape)
    for h in range(new_h):
        for w in range(new_w):
            win_x, win_y = np.where(win_wrap_img[h][w] > 0)
            dis_result[h][w] += np.max([hausdorff_dis(temp_x, win_x), hausdorff_dis(temp_y, win_y)])
    plot_3d(np.arange(new_w), np.arange(new_h), dis_result, "hausdorff dis result")
    min_local = argmin_2d(dis_result)
    print(min_local)
    return [(min_local[0] - 1) * stride, (min_local[1] - 1) * stride, *template.shape]


def hausdorff_distance_trans(bk_edge):
    row, col = bk_edge.shape
    min_dis = (row + col) * np.ones((row, col))
    a, b = np.where(bk_edge == 255)
    for i in range(row):
        for j in range(col):
            min_dis[i, j] = np.min(abs(a - i) + abs(b - j))
    return min_dis


def distance_transform(template, scene):
    template = cv2.Canny(template, 150, 250)
    scene = cv2.Canny(scene, 150, 250)
    show_gray_img(scene)
    print(np.max(scene))
    scene = hausdorff_distance_trans(scene)
    show_gray_img(scene)
    stride = 1
    template_h, template_w = template.shape
    scene_h, scene_w = scene.shape
    strides = scene.itemsize * np.array([scene_w * stride, 1 * stride, scene_w, 1])
    new_h = (scene_h - template_h) // stride + 1
    new_w = (scene_w - template_w) // stride + 1
    win_wrap_img = np.lib.stride_tricks.as_strided(scene, shape=(new_h, new_w, template_h, template_w),
                                                   strides=strides)
    dis_result = np.zeros((new_h, new_w))
    temp_x, temp_y = np.where(template > 0)
    print(win_wrap_img.shape)
    for h in range(new_h):
        for w in range(new_w):
            # dis_result[h][w] += corr(template, win_wrap_img[h][w])
            # win_x, win_y = np.where(win_wrap_img[h][w] > 0)
            # dis_result[h][w] += np.max([hausdorff_dis(temp_x, win_x), hausdorff_dis(temp_y, win_y)])
            dis_result[h][w] += np.mean(template * win_wrap_img[h][w])
    plot_3d(np.arange(new_w), np.arange(new_h), dis_result, "distance_transform corr result")
    min_local = argmin_2d(dis_result)
    # print(min_local)
    return [(min_local[0] - 1) * stride, (min_local[1] - 1) * stride, *template.shape]


def draw_rectangle(locals, scene, title=""):
    plt.imshow(scene, cmap='gray')
    plt.title(title)
    ax = plt.gca()
    # 默认框的颜色是黑色，第一个参数是左上角的点坐标
    # 第二个参数是宽，第三个参数是长
    names = ["Template_1.jpg", "Template_2.jpg"]
    for indx, local in enumerate(locals):
        ax.add_patch(plt.Rectangle((local[1], local[0]), local[2], local[3], color="blue", fill=False, linewidth=1))
        ax.text(local[1], local[0], names[indx], bbox={'facecolor': 'blue', 'alpha': 0.5})
    plt.title(title)
    plt.show()


def find_position(templates, scene, function, title):
    scene_r = scene.shape[0]
    locals = []
    for template in templates:
        locals.append(function(template, scene))
    print(locals)
    print("左下角为(0,0)情况下,模版1和2目标框的左上角坐标分别为:\n", [[scene_r - item[0], item[1]] for item in locals])
    draw_rectangle(locals, scene, title=title)


if __name__ == '__main__':
    img_home = r"./data/template_matching/"
    template_test_1 = load_img(os.path.join(img_home, "Template_1.jpg"))
    template_test_2 = load_img(os.path.join(img_home, "Template_2.jpg"))
    scene_test = load_img(os.path.join(img_home, "Scene.jpg"))

    start = time.time()
    find_position([template_test_1, template_test_2], scene_test, ncc, "ncc")
    print("ncc耗时:", time.time() - start)

    start = time.time()
    find_position([template_test_1, template_test_2], scene_test, hausdorff, "hausdorff")
    print("hausdorff耗时:", time.time() - start)

    start = time.time()
    find_position([template_test_1, template_test_2], scene_test, distance_transform, "distance_transform")
    print("distance_transform+hausdorff耗时:", time.time() - start)
