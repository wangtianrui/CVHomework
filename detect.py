# coding: utf-8
# Author：WangTianRui
# Date ：2021/1/15 13:01
import cv2
import os
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def zh_font(string):
    return string.encode("gbk").decode(errors="ignore")


def nms(detections, threshold=.5):
    """
    非极大值抑制
    """
    if len(detections) == 0:
        return []
    detections = sorted(detections, key=lambda detections: detections[2],
                        reverse=True)
    new_detections = [detections[0]]
    del detections[0]
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if overlapping_area(detection, new_detection) > threshold:
                del detections[index]
                break
        else:
            new_detections.append(detection)
            del detections[index]
    return new_detections


def file_name(file_dir):
    """
    获取所有图片名
    """
    names = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.bmp':
                names.append(os.path.join(file_dir, file))
    return names


def overlapping_area(result_1, result_2):
    """
    IOU
    """
    x_1, y_1 = result_1[0], result_1[1]
    x_2, y_2 = result_2[0], result_2[1]
    x1_br = result_1[0] + result_1[3]
    x2_br = result_2[0] + result_2[3]
    y1_br = result_1[1] + result_1[4]
    y2_br = result_2[1] + result_2[4]
    x_overlap = max(0, min(x1_br, x2_br) - max(x_1, x_2))
    y_overlap = max(0, min(y1_br, y2_br) - max(y_1, y_2))
    overlap_area = x_overlap * y_overlap
    area_1 = result_1[3] * result_2[4]
    area_2 = result_2[3] * result_2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)


if __name__ == '__main__':
    home = './data/detect/'
    time_start = time.time()
    path = os.path.join(home, "train_34x94")
    train_neg = file_name(os.path.join(path, 'neg'))
    train_pos = file_name(os.path.join(path, 'pos'))
    neg_y = np.zeros(len(train_neg))
    pos_y = np.ones(len(train_pos))
    train_path = train_neg + train_pos
    all_label = np.append(neg_y, pos_y)
    winSize = (94, 34)
    blockSize = (8, 8)
    blockStride = (2, 2)
    cellSize = (4, 4)
    nbins = 11
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    all_data = []
    winStride = (0, 0)
    for path in train_path:
        img = cv2.imread(path)
        fea = hog.compute(img, winStride)
        all_data.append(np.squeeze(fea))
    train_data, test_data, train_label, test_label = train_test_split(all_data, all_label, test_size=0.2, shuffle=True)
    print("数据划分结果，训练集：", np.shape(train_data), "测试集：", np.shape(test_data))
    # PCA
    pca = PCA(n_components=300)
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    print("pca结果，训练集：", np.shape(train_data), "测试集：", np.shape(test_data))
    # Logistic
    clf = LogisticRegression()
    clf.fit(train_data, train_label)
    print("分类器训练结果:")
    print(classification_report(clf.predict(test_data), test_label, target_names=["no car", "car"]))
    # 测试集
    test_path = os.path.join(home, "test")
    test_file = file_name(test_path)
    min_wdw_sz = (94, 34)
    visualize_det = True
    num = 0
    for idx in range(len(test_file)):
        im = cv2.imread(test_file[idx])
        detections = []
        cd = []
        step = 18
        step_size = (im.shape[1] // step, im.shape[0] // step)
        y = 0
        while y <= (im.shape[0] - step_size[1]):
            x = 0
            while x <= (im.shape[1] - step_size[0]):
                im_window = im[y:y + min_wdw_sz[1], x:x + min_wdw_sz[0]]
                if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                    break
                fd = hog.compute(im_window, winStride)
                fd = np.squeeze(fd).tolist()
                fd = [fd]
                pred = clf.predict_proba(pca.transform(fd))[0]
                # print(clf.decision_function(pca.transform(fd)))
                if pred[1] > 0.1:
                    step_size = (3, 3)
                else:
                    step_size = (im.shape[1] // step, im.shape[0] // step)
                if pred[0] < pred[1]:
                    detections.append((x, y, clf.decision_function(pca.transform(fd)),
                                       int(min_wdw_sz[0]),
                                       int(min_wdw_sz[1])))
                    cd.append(detections[-1])
                if visualize_det:
                    clone = im.copy()
                    for x1, y1, _, _, _ in cd:
                        cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                                                        im_window.shape[0]), (255, 0, 0), thickness=2)
                    cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                                                  im_window.shape[0]), (255, 0, 0), thickness=2)
                    cv2.imshow(zh_font("滑窗"), clone)
                    x += step_size[0]
                    cv2.waitKey(1)
                    # 非极大值抑制
                else:
                    x += step_size[0]
            y += step_size[1]
        clone = im.copy()
        threshold = 0.01
        detections = nms(detections, threshold)
        for (x_tl, y_tl, _, w, h) in detections:
            # Draw the detections
            num += 1
            cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (255, 0, 0), thickness=2)
        cv2.imwrite(os.path.join(home, 'result', '{0}.bmp'.format(idx)), clone)
    time_end = time.time()
    print('用时：', time_end - time_start, 's')
    print("平均每张图耗时:", (time_end - time_start) / (idx + 1), "s")
    print(num)
