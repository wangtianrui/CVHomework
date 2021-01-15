# coding: utf-8
# Author：WangTianRui
# Date ：2021/1/15 13:01
import cv2 as cv
import os
import time
import numpy as np
from sklearn.svm import LinearSVC


def file_name(file_dir):
    LIST = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.bmp':
                LIST.append(file_dir + file)
    return LIST


# 滑窗
def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield x, y, image[y:y + window_size[1], x:x + window_size[0]]


# IOU
def overlapping_area(detection_1, detection_2):
    # Calculate the x-y co-ordinates of the
    # rectangles
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[3]
    x2_br = detection_2[0] + detection_2[3]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[4]
    y2_br = detection_2[1] + detection_2[4]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[3] * detection_2[4]
    area_2 = detection_2[3] * detection_2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)


# 非极大值抑制
def nms(detections, threshold=.5):
    if len(detections) == 0:
        return []
    # Sort the detections based on confidence score
    detections = sorted(detections, key=lambda detections: detections[2],
                        reverse=True)
    # Unique detections will be appended to this list
    new_detections = []
    # Append the first detection
    new_detections.append(detections[0])
    # Remove the detection from the original list
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


time_start = time.time()
# 训练集
path = './data/detect/train_34x94/'
train_neg = file_name(path + 'neg/')
train_pos = file_name(path + 'pos/')
neg_y = np.zeros(len(train_neg))
pos_y = np.ones(len(train_pos))
train_path = train_neg + train_pos
train_label = np.append(neg_y, pos_y)
winSize = (94, 34)  # 窗口，车辆大小34*94
blockSize = (8, 8)  # 块大小
blockStride = (2, 2)  # 块滑动增量
cellSize = (4, 4)  # 胞元大小
nbins = 9  # 梯度方向数
hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
train_data = []
winStride = (0, 0)
for path in train_path:
    img = cv.imread(path)
    fea = hog.compute(img, winStride)
    train_data.append(np.squeeze(fea))
# 线性核支持向量分类
clf = LinearSVC()
clf.fit(train_data, train_label)

# 测试集
test_path = './data/detect/test/'
test_file = file_name(test_path)
min_wdw_sz = (94, 34)
step_size = (2, 2)
visualize_det = True  # 可视化True
num = 0
for idx in range(len(test_file)):
    im = cv.imread(test_file[idx])
    detections = []
    cd = []
    for (x, y, im_window) in sliding_window(im, min_wdw_sz, step_size):
        if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
            continue
        # Calculate the HOG features
        fd = hog.compute(im_window, winStride)
        fd = np.squeeze(fd).tolist()
        fd = [fd]
        pred = clf.predict(fd)
        if pred == 1:
            detections.append((x, y, clf.decision_function(fd),
                               int(min_wdw_sz[0]),
                               int(min_wdw_sz[1])))
            cd.append(detections[-1])
        # If visualize is set to true, display the working of the sliding window
        if visualize_det:
            clone = im.copy()
            for x1, y1, _, _, _ in cd:
                # Draw the detections at this scale
                cv.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                                               im_window.shape[0]), (0, 0, 0), thickness=2)
            cv.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                                         im_window.shape[0]), (255, 255, 255), thickness=2)
            cv.imshow("Sliding Window in Progress", clone)
            cv.waitKey(1)
    # 非极大值抑制
    clone = im.copy()
    threshold = 0.01
    detections = nms(detections, threshold)

    # Display the results after performing NMS
    for (x_tl, y_tl, _, w, h) in detections:
        # Draw the detections
        num += 1
        cv.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness=2)
    cv.imwrite('./data/detect/result/{0}.bmp'.format(idx), clone)
print('共检测到', num, '辆车')
# %%
time_end = time.time()
print('用时：', time_end - time_start, 's')
