# coding: utf-8
# Author：WangTianRui
# Date ：2020/12/14 11:19
# 用来测试一些方法函数
import numpy as np

# 数独是个 9x9 的二维数组
# 包含 9 个 3x3 的九宫格
sudoku = np.array([
    [2, 8, 7, 1, 6, 5, 9, 4, 3],
    [9, 5, 4, 7, 3, 2, 1, 6, 8],
    [6, 1, 3, 8, 4, 9, 7, 5, 2],
    [8, 7, 9, 6, 5, 1, 2, 3, 4],
    [4, 2, 1, 3, 9, 8, 6, 7, 5],
    [3, 6, 5, 4, 2, 7, 8, 9, 1],
    [1, 9, 8, 5, 7, 3, 4, 2, 6]
])


def argmax_2d(array):
    w, h = array.shape
    max_idx = np.argmax(array)
    return max_idx // h, max_idx % h


# print(argmax_2d(sudoku))


# 要将其变成 3x3x3x3 的四维数组
# 但不能直接 reshape，因为这样会把一行变成一个九宫格
# shape = (6, 7, 3, 3)

# 大行之间隔 27 个元素，大列之间隔 3 个元素
# 小行之间隔 9 个元素，小列之间隔 1 个元素
# strides = sudoku.itemsize * np.array([9, 1, 9, 1])

# squares = np.lib.stride_tricks.as_strided(sudoku, shape=shape, strides=strides)


# print(squares)


def split(array, win):
    array_w, array_h = array.shape
    win_w, win_h = win.shape
    strides = sudoku.itemsize * np.array([array_h * 3, 1 * 3, array_h, 1])
    return np.lib.stride_tricks.as_strided(array,
                                           shape=[(array_w - win_w) // 3 + 1, (array_h - win_h) // 3 + 1, win_w, win_h],
                                           strides=strides)


test = split(sudoku, win=np.zeros([3, 4]))
print(test)
# test_array = np.array([1, 2, 3, 4, 5, 6]).reshape([-1, 1, 1]).repeat(7, 1).repeat(9, 2)
# print(test_array - sudoku)
