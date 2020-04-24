# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:25:07 2020

@author: kumat
"""

import cv2
from matplotlib import pyplot
import numpy as np
from scipy.stats import gaussian_kde
import affine_big


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def get_threshold(img_input):
    img = cv_imread(img_input)

    # 轉換至 HSV 色彩空間
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 使用縮圖以減少計算量
    scale_percent = 20  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img_small = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img_small = np.uint8(np.clip((1.0 * img_small + 0), 0, 255))

    # 轉換至 HSV 色彩空間
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

    # 取出飽和度
    saturation = hsv[:, :, 1]
    saturation_small = hsv_small[:, :, 1]

    # 取出value
    value = hsv[:, :, 2]
    value_small = hsv_small[:, :, 2]

    # 綜合飽和度與明度
    sv_ratio = 0.85
    sv_value = cv2.addWeighted(saturation, sv_ratio, value, 1 - sv_ratio, 0)
    sv_value_small = cv2.addWeighted(saturation_small, sv_ratio, value_small, 1 - sv_ratio, 0)
    '''
    # 除錯用的圖形
    pyplot.subplot(131).set_title("Saturation"), pyplot.imshow(saturation), pyplot.colorbar()
    pyplot.subplot(132).set_title("Value"), pyplot.imshow(value), pyplot.colorbar()
    pyplot.subplot(133).set_title("SV-value"), pyplot.imshow(sv_value), pyplot.colorbar()
    pyplot.show()
    '''
    # 使用 Kernel Density Estimator 計算出分佈函數
    density = gaussian_kde(sv_value_small.ravel(), bw_method=0.2)

    # 找出 PDF 中第一個區域最小值（Local Minimum）作為門檻值
    step = 0.5
    xs = np.arange(0, 256, step)
    ys = density(xs)
    cum = 0
    threshold_value = -1
    for i in range(1, 250):
        cum += ys[i - 1] * step
        # print(cum)
        if (cum > 0.02) and (ys[i] < ys[i + 1]) and (ys[i] < ys[i - 1]):
            threshold_value = xs[i]

            break
    # threshold_value = 80

    if threshold_value == -1:
        threshold_value = np.mean(sv_value_small) + np.std(sv_value_small) * 1.5

    print("threshold:", threshold_value)

    '''   
    # 除錯用的圖形
    pyplot.hist(sv_value_small.ravel(), 256, [0, 256], True, alpha=0.5)
    pyplot.plot(xs, ys, linewidth = 2)
    pyplot.axvline(x=threshold_value, color='r', linestyle='--', linewidth = 2)
    pyplot.xlim([0, max(threshold_value*2, 80)])
    pyplot.show()
    '''

    # 以指定的門檻值篩選區域
    (_, threshold) = cv2.threshold(sv_value, threshold_value, 255.0, cv2.THRESH_BINARY)
    k = 0
    if threshold.shape[0] > threshold.shape[1]:
        k = threshold.shape[1]
    else:
        k = threshold.shape[0]
    k = k // 80
    if k % 2 == 0:
        k = k + 1
    print(k)
    kernel = np.ones((k, k), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    # 計算黑或白對應到物件
    unique, counts = np.unique(threshold, return_counts=True)
    if counts[0] > counts[1]:
        threshold = cv2.bitwise_not(threshold, threshold)
    '''
    # 除錯用的圖形
    pyplot.imshow(threshold, "gray")
    pyplot.show()
    '''
    # 產生等高線
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 建立除錯用影像
    img_debug = img.copy()

    # 線條寬度
    line_width = int(img.shape[1] / 100)

    # 以藍色線條畫出所有的等高線
    cv2.drawContours(img_debug, contours, -1, (255, 0, 0), line_width)

    # 找出面積最大的等高線區域
    c = max(contours, key=cv2.contourArea)
    print(len(c))
    # 以粉紅色畫出面積最大的等高線區域
    cv2.drawContours(img_debug, c, -1, (255, 0, 255), line_width)
    '''
    # 除錯用的圖形
    pyplot.imshow(cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB))
    pyplot.show()
    '''
    # 填滿threshold
    th_shape = threshold.shape
    threshold = np.zeros(th_shape)
    cv2.fillPoly(threshold, pts=[c], color=(255, 255, 255))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    # threshold = threshold // 255
    # print(threshold)
    cv2.imwrite('images/threshold.jpg', threshold)
    threshold = cv2.imread('images/threshold.jpg', cv2.IMREAD_GRAYSCALE)
    # print(threshold)

    # 除錯用的圖形
    pyplot.imshow(threshold, "gray")
    pyplot.show()

    return threshold


def rmBack(img_input):
    # img_input = 'P1130737.JPG'
    threshold = get_threshold(img_input)
    img_rmv = affine_big.affine(img_input, threshold)
    # cv2.imwrite('v3_'+img_input,img_rmv)
    cv2.imwrite('images/cut.jpg', img_rmv)
