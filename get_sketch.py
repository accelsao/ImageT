# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 22:25:28 2020

@author: kumat
"""

import numpy as np
import cv2
from matplotlib import pyplot
from skimage import measure

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),-1)
    return cv_img

def smooth(name, or_img_big):
    print(or_img_big.shape)
    w = or_img_big.shape[1]
    h = or_img_big.shape[0]
    if w > h:
        w_ = w/500
        h_ = int(h/w_)
        dim = (500,h_)
        or_img = cv2.resize(or_img_big, dim)
    else:
        h_ = h/500
        w_ = int(w/h_)
        dim = (w_,500)
        or_img = cv2.resize(or_img_big, dim)
    
    print('shape small: ',or_img.shape)
    pyplot.imshow(or_img)
    pyplot.show()
    hsv = cv2.cvtColor(or_img, cv2.COLOR_BGR2HSV)
    # 取出value
    value = hsv[:,:,2]
    # covert the BGR image to an YCbCr image
    gray_img = cv2.cvtColor(or_img, cv2.COLOR_RGB2GRAY)
    gray_img_copy = gray_img.copy()
    # copy the image to create a binary mask later
    binary_mask = np.copy(gray_img)
    binary_mask_line = np.copy(gray_img)
    
    # get mean value of the pixels in Y plane
    gray_mean = np.mean(gray_img)
    
    # get standard deviation of channel in Y plane
    gray_std = np.std(gray_img)
    print('before gray_mean: ', gray_mean)
    print('before gray_std: ', gray_std)
    
    pyplot.imshow(gray_img,'gray')
    pyplot.show()
    
    # get mean value of the pixels in Y plane
    v_mean = np.mean(value)
    # get standard deviation of channel in Y plane
    v_std = np.std(value)
    print('before v_mean: ', v_mean)
    print('before v_std: ', v_std)

    a_plot = np.zeros(gray_img.shape)
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            a = gray_img[i,j] - gray_mean #距離mean的距離
            binary_mask[i, j] = 0
            a_ = abs(a) / gray_std
            a_plot[i,j] = a_
            if gray_img[i,j] > gray_mean+ 0.5*gray_std:
                binary_mask[i, j] = 255
                    
                    
    pyplot.imshow(binary_mask)
    pyplot.show()
                    
    kernel = np.ones((5,5),np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    pyplot.imshow(binary_mask)
    pyplot.show()
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    pyplot.imshow(binary_mask)
    pyplot.show()  
    
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
                if  binary_mask[i, j] == 255:
                    a_ = a_plot[i,j]
                    gray_img[i,j] -= (gray_std // a_)//3
                    hsv[i,j,2] -= v_std //3
    
    
    after = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    pyplot.imshow(after)
    pyplot.show()
    
    gray_mean = np.mean(gray_img)
    gray_std = np.std(gray_img)
    
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])  # ←計算直方圖資訊
    
    #使用MatPlot繪出 histogram
    
    pyplot.figure()
    
    pyplot.title("Grayscale Histogram")
    
    pyplot.xlabel("Bins")
    
    pyplot.ylabel("# of Pixels")
    
    pyplot.plot(hist)
    
    pyplot.xlim([0, 256])
    pyplot.axvline(x=gray_mean, color='r', linestyle='--', linewidth = 2)
    pyplot.axvline(x=gray_mean+gray_std, color='g', linestyle='--', linewidth = 2)
    pyplot.axvline(x=gray_mean-gray_std, color='g', linestyle='--', linewidth = 2)
    pyplot.show()
    
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            binary_mask[i, j] = 0
            if gray_img[i,j] < gray_mean - gray_std:
                binary_mask[i, j] = 255
        
    thresh = binary_mask.copy()
    #在binary_mask用Connected-component
    labels = measure.label(thresh, connectivity=2, background=0)
    #顯示一共貼了幾個Lables（即幾個components）
    mask = np.zeros(thresh.shape, dtype="uint8")
    print("[INFO] Total {} blobs".format(len(np.unique(labels))))

    #依序處理每個labels
    for (i, label) in enumerate(np.unique(labels)):
        #如果label=0，表示它為背景
        if label == 0:
            #print("[INFO] label: 0 (background)")
            continue
        #否則為前景，顯示其label編號
        #print("[INFO] label: {} (foreground)".format(i))

        #建立該前景的Binary圖
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255        
        #有幾個非0的像素?
        numPixels = cv2.countNonZero(labelMask)
            
        #如果像素數目在
        if numPixels > 5:
            mask = cv2.add(mask, labelMask)
    '''
    #顯示所抓取到的
    cv2.imshow("Large Blobs", mask)
    cv2.imshow("Large", binary_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    or_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
 
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            if mask[i, j] == 0 :
                or_img[i,j] = [255,255,255]
    or_img = cv2.cvtColor(or_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('images/com_'+name,or_img)
    pyplot.imshow(or_img)
    pyplot.show()
    return or_img

def sketch(fileChoosed):
    fileChoosed = fileChoosed.replace('/','\\')
    print(fileChoosed)
    
    or_img = cv_imread(fileChoosed)
    or_img = cv2.cvtColor(or_img, cv2.COLOR_BGR2RGB)
    after = smooth(fileChoosed,or_img)
    cv2.imwrite('images/result.jpg',after)
