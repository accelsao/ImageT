import cv2
from matplotlib import pyplot
import math
import numpy as np
from itertools import combinations


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def len_points(p1, p2):  # 計算兩點長度
    length = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return length


def area_tri(a, b, c):  # 計算三角形面積
    s = (a + b + c) / 2
    area = math.sqrt((s * (s - a) * (s - b) * (s - c)))
    return area


def cal_area(p):  # 計算四個點圍出來的面積
    a1 = area_tri(len_points(p[0], p[1]), len_points(p[1], p[2]), len_points(p[2], p[0]))
    a2 = area_tri(len_points(p[1], p[2]), len_points(p[2], p[3]), len_points(p[3], p[1]))
    a = a1 + a2
    return a


def find_max_area(p):  # 找最大的面積
    comb = list(combinations(p, 4))
    # print(comb)
    area = []
    for c in comb:
        area.append(cal_area(c))
    # print(area)
    MAX = comb[area.index(max(area))]
    MAX = list(MAX)
    return MAX


def order_corners(corners, p):
    order = [[0, 0], [0, 0], [0, 0], [0, 0]]  # "左上 左下 右上 右下"
    for i in corners:
        if i[0] < p[0] and i[1] < p[1]:
            order[0] = i
        elif i[0] < p[0] and i[1] >= p[1]:
            order[1] = i
        elif i[0] >= p[0] and i[1] < p[1]:
            order[2] = i
        elif i[0] >= p[0] and i[1] >= p[1]:
            order[3] = i
    return order


def affine(img_input, threshold):
    img = threshold
    oimg = cv_imread(img_input)
    '''
    # 除錯用的圖形
    pyplot.imshow(cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB))
    pyplot.show()
    pyplot.imshow(img,'gray')
    pyplot.show()
    '''
    debug = oimg.copy()
    line_width = 10

    # closing
    k = 0
    if img.shape[0] > img.shape[1]:
        k = img.shape[1]
    else:
        k = img.shape[0]
    k = k // 10
    if k % 2 == 0:
        k = k + 1
    '''
    kernel = np.ones((k,k),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    # 除錯用的圖形
    pyplot.imshow(img,'gray')
    pyplot.show()
    '''
    # 產生等高線
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    '''
    cv2.drawContours(debug, contours, -1, (255, 0, 255), line_width)
    # 除錯用的圖形
    pyplot.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))
    pyplot.show()
    '''
    c = max(contours, key=cv2.contourArea)
    cv2.drawContours(debug, c, -1, (0, 255, 0), line_width)
    '''
    # 除錯用的圖形
    pyplot.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))
    pyplot.show()
    '''
    for i in range(0, len(contours)):
        cv2.fillPoly(img, pts=[contours[i]], color=(255, 255, 255))
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    kernel_size = 11
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # 在圖片周圍加上黑邊，讓處理過程不會受到邊界影響
    padding = int(img.shape[1] / 15)
    img = cv2.copyMakeBorder(img, padding, padding, padding, padding,
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    imd_pad = cv2.copyMakeBorder(oimg, padding, padding, padding, padding,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
    imd_pad_copy = imd_pad.copy()
    corners = []
    # 用 Shi-Tomasi coner 選20個點出來
    corners_img = cv2.goodFeaturesToTrack(img, 20, 0.01, padding // 2)
    corners_img = np.int0(corners_img)
    for c in corners_img:
        x, y = c.ravel()
        corners.append((x, y))
    corners.sort()

    # colors=[[0,0,255],[0,255,0],[255,0,0],[255,0,255],[0,255,255],[125,255,0]] # r,g,b,pink
    for i in range(0, len(corners)):
        # Circling the corners in green
        cv2.circle(imd_pad, corners[i], 50, [0, 255, 255], -1)
    '''
    print('6 corners:',corners)
    
    # 除錯用的圖形
    pyplot.imshow(img,'gray')
    pyplot.show()
    pyplot.imshow(cv2.cvtColor(imd_pad, cv2.COLOR_BGR2RGB))
    pyplot.show()
    '''
    # 在20個點當中找能圍出最大面積的四個點
    corners = find_max_area(corners)
    imd_pad = imd_pad_copy.copy()
    for i in range(0, len(corners)):
        cv2.circle(imd_pad, corners[i], 50, [0, 255, 0], -1)
        corners[i] = list(corners[i])

    # 平均中心點
    row_center = 0
    col_center = 0
    for i in range(0, 4):
        row_center += corners[i][0]
        col_center += corners[i][1]
    row_center = row_center // 4
    col_center = col_center // 4
    '''
    # 除錯用的圖形
    cv2.circle(imd_pad,(row_center,col_center),50,[255,0,255],-1)
    print('final corners: ',corners)
    '''
    img = cv2.cvtColor(imd_pad, cv2.COLOR_BGR2RGB)
    cv2.imwrite('images/imd_pad.png', img)
    pyplot.imshow(img)
    pyplot.show()

    # 以平均中心點為基準，將corners依照"左上 左下 右上 右下"順序排列
    corners = order_corners(corners, (row_center, col_center))
    print('ordered corners: ', corners)

    # Affine the image
    rows, cols, ch = oimg.shape
    # print(rows,cols,ch)
    pts1 = np.float32(corners)

    corners_ = [[0, 0], [0, rows], [cols, 0], [cols, rows]]  # 左上 左下 右上 右下
    # print(corners_)
    pts2 = np.float32(corners_)

    M = cv2.getPerspectiveTransform(pts1, pts2)
    # print(M)
    dst = cv2.warpPerspective(imd_pad_copy, M, (cols, rows))

    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    pyplot.imshow(dst)
    cv2.imwrite('images/dst.png', dst)
    pyplot.show()

    return dst
