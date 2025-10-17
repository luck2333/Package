import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from yolox_onnx_py.onnx_output_bottom_pin_location import begain_output_bottom_pin_location
from packagefiles.PackageExtract.yolox_onnx_py.onnx_detect_pin import onnx_output_pairs_data_pin_5
from packagefiles.PackageExtract.yolox_onnx_py.onnx_yolox_output_waikuang import onnx_output_waikuang
import cv2
import numpy as np
import math
import statistics
# 全局路径
DATA = 'Result/Package_extract/data'
DATA_BOTTOM_CROP = 'Result/Package_extract/data_bottom_crop'
DATA_COPY = 'Result/Package_extract/data_copy'
ONNX_OUTPUT = 'Result/Package_extract/onnx_output'
OPENCV_OUTPUT = 'Result/Package_extract/opencv_output'
OPENCV_OUTPUT_LINE = 'Result/Package_extract/opencv_output_yinXian'
BGA_BOTTOM = 'Result/Package_extract/bga_bottom'
PINMAP = 'Result/Package_extract/pinmap'
YOLOX_DATA = 'Result/Package_extract/yolox_data'

def choose_x(binary):  # 这个函数用于筛选出所有长度大于某自适应阈值的横线，函数的输入和输出都是二值图。自适应阈值的大小为：最长的成对出现的线的长度的0.6
    height, width = binary.shape[:2]
    cnt_length = []
    length_cnt = []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = w
        cnt_length.append(contour)
        length_cnt.append(length)

    length_num = sorted(length_cnt, reverse=True)
    mx = min(height,
             width) * 0.8  # 线的长度不能长过图片宽度的0.8，否则认为是类似package外框那样的东西（如果图片是经过package坐标矫正然后裁剪来的并且去除了外边框，那么可能就不需要这一阈值了）
    th = 0
    for i in range(len(length_num) - 1):
        if length_num[i] <= mx and length_num[i] / length_num[i + 1] <= 1.05:  # 找出最长的成对出现（长度差不超过0.05）的线。
            th = length_num[i] * 0.6  # 变量th就是那个自适应的阈值
            break

    cnts = []
    for cnt, lgt in zip(cnt_length, length_cnt):
        if mx >= lgt >= th:  # 根据阈值筛选线。
            cnts.append(cnt)

    binary_image = np.zeros((height, width, 1), dtype=np.uint8)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 10:
            cv2.line(binary_image, (int(x - (w * 0.05)), int(y + (h * 0.5))), (int(x + (w * 1.05)), int(y + (h * 0.5))),
                     (255), 1)  # 将筛选出来的线画到空白图上并返回.线必须要延长一些，这样就算角是缺的，两条边也能相交，
        else:
            cv2.line(binary_image, (int(x - (w * 0.05)), int(y)), (int(x + (w * 1.05)), int(y)), (255), 1)
            cv2.line(binary_image, (int(x - (w * 0.05)), int(y + h)), (int(x + (w * 1.05)), int(y + h)), (255), 1)
    return binary_image


def choose_y(binary):  # 这个函数用于筛选出所有长度大于某自适应阈值的竖线，与上面那个横线的一样
    height, width = binary.shape[:2]
    cnt_length = []
    length_cnt = []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = h
        cnt_length.append(contour)
        length_cnt.append(length)

    length_num = sorted(length_cnt, reverse=True)
    mx = min(height, width) * 0.8
    th = 0
    for i in range(len(length_num) - 1):
        if length_num[i] <= mx and length_num[i] / length_num[i + 1] <= 1.05:
            th = length_num[i] * 0.6
            break

    cnts = []
    for cnt, lgt in zip(cnt_length, length_cnt):
        if mx >= lgt >= th:
            cnts.append(cnt)

    binary_image = np.zeros((height, width, 1), dtype=np.uint8)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10:
            cv2.line(binary_image, (int(x + (w * 0.5)), int(y - (h * 0.05))), (int(x + (w * 0.5)), int(y + (h * 1.05))),
                     (255), 1)
        else:
            cv2.line(binary_image, (int(x), int(y - (h * 0.05))), (int(x), int(y + (h * 1.05))), (255), 1)
            cv2.line(binary_image, (int(x + w), int(y - (h * 0.05))), (int(x + w), int(y + (h * 1.05))), (255), 1)
    return binary_image


def waibiankuang(img):  # 这个函数实现主要功能，输入图片，输出bga的顶部和底部的外边框的坐标，这个坐标大部分情况下是精确的，有些时候可能稍有偏差。
    src_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 下面这十几行用腐蚀膨胀的方法从img中提取出所有横线和竖线
    AdaptiveThreshold = cv2.bitwise_not(src_img1)

    # AdaptiveThreshold = cv2.adaptiveThreshold(AdaptiveThreshold, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
    thresh, AdaptiveThreshold = cv2.threshold(AdaptiveThreshold, 122, 255, 0)  # 建议用固定阈值而不是自适应阈值

    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()

    horizontalSize = int(horizontal.shape[1] / 32)  # 横线和竖线的长度需大于图片宽、高的1/8
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    verticalSize = int(vertical.shape[0] / 32)  # 横线和竖线的长度需大于图片宽、高的1/8
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    horizontal = choose_x(horizontal)  # choose_x这个函数用于筛选出所有长度大于某自适应阈值的横线，函数的输入和输出都是二值图
    vertical = choose_y(vertical)  # choose_y这个函数用于筛选出所有长度大于某自适应阈值的竖线，函数的输入和输出都是二值图
    mask = horizontal + vertical

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    height, width = img.shape[:2]
    binary_image = np.zeros((height, width, 1), dtype=np.uint8)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        area_cnt = cv2.contourArea(cnt)
        if w > width / 32 and h > height / 32 and area_cnt / area >= 0.95:  # 筛选出长宽都足够大，并且是方形的轮廓。轮廓面积除以其外接矩形的面积，越接近1说明越接近方形
            cv2.rectangle(binary_image, (x, y), (x + w, y + h), (255), 3)

    contours2, hierarchy2 = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 下面这几行用于合并重叠的矩形
    wbk = []
    for cnt in contours2:
        x, y, w, h = cv2.boundingRect(cnt)
        wbk.append((x + 3, y + 3, x + w - 4, y + h - 4))  # 这里减3是因为第101行画框时线宽为3
    return wbk


def get_waikaung():
    source_folder = r'Result/Package_extract/bga'
    target_folder = f'{BGA_BOTTOM}'

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img_path = os.path.join(source_folder, filename)
            src_img = cv2.imread(img_path)

            wbk = waibiankuang(src_img)

            # rec = src_img.copy()
            no = 1
            for wk in wbk:
                x1, y1, x2, y2 = wk
                cropped_img = src_img[y1:y2, x1:x2]
                # cv2.rectangle(rec, (x1, y1), (x2, y2), (255, 0, 0), 3)
                # print(target_folder + '/' + str(no) + '.jpg')
                cv2.imwrite(target_folder + '/' + str(no) + '.jpg', cropped_img)
                no += 1
            if no == 1:
                print("在BGA三视图中未找到外框")
            # target_path = os.path.join(target_folder, filename)
            # cv2.imwrite(target_path, rec)




def choose_x(binary):
    height, width = binary.shape[:2]
    cnt_length = []
    length_cnt = []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = w
        cnt_length.append(contour)
        length_cnt.append(length)

    length_num = sorted(length_cnt, reverse=True)
    th = 0
    for i in range(len(length_num) - 1):
        if length_num[i] <= width * 0.8 and length_num[i] / length_num[i + 1] <= 1.05:
            th = length_num[i] * 0.6
            break

    cnts = []
    for cnt, lgt in zip(cnt_length, length_cnt):
        if width * 0.8 >= lgt >= th:
            cnts.append(cnt)

    binary_image = np.zeros((height, width, 1), dtype=np.uint8)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.line(binary_image, (int(x - (w * 0.05)), int(y + (h * 0.5))), (int(x + (w * 1.05)), int(y + (h * 0.5))),
                 (255), 1)

    return binary_image


def choose_y(binary):
    height, width = binary.shape[:2]
    cnt_length = []
    length_cnt = []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = h
        cnt_length.append(contour)
        length_cnt.append(length)

    length_num = sorted(length_cnt, reverse=True)
    th = 0
    for i in range(len(length_num) - 1):
        if length_num[i] <= height * 0.8 and length_num[i] / length_num[i + 1] <= 1.05:
            th = length_num[i] * 0.6
            break

    cnts = []
    for cnt, lgt in zip(cnt_length, length_cnt):
        if height * 0.8 >= lgt >= th:
            cnts.append(cnt)

    binary_image = np.zeros((height, width, 1), dtype=np.uint8)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.line(binary_image, (int(x + (w * 0.5)), int(y - (h * 0.05))), (int(x + (w * 0.5)), int(y + (h * 1.05))),
                 (255), 1)

    return binary_image


def output_body(img_path):
    src_img = cv2.imread(img_path)
    src_img1 = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    AdaptiveThreshold = cv2.bitwise_not(src_img1)
    thresh, AdaptiveThreshold = cv2.threshold(AdaptiveThreshold, 10, 255, 0)

    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()

    horizontalSize = int(horizontal.shape[1] / 8)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    verticalSize = int(vertical.shape[0] / 8)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    horizontal = choose_x(horizontal)
    vertical = choose_y(vertical)
    mask = horizontal + vertical

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    height, width = src_img.shape[:2]
    binary_image = np.zeros((height, width, 1), dtype=np.uint8)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        area_cnt = cv2.contourArea(cnt)
        if w > width / 8 and h > height / 8 and area_cnt / area >= 0.95:
            cv2.rectangle(binary_image, (x, y), (x + w, y + h), (255), 3)

    contours2, hierarchy2 = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rec = src_img.copy()
    for cnt in contours2:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(rec, (x, y), (x + w - 3, y + h - 3), (255, 0, 0), 3)
    # cv2.namedWindow('body', 0)
    # cv2.imshow('body', rec)
    # cv2.waitKey(0)
    try:
        location = np.array([[x + int(w / 40), y + int(h / 40), x + w - int(w / 40), y + h - int(h / 40)]])
    except:
        location = np.array([])
        print("opencv函数找不到外框")
    return location


def crop_img_save(path_img, path_crop, x_min, y_min, x_max, y_max):
    img = cv2.imread(path_img)
    cropped = img[max(y_min, 0):min(y_max, img.shape[0]),
              max(x_min, 0):min(x_max, img.shape[1])]  # 裁剪坐标为[y0:y1, x0:x1]必须为整数
    cv2.imwrite(path_crop, cropped)
    print("保存图", path_crop)


def get_pinmap():
    # 由用户决定是否使用人工框选pinmap
    # 创建文件夹
    location = output_body(f'{BGA_BOTTOM}/bottom.jpg')
    print("location", location)
    path_img = f'{BGA_BOTTOM}/bottom.jpg'
    path_crop = f'{DATA_BOTTOM_CROP}/pinmap.jpg'

    if not np.array_equal(location, np.array([])):
        print("在单张bottom中找到外框")

        x_min = min(location[0, 0], location[0, 2])
        y_min = min(location[0, 1], location[0, 3])
        x_max = max(location[0, 0], location[0, 2])
        y_max = max(location[0, 1], location[0, 3])
        crop_img_save(path_img, path_crop, x_min, y_min, x_max, y_max)
        shutil.copy(path_crop, path_img)


        # pin = onnx_output_pairs_data_pin_5(path_crop)
        # # pin中找x_min和y_min和x_max和y_max
        # x_min = 9999
        # y_min = 9999
        # x_max = 0
        # y_max = 0
        # for i in range(len(pin)):
        #     if pin[i][0] < x_min:
        #         x_min = pin[i][0]
        #     if pin[i][1] < y_min:
        #         y_min = pin[i][1]
        #     if pin[i][2] > x_max:
        #         x_max = pin[i][2]
        #     if pin[i][3] > y_max:
        #         y_max = pin[i][3]
        # # 判断x_min和y_min和x_max和y_max所占的面积是否占比超过0.4
        # img = cv2.imread(f'{DATA_BOTTOM_CROP}/pinmap.jpg')
        # if (x_max - x_min) * (y_max - y_min) / img.shape[0] * img.shape[1] > 0.4:
        #     crop_img_save(f'{DATA_BOTTOM_CROP}/pinmap.jpg', f'{DATA_BOTTOM_CROP}/pinmap.jpg', max(int(x_min) - 1, 0),
        #                   max(int(y_min) - 1, 0), min(int(x_max) + 1, img.shape[1]),
        #                   min(int(y_max) + 1, img.shape[0]))
        #     print("经过yolox辅助得到pinmap")
        # except:
        #     pass
    else:
        print("在单张bottom中找不到外框")
        shutil.copy(path_crop, path_img)

        # pin_map_limation_1 = begain_output_bottom_pin_location()
        # path_img = 'bga_bottom/bottom.jpg'
        # # filter_black_point(path_img, path_img)
        # path_crop = 'data_bottom_crop/pinmap.jpg'
        # x_min = int(pin_map_limation_1[0][0])
        # y_min = int(pin_map_limation_1[0][1])
        # x_max = int(pin_map_limation_1[0][2])
        # y_max = int(pin_map_limation_1[0][3])
        # crop_img_save(path_img, path_crop, x_min, y_min, x_max, y_max)
        # np.savetxt('yolox_data/pin_map_limation.txt', pin_map_limation_1, delimiter=',')

        # pin = onnx_output_pairs_data_pin_5(path_crop)
        # # pin中找x_min和y_min和x_max和y_max
        # x_min = 9999
        # y_min = 9999
        # x_max = 0
        # y_max = 0
        # for i in range(len(pin)):
        #     if pin[i][0] < x_min:
        #         x_min = pin[i][0]
        #     if pin[i][1] < y_min:
        #         y_min = pin[i][1]
        #     if pin[i][2] > x_max:
        #         x_max = pin[i][2]
        #     if pin[i][3] > y_max:
        #         y_max = pin[i][3]
        # # 判断x_min和y_min和x_max和y_max所占的面积是否占比超过0.4
        # img = cv2.imread(f'{DATA_BOTTOM_CROP}/pinmap.jpg')
        # if (x_max - x_min) * (y_max - y_min) / img.shape[0] * img.shape[1] > 0.4:
        #     crop_img_save(f'{DATA_BOTTOM_CROP}/pinmap.jpg', f'{DATA_BOTTOM_CROP}/pinmap.jpg', max(int(x_min) - 1, 0),
        #                   max(int(y_min) - 1, 0), min(int(x_max) + 1, img.shape[1]),
        #                   min(int(y_max) + 1, img.shape[0]))
        #     print("经过yolox辅助得到pinmap")
    # 清空文件夹


def filter_a_b_old(a):
    '''
    a list[int,int]
    滤波知道空白占比达到0.5
    '''
    # 统计平均值
    mean1 = int(sum(a) / len(a))  # 高于mean1的值重设为mean1
    for i in range(len(a)):
        if a[i] > mean1:
            a[i] = mean1

    zer_num = 0
    ratio = 0.6
    key = 0
    while zer_num / len(a) < 0.5:
        key += 1
        mean2 = int(sum(a) * (ratio + 0.05 * key) / len(a))  # 低于mean2的值重设为0
        # print(ratio + 0.05 * key)
        for i in range(len(a)):
            if a[i] < mean2:
                a[i] = 0
        # 统计a中为0数量

        for i in range(len(a)):
            if a[i] == 0:
                zer_num += 1

    # 清除孤立的冲激
    try:
        for i in range(len(a)):
            if i > 0:
                if a[i] > 0 and a[i - 1] == 0 and a[i + 1] == 0:
                    a[i] == 0
    except:
        pass
    # 统计每一组的平均占用像素点数量
    mean3 = []
    every_pin = []  # 记录每组起始和终点位置
    next_i = 0
    for i in range(len(a)):
        if i >= next_i:
            if a[i] > 0:
                long = 1  # 每组占用像素点数量
                for j in range(len(a)):
                    if j > i:
                        if a[j] > 0:
                            long += 1
                        else:
                            break
                next_i = j
                mid = {"from": i, "to": (next_i - 1)}
                every_pin.append(mid)
                mean3.append(long)
    # print(mean3)
    # print(every_pin)
    if len(mean3) != 0:
        mean4 = sum(mean3) / len(mean3)
    else:
        # 主动抛出异常"pin数量太少，算法不适用，请收集改pdf并反馈以便修复"

        raise ValueError("pin数量太少，算法不适用，请记录此pdf并反馈以便修复")

    # print(mean4)
    # 消除过短的
    for i in range(len(mean3)):
        if mean3[i] / mean4 < 0.31:
            for j in range(len(a)):
                if every_pin[i]['from'] <= j <= every_pin[i]['to']:
                    a[j] = 0

    return a


def filter_a_b(a):
    '''
    a list[int,int]
    滤波知道空白占比达到0.5
    '''
    # 统计平均值
    mean1 = int(sum(a) / len(a))  # 高于mean1的值重设为mean1
    for i in range(len(a)):
        if a[i] > mean1:
            a[i] = mean1

    zer_num = 0
    ratio = 0.6
    key = 0
    while zer_num / len(a) < 0.5:
        key += 1
        mean2 = int(sum(a) * (ratio + 0.05 * key) / len(a))  # 低于mean2的值重设为0
        # print(ratio + 0.05 * key)
        for i in range(len(a)):
            if a[i] < mean2:
                a[i] = 0
        # 统计a中为0数量

        for i in range(len(a)):
            if a[i] == 0:
                zer_num += 1

    # 清除孤立的冲激
    try:
        for i in range(len(a)):
            if i > 0:
                if a[i] > 0 and a[i - 1] == 0 and a[i + 1] == 0:
                    a[i] == 0
    except:
        pass
    # 统计每一组的平均占用像素点数量
    mean3 = []
    every_pin = []  # 记录每组起始和终点位置
    next_i = 0
    for i in range(len(a)):
        if i >= next_i:
            if a[i] > 0:
                long = 1  # 每组占用像素点数量
                for j in range(len(a)):
                    if j > i:
                        if a[j] > 0:
                            long += 1
                        else:
                            break
                next_i = j
                mid = {"from": i, "to": (next_i - 1)}
                every_pin.append(mid)
                mean3.append(long)
    # print(mean3)
    # print(every_pin)
    if len(mean3) != 0:
        mean4 = sum(mean3) / len(mean3)
    else:
        # 主动抛出异常"pin数量太少，算法不适用，请收集改pdf并反馈以便修复"

        raise ValueError("pin数量太少，算法不适用，请记录此pdf并反馈以便修复")

    # print(mean4)
    # 消除过短的
    for i in range(len(mean3)):
        if mean3[i] / mean4 < 0.31:
            for j in range(len(a)):
                if every_pin[i]['from'] <= j <= every_pin[i]['to']:
                    a[j] = 0

    return a


def find_line_old(a):
    '''
    a:记录滤波后x轴或者y轴上每一个像素位置的像素投影 list[int,int]
    '''
    # 寻找突变点
    c = []  # 记录突然变小的点
    mean1 = int(sum(a) / len(a))
    for i in range(len(a) - 1):
        if a[i] > 0 and a[i + 1] == 0:
            c.append(mean1 * 3)
        else:
            c.append(0)
    c.append(0)

    d = []
    for i in range(len(a) - 1):
        if a[i - 1] == 0 and a[i] > 0:
            d.append(mean1 * 3)
        else:
            d.append(0)
    d.append(0)

    # 取两个突变点的中间
    e = -1
    f = -1
    g = [0 for i in range(len(c))]
    h = 0
    x = 0
    for i in range(len(c)):
        if c[i] != 0:
            e = i
        if d[i] != 0:
            f = i
        if e != -1 and f != -1 and a[int((e + f) * 0.5)] == 0:
            g[int((e + f) * 0.5)] = mean1 * 9
            h += abs(e - f)
            x += 1
    # h = int(h/x)
    # g_copy = g.copy()
    # for i in range(len(g)):
    #     if g[i] > 0:
    #         for j in range(len(g)):
    #             if j > i and g[j] > 0:
    #                 if j - i > 4 * h:
    #                     g_copy[int((i + j) * 0.5)] = mean1 * 9
    #                 break
    # 根据网格线的普遍规律，重组为规范的网格
    # print("g", g)
    wangge = []  # 记录每个网格的位置
    for i in range(len(g)):
        if g[i] > 0:
            wangge.append(i)

    # import statistics

    # num = statistics.mode(wangge)
    # print("众数为:", num)

    num = 0
    for i in range(len(wangge) - 1):
        num += wangge[i + 1] - wangge[i]
    num = int(num / (len(wangge) - 1))  # 网格的普遍间隔
    # print("num", num)
    # 根据网格的普遍间隔，
    '''
    1.找符合网格普遍间隔的线
    2.针对这些线往左右看，推理出正确位置
    '''
    new_wangge = []
    for i in range(len(wangge) - 1):
        if abs((wangge[i + 1] - wangge[i]) - num) <= 3:
            if wangge[i] not in new_wangge:
                new_wangge.append(wangge[i])
            if wangge[i + 1] not in new_wangge:
                new_wangge.append(wangge[i + 1])
    # print("wangge", wangge)
    # print("new_wangge", new_wangge)
    out_key = 0
    acc = 0
    while out_key != 1:
        loc = 0
        new_new_wangge = []
        for i in range(len(new_wangge) - 1):
            if i == len(new_wangge) - 2:
                out_key = 1
            if acc == 1:
                if i == loc:
                    if abs(new_wangge[i + 1] - new_wangge[i] - num) / num > 0.5:
                        loc_1 = int(new_wangge[i] + num)
                        loc_2 = int(new_wangge[i] + num)
                        loc_1n = 0
                        loc_2n = 0
                        try:
                            while a[loc_1] != 0:
                                loc_1 += 1
                                loc_1n += 1
                                if loc_1 == int(new_wangge[i + 1] - 0.5 * num):
                                    loc_1 = -1
                                    loc_1n = 999
                                    break
                        except:
                            loc_1 -= 1
                        try:
                            while a[loc_2] != 0:
                                loc_2 -= 1
                                loc_2n += 1
                                if loc_2 == int(new_wangge[i] + 0.5 * num):
                                    loc_2 = -1
                                    loc_2n = 999
                                    break
                        except:
                            loc_2 += 1
                        if loc_1 == loc_2 == -1:
                            loc = int((new_wangge[i] + new_wangge[i + 1]) * 0.5)
                        else:
                            if loc_1n < loc_2n:
                                loc = loc_1
                            else:
                                loc = loc_2
                        new_wangge.append(loc)
                        new_wangge.sort()
                        break
            else:

                if abs(new_wangge[i + 1] - new_wangge[i] - num) / num > 0.5:
                    acc = 1
                    loc_1 = int(new_wangge[i] + num)
                    loc_2 = int(new_wangge[i] + num)
                    loc_1n = 0
                    loc_2n = 0
                    try:
                        while a[loc_1] != 0:
                            loc_1 += 1

                            loc_1n += 1
                            if loc_1 == int(new_wangge[i + 1] - 0.5 * num):
                                loc_1 = -1
                                loc_1n = 999
                                break
                    except:
                        loc_1 -= 1
                    try:
                        while a[loc_2] != 0:
                            loc_2 -= 1
                            loc_2n += 1
                            if loc_2 == int(new_wangge[i] + 0.5 * num):
                                loc_2 = -1
                                loc_2n = 999
                                break
                    except:
                        loc_2 += 1
                    if loc_1 == loc_2 == -1:
                        loc = int((new_wangge[i] + new_wangge[i + 1]) * 0.5)
                    else:
                        if loc_1n < loc_2n:
                            loc = loc_1
                        else:
                            loc = loc_2
                    new_wangge.append(loc)
                    new_wangge.sort()
                    break
            # new_wangge = (new_new_wangge + new_wangge)

    # print("new_wangge", new_wangge)
    # 按照规律扩展到全图
    leng = len(g)
    while new_wangge[0] > num:
        loc_1 = int(new_wangge[0] - num)
        loc_2 = int(new_wangge[0] - num)
        loc_1n = 0
        loc_2n = 0
        try:
            while a[loc_1] != 0:
                loc_1 -= 1
                loc_1n += 1
        except:
            loc_1 += 1
        try:
            while a[loc_2] != 0:
                loc_2 += 1
                loc_2n += 1
        except:
            loc_2 -= 1
        if loc_1n < loc_2n:
            loc = loc_1
        else:
            loc = loc_2
        new_wangge.append(loc)
        new_wangge.sort()
    while leng - 1 - new_wangge[len(new_wangge) - 1] > num:
        loc_1 = int(new_wangge[len(new_wangge) - 1] + num)
        loc_2 = int(new_wangge[len(new_wangge) - 1] + num)
        loc_1n = 0
        loc_2n = 0
        try:
            while a[loc_1] != 0:
                loc_1 += 1
                loc_1n += 1
        except:
            loc_1 -= 1
        try:
            while a[loc_2] != 0:
                loc_2 -= 1
                loc_2n += 1
        except:
            loc_2 += 1
        if loc_1n < loc_2n:
            loc = loc_1
        else:
            loc = loc_2
        new_wangge.append(loc)
        new_wangge.sort()
    if new_wangge[0] / num > 0.8:
        new_wangge.append(0)
        new_wangge.sort()
    if (leng - 1 - new_wangge[len(new_wangge) - 1]) / num > 0.8:
        new_wangge.append(leng - 1)
    # print("new_wangge", new_wangge)
    new_g = [0 for i in range(len(c))]
    for i in range(len(new_g)):
        if i in new_wangge:
            new_g[i] = 9999
    # print("new_g", new_g)
    return new_g


def remove_min_max(lst):
    if len(lst) < 4:
        print("列表中元素不足4个，无法删除最小值和最大值")
        return lst

    # 找到最小值和最大值
    min_value = min(lst)
    max_value = max(lst)

    # 删除最小值
    lst.remove(min_value)

    # 删除最大值
    lst.remove(max_value)

    return lst




def find_line(a, ave_width, test_mode):

    '''
    a:记录滤波后x轴或者y轴上每一个像素位置的像素投影 list[int,int]
    '''
    # test_mode = False  # 测试模式
    # 寻找突变点
    c = []  # 记录突然变小的点
    mean1 = int(sum(a) / len(a))
    for i in range(len(a) - 1):
        if a[i] > 0 and a[i + 1] == 0:
            c.append(mean1 * 3)
        else:
            c.append(0)
    c.append(0)
    if test_mode:
        print("c", c)
        print("len(c)", len(c))

    d = [] # 记录突然变大的点
    d.append(0)
    for i in range(1, len(a)):
        if a[i - 1] == 0 and a[i] > 0:
            d.append(mean1 * 3)
        else:
            d.append(0)
    # 如果一个点既突然变大又突然变小，删除该点
    for i in range(len(c)):
        if c[i] != 0 and d[i] != 0:
            c[i] = 0
            d[i] = 0
    if test_mode:
        print("d", d)
        print("len(d)", len(d))
    # 取两个突变点的中间
    # e = -1
    # f = -1
    g = [0 for i in range(len(c))]
    # h = 0
    # x = 0

    for i in range(len(c)):
        if c[i] > 0:
            e = i
            for j in range(i + 1, len(c)):
                if d[j] != 0:
                    f = j
                    # 当整行整列缺pin时，网格不能划分为中间
                    if abs(e - f) > 2 * ave_width:
                        g[e + 1] = mean1 * 9
                        g[f - 1] = mean1 * 9
                        break
                    else:
                        g[int((e + f) * 0.5)] = mean1 * 9
                        # h += abs(e - f)
                        # x += 1
                        break
    if test_mode:
        print("g", g)
        print("len(g)", len(g))
    # 根据网格线的普遍规律，重组为规范的网格
    # 根据相邻网格差值的众数作为平均网格宽度，划分网格
    wangge = []  # 记录每个网格的位置
    find_num_list = []
    for i in range(len(g)):
        if g[i] > 0:
            wangge.append(i)

    for i in range(len(wangge) - 1):
        find_num_list.append(abs(wangge[i] - wangge[i + 1]))
    if test_mode:
        print("find_num_list", find_num_list)

    find_num_list = remove_min_max(find_num_list)
    num = statistics.mode(find_num_list)
    if test_mode:
        print("网格间距众数为:", num)

    # num = 0
    # for i in range(len(wangge) - 1):
    #     num += wangge[i + 1] - wangge[i]
    # num = int(num / (len(wangge) - 1))  # 网格的普遍间隔
    if test_mode:
        print("网格的普遍间隔num", num)
    '''
    1.找符合网格普遍间隔的线
    2.针对这些线往左右看，推理出正确位置
    '''
    # img = cv2.imread(f'{DATA_BOTTOM_CROP}/pinmap.jpg')
    new_wangge = []
    # for i in range(len(wangge) - 1):
    #     # print(round(img.shape[0]/200))
    #     # print(abs((wangge[i + 1] - wangge[i]) - num))
    #     # print(round(img.shape[0]/100))
    #     if round(img.shape[0] / 200) <= abs((wangge[i + 1] - wangge[i]) - num) <= round(img.shape[0] / 100):
    #         if wangge[i] not in new_wangge:
    #             new_wangge.append(wangge[i])
    #         if wangge[i + 1] not in new_wangge:
    #             new_wangge.append(wangge[i + 1])
    if test_mode:
        print("wangge", wangge)
        print("new_wangge", new_wangge)
    # 解决筛选网格为空造成的死循环问题
    if new_wangge == []:
        new_wangge = wangge
    for m in range(5):
        out_key = 0
        acc = 0
        while out_key != 1:
            loc = 0  # 记录需要插入的新位置
            new_new_wangge = []
            for i in range(len(new_wangge) - 1):
                if i == len(new_wangge) - 2:
                    out_key = 1
                if acc == 1:
                    if i == loc:
                        if abs(new_wangge[i + 1] - new_wangge[i] - num) / num > 0.5:
                            loc_1 = int(new_wangge[i] + num)
                            loc_2 = int(new_wangge[i] + num)
                            loc_1n = 0
                            loc_2n = 0
                            try:
                                while a[loc_1] != 0:
                                    loc_1 += 1
                                    loc_1n += 1
                                    if loc_1 == int(new_wangge[i + 1] - 0.5 * num):
                                        loc_1 = -1
                                        loc_1n = 999
                                        break
                            except:
                                loc_1 -= 1
                            try:
                                while a[loc_2] != 0:
                                    loc_2 -= 1
                                    loc_2n += 1
                                    if loc_2 == int(new_wangge[i] + 0.5 * num):
                                        loc_2 = -1
                                        loc_2n = 999
                                        break
                            except:
                                loc_2 += 1
                            if loc_1 == loc_2 == -1:
                                loc = int((new_wangge[i] + new_wangge[i + 1]) * 0.5)
                            else:
                                if loc_1n < loc_2n:
                                    loc = loc_1
                                else:
                                    loc = loc_2
                            new_wangge.append(loc)
                            new_wangge.sort()
                            break
                else:

                    if abs(new_wangge[i + 1] - new_wangge[i] - num) / num > 0.5:
                        acc = 1
                        loc_1 = int(new_wangge[i] + num)
                        loc_2 = int(new_wangge[i] + num)
                        loc_1n = 0
                        loc_2n = 0
                        try:
                            while a[loc_1] != 0:
                                loc_1 += 1

                                loc_1n += 1
                                if loc_1 == int(new_wangge[i + 1] - 0.5 * num):
                                    loc_1 = -1
                                    loc_1n = 999
                                    break
                        except:
                            loc_1 -= 1
                        try:
                            while a[loc_2] != 0:
                                loc_2 -= 1
                                loc_2n += 1
                                if loc_2 == int(new_wangge[i] + 0.5 * num):
                                    loc_2 = -1
                                    loc_2n = 999
                                    break
                        except:
                            loc_2 += 1
                        if loc_1 == loc_2 == -1:
                            loc = int((new_wangge[i] + new_wangge[i + 1]) * 0.5)
                        else:
                            if loc_1n < loc_2n:
                                loc = loc_1
                            else:
                                loc = loc_2
                        new_wangge.append(loc)
                        new_wangge.sort()
                        break

    if test_mode:
        print("new_wangge", new_wangge)
    # 为网格找到起始线
    leng = len(g)
    while new_wangge and new_wangge[0] > num:
        loc_1 = int(new_wangge[0] - num)
        loc_2 = int(new_wangge[0] - num)
        loc_1n = 0
        loc_2n = 0
        try:
            while a[loc_1] != 0:
                loc_1 -= 1
                loc_1n += 1
        except:
            loc_1 += 1
        try:
            while a[loc_2] != 0:
                loc_2 += 1
                loc_2n += 1
        except:
            loc_2 -= 1
        if loc_1n < loc_2n or abs(new_wangge[0] - loc_2) < num * 0.5:
            loc = loc_1
        else:
            loc = loc_2
        new_wangge.append(loc)
        if test_mode:
            print("new_wangge", new_wangge)
        new_wangge.sort()
    while leng - 1 - new_wangge[len(new_wangge) - 1] > num:
        loc_1 = int(new_wangge[len(new_wangge) - 1] + num)
        loc_2 = int(new_wangge[len(new_wangge) - 1] + num)
        loc_1n = 0
        loc_2n = 0
        try:
            while a[loc_1] != 0:
                loc_1 += 1
                loc_1n += 1
        except:
            loc_1 -= 1
        try:
            while a[loc_2] != 0:
                loc_2 -= 1
                loc_2n += 1
        except:
            loc_2 += 1
        if loc_1n < loc_2n or abs(new_wangge[len(new_wangge) - 1] - loc_2) < num * 0.5:
            loc = loc_1
        else:
            loc = loc_2
        new_wangge.append(loc)
        if test_mode:
            print("new_wangge", new_wangge)
        new_wangge.sort()
    if new_wangge[0] / num > 0.8:
        new_wangge.append(0)
        new_wangge.sort()
    # 为网格找到终点线
    if (leng - 1 - new_wangge[len(new_wangge) - 1]) / num > 0.8:
        new_wangge.append(leng - 1)
    if test_mode:
        print("new_wangge", new_wangge)
    new_g = [0 for i in range(len(c))]
    for i in range(len(new_g)):
        if i in new_wangge:
            new_g[i] = 9999
    if test_mode:
        print("new_g", new_g)
    return new_g


def empty_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except FileNotFoundError:
        print(f"文件夹 {folder_path} 不存在！无须删除文件夹")


def get_pin_grid_old(a, b, c, d):
    x_min = -1
    # print(a)
    x_max = -1
    for i in range(len(a)):
        if a[i] > 0:
            if x_min == -1:
                x_min = i
            x_max = i
    y_min = -1
    y_max = -1
    for i in range(len(b)):
        if b[i] > 0:
            if y_min == -1:
                y_min = i
            y_max = i
    c[x_min] = 100
    c[x_max] = 100
    d[y_min] = 100
    d[y_max] = 100
    lie_line = []  # 记录竖向直线所在位置
    lie_num = 0
    for i in range(len(c)):
        if c[i] > 0:
            lie_num += 1
            lie_line.append(i)
    # print(lie_line)
    hang_line = []  # 记录横向直线所在位置
    hang_num = 0
    for i in range(len(d)):
        if d[i] > 0:
            hang_num += 1
            hang_line.append(i)
    # print(hang_line)
    # print("lie_line", lie_line)
    # print("hang_line", hang_line)
    # 将距离相近的行列线合并取中间没有pin的位置
    length = []  # 存储每组网格之间的距离
    for i in range(len(lie_line) - 1):
        length.append(int(lie_line[i + 1] - lie_line[i]))
    average_lie_length = sum(length) / len(length)
    # print("average_lie_length", average_lie_length)
    for i in range(len(lie_line) - 1):
        if (lie_line[i + 1] - lie_line[i]) / average_lie_length < 0.5:
            new_lie_line_location = lie_line[i] + 1

            while a[new_lie_line_location] != 0:
                if new_lie_line_location == lie_line[i] + 1:
                    break
                new_lie_line_location += 1
            lie_line[i] = new_lie_line_location
            lie_line[i + 1] = new_lie_line_location
            lie_line.sort()
    lie_line = list(set(lie_line))
    lie_line.sort()
    # print("lie_line", lie_line)

    length = []  # 存储每组网格之间的距离
    for i in range(len(hang_line) - 1):
        length.append(int(hang_line[i + 1] - hang_line[i]))
    average_hang_length = sum(length) / len(length)
    for i in range(len(hang_line) - 1):
        if (hang_line[i + 1] - hang_line[i]) / average_hang_length < 0.5:
            new_hang_line_location = hang_line[i] + 1

            while a[new_hang_line_location] != 0:
                if new_hang_line_location == hang_line[i] + 1:
                    break
                new_hang_line_location += 1
            hang_line[i] = new_hang_line_location
            hang_line[i + 1] = new_hang_line_location
            # hang_line = list(set(hang_line))
            hang_line.sort()
    hang_line = list(set(hang_line))
    hang_line.sort()
    # print("hang_line", hang_line)

    hang_num = len(hang_line)
    lie_num = len(lie_line)
    # print("行数", hang_num)
    # print("列数", lie_num)
    pin_map = np.zeros((hang_num - 1, lie_num - 1, 5))
    no = 0
    for i in range(len(lie_line) - 1):
        pin_map[:, no, 0] = lie_line[i]
        pin_map[:, no, 2] = lie_line[i + 1]
        no += 1
    no = 0
    for i in range(len(hang_line) - 1):
        pin_map[no, :, 1] = hang_line[i]
        pin_map[no, :, 3] = hang_line[i + 1]
        no += 1

    # print(pin_map)
    return pin_map


def get_pin_grid(a, b, c, d, test_mode):
    x_min = -1
    # print(a)
    x_max = -1
    for i in range(len(a)):
        if a[i] > 0:
            if x_min == -1:
                x_min = i
            x_max = i
    y_min = -1
    y_max = -1
    for i in range(len(b)):
        if b[i] > 0:
            if y_min == -1:
                y_min = i
            y_max = i
    c[x_min] = 100
    c[x_max] = 100
    d[y_min] = 100
    d[y_max] = 100
    lie_line = []  # 记录竖向直线所在位置
    lie_num = 0
    for i in range(len(c)):
        if c[i] > 0:
            lie_num += 1
            lie_line.append(i)

    # print(lie_line)
    hang_line = []  # 记录横向直线所在位置
    hang_num = 0
    for i in range(len(d)):
        if d[i] > 0:
            hang_num += 1
            hang_line.append(i)

    # print(hang_line)
    # print("lie_line", lie_line)
    # print("hang_line", hang_line)
    # 将距离相近的行列线合并取中间没有pin的位置
    length = []  # 存储每组网格之间的距离
    for i in range(len(lie_line) - 1):
        length.append(int(lie_line[i + 1] - lie_line[i]))
    average_lie_length = sum(length) / len(length)
    if test_mode:
        print("average_lie_length", average_lie_length)
    for i in range(len(lie_line) - 1):
        if (lie_line[i + 1] - lie_line[i]) / average_lie_length <= 0.55:
            if test_mode:
                print("len(a)", len(a))
                print("len(c)", len(c))
                print("lie_line[i] , lie_line[i+1]", lie_line[i], lie_line[i + 1])
            new_lie_line_location = lie_line[i] + 1

            while a[new_lie_line_location] != 0:
                if new_lie_line_location == lie_line[i] + 1:
                    break
                new_lie_line_location += 1
                if new_lie_line_location >= len(a) - 1:
                    break
            lie_line[i] = new_lie_line_location
            lie_line[i + 1] = new_lie_line_location
            lie_line.sort()
    lie_line = list(set(lie_line))
    lie_line.sort()
    # print("lie_line", lie_line)

    length = []  # 存储每组网格之间的距离
    for i in range(len(hang_line) - 1):
        length.append(int(hang_line[i + 1] - hang_line[i]))
    average_hang_length = sum(length) / len(length)
    for i in range(len(hang_line) - 1):
        if (hang_line[i + 1] - hang_line[i]) / average_hang_length <= 0.55:

            new_hang_line_location = hang_line[i] + 1

            while new_hang_line_location < len(a) and a[new_hang_line_location] != 0:
                if new_hang_line_location == hang_line[i] + 1:
                    break
                new_hang_line_location += 1
                if new_hang_line_location >= len(a) - 1:
                    break
            hang_line[i] = new_hang_line_location
            hang_line[i + 1] = new_hang_line_location
            # hang_line = list(set(hang_line))
            hang_line.sort()
    hang_line = list(set(hang_line))
    hang_line.sort()
    # print("hang_line", hang_line)

    hang_num = len(hang_line)
    lie_num = len(lie_line)
    # print("行数", hang_num)
    # print("列数", lie_num)
    pin_map = np.zeros((hang_num - 1, lie_num - 1, 5))
    no = 0
    for i in range(len(lie_line) - 1):
        pin_map[:, no, 0] = lie_line[i]
        pin_map[:, no, 2] = lie_line[i + 1]
        no += 1
    no = 0
    for i in range(len(hang_line) - 1):
        pin_map[no, :, 1] = hang_line[i]
        pin_map[no, :, 3] = hang_line[i + 1]
        no += 1

    # print(pin_map)
    return pin_map


def show_pin_grid(pin_map, c, d):
    '''
    展示网格
    '''

    # path = r'data_bottom_crop/pinmap.jpg'
    # img = cv2.imread(path)
    # point_color = (0, 255, 0)  # BGR
    # thickness = 5
    # lineType = 4
    #
    # for i in range(len(c)):
    #     if c[i] > 0:
    #         ptStart = (i, 0)
    #         ptEnd = (i, len(c))
    #         cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
    #
    # for i in range(len(d)):
    #     if d[i] > 0:
    #         ptStart = (0, i)
    #         ptEnd = (len(d), i)
    #         cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
    #
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    path = f'{DATA_BOTTOM_CROP}/pinmap.jpg'
    img = cv2.imread(path)
    point_color = (0, 255, 0)  # BGR
    thickness = 1
    lineType = 4
    for i in range(len(pin_map)):
        for j in range(len(pin_map[i])):
            try:
                ptStart = (int(pin_map[i][j][0]), int(pin_map[i][j][1]))
                ptEnd = (int(pin_map[i][j][0]), int(pin_map[i][j][3]))
                cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
                ptStart = (int(pin_map[i][j][2]), int(pin_map[i][j][1]))
                ptEnd = (int(pin_map[i][j][2]), int(pin_map[i][j][3]))
                cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
                ptStart = (int(pin_map[i][j][0]), int(pin_map[i][j][1]))
                ptEnd = (int(pin_map[i][j][2]), int(pin_map[i][j][1]))
                cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
                ptStart = (int(pin_map[i][j][0]), int(pin_map[i][j][3]))
                ptEnd = (int(pin_map[i][j][2]), int(pin_map[i][j][3]))
                cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
            except:
                pass
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_entropy(img_):
    x, y = img_.shape[0:2]
    img_ = cv2.resize(img_, (100, 100))  # 缩小的目的是加快计算速度
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    img = np.array(img_)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            # print(val)
            # print(tmp)
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if (tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res


# def
#     img = cv2.imread('image.jpg', 0)
#     # 将图像转换为灰度图
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 计算图像的直方图
#     hist, _ = np.histogram(gray, bins=256, range=[0, 256])
#     # 计算直方图的概率分布
#     prob = hist / (gray.shape[0] * gray.shape[1])
#     # 计算信息熵
#     entropy = -np.sum(prob * np.log2(prob + np.finfo(float).eps))
#     print('Image entropy:', entropy)
def cal_grid_shang(pin_map, test_mode):
    path = f'{DATA_BOTTOM_CROP}/pinmap.jpg'
    img = cv2.imread(path)
    new_pin_map = np.zeros((pin_map.shape[0], pin_map.shape[1]))
    for i in range(len(pin_map)):
        for j in range(len(pin_map[i])):
            img1 = img[int(pin_map[i][j][1]): int(pin_map[i][j][3]), int(pin_map[i][j][0]): int(pin_map[i][j][2])]
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            # 计算图像的直方图
            hist, _ = np.histogram(gray, bins=256, range=[0, 256])
            # 计算直方图的概率分布
            prob = hist / (gray.shape[0] * gray.shape[1])
            # 计算信息熵
            entropy = -np.sum(prob * np.log2(prob + np.finfo(float).eps))
            new_pin_map[i][j] = entropy
    if test_mode:
        print(new_pin_map)
    return new_pin_map

def cal_gray_variance(pin_map):
    path = f'{DATA_BOTTOM_CROP}/pinmap.jpg'
    img = cv2.imread(path, 0)
    new_pin_map = np.zeros((pin_map.shape[0], pin_map.shape[1]))
    for t in range(len(pin_map)):
        for s in range(len(pin_map[t])):
            img1 = img[int(pin_map[t][s][1]): int(pin_map[t][s][3]), int(pin_map[t][s][0]): int(pin_map[t][s][2])]
            height, width = img1.shape
            size = img1.size

            p = [0] * 256

            for i in range(height):
                for j in range(width):
                    p[img1[i][j]] += 1

            m = 0
            for i in range(256):
                p[i] /= size
                m += i * p[i]

            var_value = 0
            for i in range(256):
                var_value += (i - m) * (i - m) * p[i]
            new_pin_map[t][s] = var_value

    return new_pin_map
def output_pin(pin_map):
    print("每个网格中的信息熵计算值:")
    print(pin_map.astype(int))
    for i in range(len(pin_map)):
        for j in range(len(pin_map[i])):
            if pin_map[i][j] > 2:
                pin_map[i][j] = 1
            else:
                pin_map[i][j] = 0
    print("行:", pin_map.shape[0], "列:", pin_map.shape[1])

    # print(pin_map.astype(int))
    return pin_map


def cut_pinmap(pin_map, color, test_mode):
    while np.array_equal(pin_map[0], np.zeros(pin_map.shape[1]).astype(int)):
        pin_map = np.delete(pin_map, 0, 0)
        color = np.delete(color, 0, 0)
    while np.array_equal(pin_map[len(pin_map) - 1], np.zeros(pin_map.shape[1]).astype(int)):
        pin_map = np.delete(pin_map, [len(pin_map) - 1], 0)
        color = np.delete(color, [len(color) - 1], 0)
    while np.array_equal(pin_map[:, 0], np.zeros(pin_map.shape[0]).astype(int)):
        pin_map = np.delete(pin_map, [0], 1)
        color = np.delete(color, [0], 1)
    while np.array_equal(pin_map[:, -1], np.zeros(pin_map.shape[0]).astype(int)):
        pin_map = np.delete(pin_map, -1, 1)
        color = np.delete(color, -1, 1)
    if test_mode:
        print(pin_map.astype(int))
    return pin_map, color

def judge_singular_point(arr, color_map, test_mode):
    '''

    :param pin_map:
    :param color_map:
    :param test_mode:
    :return:
    '''
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            if arr[i, j] == 0:
                neighbors = []

                # 上下左右四个方向，边界外的默认不计入
                if i > 0:
                    neighbors.append(arr[i - 1, j])  # 上
                if i < rows - 1:
                    neighbors.append(arr[i + 1, j])  # 下
                if j > 0:
                    neighbors.append(arr[i, j - 1])  # 左
                if j < cols - 1:
                    neighbors.append(arr[i, j + 1])  # 右

                ones_count = sum(1 for val in neighbors if val == 1)

                # 判断是否符合奇异点标准
                if len(neighbors) == 4 and ones_count >= 3:  # 中间区域
                    color_map[i][j] = 3
                elif len(neighbors) == 3 and ones_count == 3:  # 边缘非角落
                    color_map[i][j] = 3
                elif len(neighbors) == 2 and ones_count == 2:  # 四个角落
                    color_map[i][j] = 3
    # rows, cols = arr.shape
    #
    #
    # for i in range(rows):
    #     for j in range(cols):
    #         if arr[i, j] == 0:
    #             count = 0
    #             # 上
    #             if i > 0 and arr[i - 1, j] == 1:
    #                 count += 1
    #             else:
    #                 count += 0
    #             # 下
    #             if i < rows - 1 and arr[i + 1, j] == 1:
    #                 count += 1
    #             else:
    #                 count += 0
    #             # 左
    #             if j > 0 and arr[i, j - 1] == 1:
    #                 count += 1
    #             else:
    #                 count += 0
    #             # 右
    #             if j < cols - 1 and arr[i, j + 1] == 1:
    #                 count += 1
    #             else:
    #                 count += 0
    #
    #             # 判断是否至少有三面是 1
    #             if count >= 3:
    #                 color_map[i][j] = 3
    return arr, color_map

# def output_color(color_map):
#
#     for i in range(len(color_map)):
#         for j in range(len(color_map[i])):
#             if color_map[i][j] == 0:
#                 color_map[i][j] =
#             elif color_map[i][j] == 1:
#                 color_map[i][j] = 2
#             elif color_map[i][j] == 2:
#                 color_map[i][j] = 3
#             elif color_map

def find_right_img():
    # 检测文件夹下有几张图片
    path =BGA_BOTTOM
    filelist = os.listdir(path)

    if len(filelist) == 0:
        # 从data文件夹把bottom.jpg复制到bga_bottom文件夹下
        print("在三视图中没有找到外框")
        shutil.copyfile(f'{DATA}/bottom.jpg', f'{BGA_BOTTOM}/bottom.jpg')

    filelist = os.listdir(path)
    if len(filelist) == 1:
        # 将文件夹bga_bottom下唯一一张图片改名为bottom.jpg
        os.rename(path + '/' + filelist[0], f'{BGA_BOTTOM}/bottom.jpg')
        get_pinmap()

    if len(filelist) == 2:
        print("在三视图中找到两个外框")
        img_path1 = path + '/' + filelist[0]
        pin1 = onnx_output_pairs_data_pin_5(img_path1)
        img_path2 = path + '/' + filelist[1]
        pin2 = onnx_output_pairs_data_pin_5(img_path2)
        if len(pin1) > len(pin2):
            pin = pin1
            os.remove(img_path2)
            # 改图片名为pinmap.jpg
            shutil.copyfile(img_path1, f'{DATA_BOTTOM_CROP}/pinmap.jpg')

        else:
            pin = pin2
            os.remove(img_path1)
            shutil.copyfile(img_path2, f'{DATA_BOTTOM_CROP}/pinmap.jpg')
            # os.rename(img_path2, r'data_bottom_crop/pinmap.jpg')

        # pin中找x_min和y_min和x_max和y_max

        x_min = 9999
        y_min = 9999
        x_max = 0
        y_max = 0
        for i in range(len(pin)):
            if pin[i][0] < x_min:
                x_min = pin[i][0]
            if pin[i][1] < y_min:
                y_min = pin[i][1]
            if pin[i][2] > x_max:
                x_max = pin[i][2]
            if pin[i][3] > y_max:
                y_max = pin[i][3]
        # 判断x_min和y_min和x_max和y_max所占的面积是否占比超过0.4
        img = cv2.imread(f'{DATA_BOTTOM_CROP}/pinmap.jpg')
        if (x_max - x_min) * (y_max - y_min) / img.shape[0] * img.shape[1] > 0.4:
            crop_img_save(f'{DATA_BOTTOM_CROP}/pinmap.jpg', f'{DATA_BOTTOM_CROP}/pinmap.jpg', max(int(x_min) - 1, 0),
                          max(int(y_min) - 1, 0), min(int(x_max) + 1, img.shape[1]),
                          min(int(y_max) + 1, img.shape[0]))
            print("经过yolox辅助得到pinmap")


def yolox_find_waikuang(location):
    '''
    在bottom图上拿yolox检测外框，再用yolox检测pin，按照pin的四个坐标的极限位置裁剪下pinmap
    '''
    path = f'{DATA}/bottom.jpg'
    # img = cv2.imread(path)
    # location = onnx_output_waikuang(path)
    print("location", location)
    if location.size == 0:
        key = True
        print("***/执行opencv检测外框+yolox检测pin/***")
    else:
        key = False
        print("***/执行yolox检测外框+yolox检测pin/***")
        print("location", location)
        # 从img中按照location截图
        # 将列表location里面的元素全化为整数
        location = [int(location[0][0]), int(location[0][1]), int(location[0][2]), int(location[0][3])]
        print("location", location)

        crop_img_save(path, f'{DATA_BOTTOM_CROP}/pinmap.jpg', location[0], location[1], location[2], location[3])
        # # 用yolox检测pin，按照pin的四个坐标的极限位置裁剪下pinmap
        # img_path1 = f'{DATA_BOTTOM_CROP}/pinmap.jpg'
        # pin = onnx_output_pairs_data_pin_5(img_path1)
        #
        # # pin中找x_min和y_min和x_max和y_max
        #
        # x_min = 9999
        # y_min = 9999
        # x_max = 0
        # y_max = 0
        # for i in range(len(pin)):
        #     if pin[i][0] < x_min:
        #         x_min = pin[i][0]
        #     if pin[i][1] < y_min:
        #         y_min = pin[i][1]
        #     if pin[i][2] > x_max:
        #         x_max = pin[i][2]
        #     if pin[i][3] > y_max:
        #         y_max = pin[i][3]
        # # 判断x_min和y_min和x_max和y_max所占的面积是否占比超过0.4
        # img = cv2.imread(f'{DATA_BOTTOM_CROP}/pinmap.jpg')
        # if (x_max - x_min) * (y_max - y_min) / img.shape[0] * img.shape[1] > 0.4:
        #     crop_img_save(f'{DATA_BOTTOM_CROP}/pinmap.jpg', f'{DATA_BOTTOM_CROP}/pinmap.jpg', max(int(x_min) - 1, 0),
        #                   max(int(y_min) - 1, 0), min(int(x_max) + 1, img.shape[1]),
        #                   min(int(y_max) + 1, img.shape[0]))
        #     print("经过yolox辅助得到pinmap")

    return key
def average_width_of_positive_intervals(a):
    intervals = []
    start = -1

    for i in range(len(a)):
        if a[i] > 0 and start == -1:
            start = i
        elif a[i] == 0 and start != -1:
            intervals.append(i - start)
            start = -1

    # 如果最后一个区间是以大于0的数结束的，需要额外处理
    if start != -1:
        intervals.append(len(a) - start)
    # print("intervals", intervals)
    if not intervals:
        return 0

    return sum(intervals) / len(intervals)

def find_start_p(a, ave_width, p):
    for i in range(len(a) - 1):
        if a[i] == 0 and a[i + 1] > 0:
            start_p = i + 1
            width = len(a) - 1 - i
            for j in range(i + 1, len(a)):

                if a[j] == 0:
                    width = j-i - 1
                    break
            if width < ave_width * p:
                i = j
                start_p = j
            else:
                break
    return start_p
def crop_pin_map_x(a,w, h,  test_mode):
    '''
    根据宽度判断最外围是否为PIN，并裁剪图片使最外部PIN紧贴图片边缘
    :param a:
    :return:
    '''

    start_p  = 0
    end_p = len(a) - 1
    p = 0.5
    ave_width = average_width_of_positive_intervals(a)
    if test_mode:
        print("ave_width", ave_width)
    # 抹平a中宽度小于平均宽度*p的区间和大于平均宽度/p的区间的宽度
    for i in range(1, len(a)):
        if a[i] > 0 and a[i - 1] == 0:
            a_start = i
            for j in range(i + 1, len(a)):
                if a[j] == 0:
                    a_end = j
                    break
            if abs(a_end - a_start) < ave_width * p or abs(a_end - a_start) > ave_width / p:
                for k in range(a_start, a_end):
                    a[k] = 0
    # 最左侧区间宽度小于平均宽度*p，则起始点下一个区间起点
    start_p = find_start_p(a, ave_width, p)
    a_reversed = a[::-1]
    end_p = find_start_p(a_reversed, ave_width, p)
    end_p = len(a) - end_p - 1
    # 根据开始和结束点裁剪图片和a
    crop_img_save(f'{DATA_BOTTOM_CROP}/pinmap.jpg', f'{DATA_BOTTOM_CROP}/pinmap.jpg', max(start_p - 1, 0), 0, min(end_p + 2, w - 1), h - 1)

    a = a[max(start_p - 1, 0):min(end_p + 2, w - 1)]
    if test_mode:
        print("a", a)
        print("len(a)", len(a))
    return a, ave_width

def crop_pin_map_y(a,w,h, test_mode):
    '''
    根据宽度判断最外围是否为PIN，并裁剪图片使最外部PIN紧贴图片边缘
    :param a:
    :return:
    '''

    start_p  = 0
    end_p = len(a) - 1
    p = 0.5
    ave_width = average_width_of_positive_intervals(a)
    # 抹平a中宽度小于平均宽度*p的区间
    for i in range(1, len(a)):
        if a[i] > 0 and a[i - 1] == 0:
            a_start = i
            for j in range(i + 1, len(a)):
                if a[j] == 0:
                    a_end = j
                    break
            if abs(a_end - a_start) < ave_width * p:
                for k in range(a_start, a_end):
                    a[k] = 0
    # 最左侧区间宽度小于平均宽度*p，则起始点下一个区间起点
    start_p = find_start_p(a, ave_width, p)
    a_reversed = a[::-1]
    end_p = find_start_p(a_reversed, ave_width, p)
    end_p = len(a) - end_p - 1
    # 根据开始和结束点裁剪图片和a
    crop_img_save(f'{DATA_BOTTOM_CROP}/pinmap.jpg', f'{DATA_BOTTOM_CROP}/pinmap.jpg', 0, max(0,start_p -1), w - 1, min(end_p + 2, h - 1))

    a = a[max(start_p - 1,0):min(end_p+2, h - 1)]
    if test_mode:
        print("b", a)
    return a, ave_width


def find_pin_core():
    test_mode = False
    # 读取图片
    image1 = cv2.imread(f'{DATA_BOTTOM_CROP}/pinmap.jpg')
    # 灰度图像
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    (h, w) = binary.shape  # 返回高和宽
    # 垂直投影
    vproject = binary.copy()
    a = [0 for x in range(0, w)]
    # 记录每一列的波峰
    for j in range(0, w):  # 遍历一列
        for i in range(0, h):  # 遍历一行
            if vproject[i, j] == 0:  # 如果改点为黑点
                a[j] += 1  # 该列的计数器加1计数
                vproject[i, j] = 255  # 记录完后将其变为白色

    vproject_copy = vproject.copy()
    for j in range(0, w):  # 遍历每一列
        for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
            try:
                vproject_copy[i, j] = 0  # 涂黑
            except:
                pass
    if test_mode:
        cv2.namedWindow('vproject', cv2.WINDOW_NORMAL)
        cv2.imshow("vproject", vproject_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 滤波
    # print("垂直投影", a)
    a = filter_a_b(a)
    for j in range(0, w):  # 遍历每一列
        for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
            try:
                vproject[i, j] = 0  # 涂黑
            except:
                pass
    if test_mode:
        cv2.namedWindow('vproject', cv2.WINDOW_NORMAL)
        cv2.imshow("vproject", vproject)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 寻找PIN map准确位置并沿着边缘裁剪为完整PINmap
    a, a_ave_width = crop_pin_map_x(a, w, h,test_mode)
    # print("滤波后垂直投影", a)
    # 找网格
    c = find_line(a, a_ave_width, test_mode)
    a = [i + j for i, j in zip(a, c)]

    # 读取图片
    image1 = cv2.imread(f'{DATA_BOTTOM_CROP}/pinmap.jpg')
    # 灰度图像
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    (h, w) = binary.shape  # 返回高和宽
    # # 垂直投影
    # vproject = binary.copy()
    # # a = [0 for x in range(0, w)]
    # # 记录每一列的波峰
    # for j in range(0, w):  # 遍历一列
    #     for i in range(0, h):  # 遍历一行
    #         if vproject[i, j] == 0:  # 如果改点为黑点
    #             a[j] += 1  # 该列的计数器加1计数
    #             vproject[i, j] = 255  # 记录完后将其变为白色

    for j in range(0, w):  # 遍历每一列
        for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
            try:
                vproject[i, j] = 0  # 涂黑
            except:
                pass
    # cv2.putText(vproject, "verticality", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 4)
    # 水平投影
    hproject = binary.copy()
    b = [0 for x in range(0, h)]
    for j in range(0, h):
        for i in range(0, w):
            if hproject[j, i] == 0:
                b[j] += 1
                hproject[j, i] = 255
    # 滤波
    # print("水平投影", b)
    b = filter_a_b(b)
    # print("滤波后水平投影", b)
    b, b_ave_width = crop_pin_map_y(b, w, h, test_mode)
    # 找网格
    d = find_line(b, b_ave_width, test_mode)
    b = [i + j for i, j in zip(b, d)]

    # 读取图片
    image1 = cv2.imread(f'{DATA_BOTTOM_CROP}/pinmap.jpg')
    # 灰度图像
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    (h, w) = binary.shape  # 返回高和宽
    # # 垂直投影
    # vproject = binary.copy()
    # a = [0 for x in range(0, w)]
    # # 记录每一列的波峰
    # for j in range(0, w):  # 遍历一列
    #     for i in range(0, h):  # 遍历一行
    #         if vproject[i, j] == 0:  # 如果改点为黑点
    #             a[j] += 1  # 该列的计数器加1计数
    #             vproject[i, j] = 255  # 记录完后将其变为白色

    for j in range(0, h):
        for i in range(0, b[j]):
            try:
                hproject[j, i] = 0
            except:
                pass
    # cv2.putText(hproject, "horizontal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 4)
    # 一次显示四张图，两行两列，第一张是data/bottom.jpg，第二张图是image1，第三张是vproject，，第四张是hproject
    if test_mode:
        cv2.imshow("image1", image1)
        cv2.namedWindow('vproject', cv2.WINDOW_NORMAL)
        cv2.imshow("vproject", vproject)
        cv2.namedWindow('hproject', cv2.WINDOW_NORMAL)
        cv2.imshow("hproject", hproject)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 存储网格
    pin_map = get_pin_grid(a, b, c, d, test_mode)
    # # 显示各个网格
    # show_pin_grid(pin_map, c, d)
    # # 计算各个网格中的信息熵
    pin_map_wangge = pin_map.copy()


    pin_map = cal_grid_shang(pin_map, test_mode)
    # 每张网格中的图片与pin1和pin2比较相似度
    # pin_map = compare_pin_2(pin_map, pin_map_wangge)
    #灰度方差
    # pin_map = cal_gray_variance(pin_map)
    # 将信息熵高的网格内容提取，计算两两之间的图片相似度，选取最多相似的网格类型作为模板，筛选所有信息熵高的网格，删除掉相似度低于阈值的网格
    pin_map, color_map = compare_pin(pin_map, pin_map_wangge,  test_mode)
    # 根据熵输出每个网格是否存在pin
    # pin_map = output_pin(pin_map)
    # # 裁剪pinmap
    pin_map, color_map = cut_pinmap(pin_map, color_map, test_mode)
    # 判断是否存在奇异点
    pin_map, color_map = judge_singular_point(pin_map, color_map, test_mode)

    # output_color(color_map)
    output = np.vstack((pin_map, color_map))
    # print("out", output)
    return output

def compare_pin_2(pin_map, pin_map_wangge):
    path1 = f'{DATA_BOTTOM_CROP}/pinmap.jpg'
    img = cv2.imread(path1, 0)
    path2 = f'Result/pin1.jpg'
    path3 = f'Result/pin2.jpg'
    img_pin1 = cv2.imread(path2)
    img_pin2 = cv2.imread(path3)
    row, col = pin_map.shape
    pin_compare1 = np.zeros((row, col))
    pin_compare2 = np.zeros((row, col))
    for i in range(row):
        for j in range(col):

            img1 = img[int(pin_map_wangge[i][j][1]): int(pin_map_wangge[i][j][3]),
                   int(pin_map_wangge[i][j][0]): int(pin_map_wangge[i][j][2])]
            # 归一化img1为20*20分辨率
            img1 = cv2.resize(img1, (20, 20))
            # hash1 = dHash(img1)
            # hash2 = dHash(img2)
            # n = cmpHash(hash1, hash2)
            n = classify_hist_with_split(img1, img_pin1)
            pin_compare1[i][j] = np.round(n, 2)
            n = classify_hist_with_split(img1, img_pin2)
            pin_compare2[i][j] = np.round(n, 2)
    print("pin_compare1", pin_compare1)
    print("pin_compare2", pin_compare2)

def compare_pin(pin_map, pin_map_wangge, test_mode):
    '''
    将信息熵高的网格内容提取，计算两两之间的图片相似度，选取最多相似的网格类型作为模板，筛选所有信息熵高的网格，删除掉相似度低于阈值的网格
    '''
    # 0 = 空 1 = 非空但和模板不相似 2 = 和模板相似 3 = 奇异点
    color_array = np.zeros_like(pin_map)
    path = f'{DATA_BOTTOM_CROP}/pinmap.jpg'
    img = cv2.imread(path)
    # (1)信息熵最大与最小的差值*0.2+最小值作为界限筛选出信息熵
    try:
        min_entropy = np.min(pin_map)
        max_entropy = np.max(pin_map)
        # threshold = (max_entropy - min_entropy) * 0.2 + min_entropy
        threshold = 0.5
    except:
        print("pin_map为空")
        threshold = 0
    # (2)将这些信息熵的网格图片内容保存在数组中

    # (3)对比图片相似度
    row, col = pin_map.shape
    pin_compare_1 = np.empty((row, col, row * col))
    pin_compare_2 = np.empty((row, col))
    pin_compare = np.zeros((row, col))
    pin_map_compare = np.zeros((row, col))
    for i in range(int(row * 0.25)):
        for j in range(int(col * 0.25)):
            if pin_map[i][j] > threshold:
                color_array[i][j] = 1
                for k in range(int(row * 0.25)):
                    for l in range(int(col * 0.25)):
                        if pin_map[k][l] > threshold:
                            img1 = img[int(pin_map_wangge[i][j][1]): int(pin_map_wangge[i][j][3]),
                                   int(pin_map_wangge[i][j][0]): int(pin_map_wangge[i][j][2])]
                            img2 = img[int(pin_map_wangge[k][l][1]): int(pin_map_wangge[k][l][3]),
                                   int(pin_map_wangge[k][l][0]): int(pin_map_wangge[k][l][2])]

                            # hash1 = dHash(img1)
                            # hash2 = dHash(img2)
                            # n = cmpHash(hash1, hash2)
                            n = classify_hist_with_split(img1, img2)

                            pin_compare_1[i][j][k * col + l] = n
                            pin_compare[i][j] += n
                            if test_mode:
                                print(i, j, "与", k, l, "得到", n)
    if test_mode:
        print(pin_compare)
    # 找到pin_compare中不为0的最大值，取其行数和列数,将其作为存在PIN的比较的模板
    max_value = pin_compare[0][0]
    max_row = 0
    max_col = 0

    for i in range(len(pin_compare)):
        for j in range(len(pin_compare[i])):
            if pin_compare[i][j] > max_value:
                max_value = pin_compare[i][j]
                max_row = i
                max_col = j

    img1 = img[int(pin_map_wangge[max_row][max_col][1]): int(pin_map_wangge[max_row][max_col][3]),
           int(pin_map_wangge[max_row][max_col][0]): int(pin_map_wangge[max_row][max_col][2])]
    # 逐个比较是否和模板相似
    for i in range(row):
        for j in range(col):
            if pin_map[i][j] < threshold:
                pin_map_compare[i][j] = 0
                pin_compare_2[i][j] = 0
            else:

                img2 = img[int(pin_map_wangge[i][j][1]): int(pin_map_wangge[i][j][3]),
                       int(pin_map_wangge[i][j][0]): int(pin_map_wangge[i][j][2])]

                # hash1 = dHash(img1)
                # hash2 = dHash(img2)
                # n = cmpHash(hash1, hash2)
                n = classify_hist_with_split(img1, img2)

                pin_compare_2[i][j] = n
                if n > 0.7:
                    pin_map_compare[i][j] = 1
                    color_array[i][j] = 2
                else:
                    pin_map_compare[i][j] = 0
    if test_mode:
        print(pin_map_compare)
        print(pin_compare_2)
    return pin_map_compare, color_array

# 通过得到RGB每个通道的直方图来计算相似度
def classify_hist_with_split(image1, image2, size=(256, 256)):
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data

# 计算单通道的直方图的相似值
def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

def find_pin(bottom_border):
    # 重置行列数
    pin_num_x_y = np.array([0, 0])
    pin_num_x_y = pin_num_x_y.astype(int)
    path = f'{YOLOX_DATA}\pin_num.txt'
    np.savetxt(path, pin_num_x_y)

    empty_folder('Result/Package_extract/bga')
    os.makedirs('Result/Package_extract/bga')
    empty_folder(BGA_BOTTOM)
    os.makedirs(BGA_BOTTOM)
    # bga_bottom中如果存在两张图片，yolox检测，保留pin数量多的存为pinmap.jpg
    key = yolox_find_waikuang(bottom_border)
    if key:
        get_waikaung()
        find_right_img()
    # get_pinmap()
    output_list = find_pin_core()
    return output_list


# if __name__ == '__main__':
# image1 = cv2.imread(r'data_bottom_crop/pinmap.jpg')
# # 灰度图像
# gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# # 二值化
# ret, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
# (h, w) = binary.shape  # 返回高和宽
# # 垂直投影
# vproject = binary.copy()
# a = [0 for x in range(0, w)]
# # 记录每一列的波峰
# for j in range(0, w):  # 遍历一列
#     for i in range(0, h):  # 遍历一行
#         if vproject[i, j] == 0:  # 如果改点为黑点
#             a[j] += 1  # 该列的计数器加1计数
#             vproject[i, j] = 255  # 记录完后将其变为白色
# # 滤波
# # print("垂直投影", a)
# a = filter_a_b_test(a)
# # print("滤波后垂直投影", a)
# # 找网格
# c = find_line_test(a)
# a = [i + j for i, j in zip(a, c)]
#
# for j in range(0, w):  # 遍历每一列
#     for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
#         try:
#             vproject[i, j] = 0  # 涂黑
#         except:
#             pass
# cv2.putText(vproject, "verticality", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 4)
# # 水平投影
# hproject = binary.copy()
# b = [0 for x in range(0, h)]
# for j in range(0, h):
#     for i in range(0, w):
#         if hproject[j, i] == 0:
#             b[j] += 1
#             hproject[j, i] = 255
# # 滤波
# # print("水平投影", b)
# b = filter_a_b_test(b)
# # print("滤波后水平投影", b)
# # 找网格
# d = find_line_test(b)
# b = [i + j for i, j in zip(b, d)]
#
# for j in range(0, h):
#     for i in range(0, b[j]):
#         try:
#             hproject[j, i] = 0
#         except:
#             pass
# cv2.putText(hproject, "horizontal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 4)
#
# # cv2.imshow("image1", image1)
# # cv2.namedWindow('vproject', cv2.WINDOW_NORMAL)
# # cv2.imshow("vproject", vproject)
# # cv2.namedWindow('hproject', cv2.WINDOW_NORMAL)
# # cv2.imshow("hproject", hproject)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # 存储网格
# pin_map = get_pin_grid_test(a, b, c, d)
# # 显示各个网格
# show_pin_grid(pin_map, c, d)
# # 计算各个网格中的混乱熵
# pin_map = cal_grid_shang(pin_map)
# # 根据熵输出每个网格是否存在pin
# pin_map = output_pin(pin_map)
# # 裁剪pinmap
# pin_map = cut_pinmap(pin_map)
if __name__ == '__main__':
    find_pin_core()