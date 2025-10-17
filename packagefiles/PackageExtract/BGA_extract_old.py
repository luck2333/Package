import cv2
import glob
import fitz
from PIL import Image
import shutil
import logging
import os
import math
import numpy as np
import pandas as pd
import threading
from packagefiles.PackageExtract.BGA_cal_pin import find_pin
import queue
import json
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
# from output_pin_num4 import begain_output_pin_num_pin_map
from packagefiles.PackageExtract.yolox_onnx_py.onnx_output_pin_num4 import begain_output_pin_num_pin_map
from packagefiles.PackageExtract.yolox_onnx_py.onnx_output_bottom_pin_location import begain_output_bottom_pin_location

from packagefiles.PackageExtract.get_pairs_data_present5 import begain_get_pairs_data_present2
from packagefiles.PackageExtract.get_pairs_data_present5 import yinXian_begain_get_data_present
from packagefiles.PackageExtract.get_pairs_data_present5 import get_pinmap_table

# from mmpretrain_main.inferen import classify_img
# from img_divide import bga_divide

logging.disable(logging.INFO)
logging.disable(logging.WARNING)


def detect_img_top_bottom_side(path):
    img_list = os.listdir(path)
    if 'top.jpg' not in img_list:
        img = np.ones((350, 500, 3), dtype=np.uint8)
        img = 255 * img
        cv2.imwrite(path + '/top.jpg', img)
        print(path, "中未检测到top视图")
    if 'bottom.jpg' not in img_list:
        img = np.ones((350, 500, 3), dtype=np.uint8)
        img = 255 * img
        cv2.imwrite(path + '/bottom.jpg', img)
        print(path, "中未检测到bottom视图")
    if 'side.jpg' not in img_list:
        img = np.ones((350, 500, 3), dtype=np.uint8)
        img = 255 * img
        cv2.imwrite(path + '/side.jpg', img)
        print(path, "中未检测到side视图")


def empty_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except FileNotFoundError:
        print(f"文件夹 {folder_path} 不存在！无须删除文件夹")


# 指定pdf页面转成图片
def pdf2img(pdfPath, pageNumber, imgfilePath, save_name):
    scale = 3  # 放大倍率
    images_np = []
    with fitz.open(pdfPath) as pdfDoc:
        # for pageNumber in pageNumbers:
        page = pdfDoc.load_page(pageNumber - 1)
        mat = fitz.Matrix(scale, scale).prerotate(0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(imgfilePath + '/' + save_name)  # 将图片写入指定的文件夹内
    return scale, images_np


def manual_get_boxes(folder_path, save_path, save_name):
    # 定义文件夹路径和保存路径
    global rect_list
    # 获取文件夹内所有图片文件
    file_list = glob.glob(os.path.join(folder_path, '*.jpg'))
    # 循环处理每张图片
    for i, file_path in enumerate(file_list):
        # 读取图片
        img = cv2.imread(file_path)
        # 获取图片尺寸
        height, width, _ = img.shape
        # 计算缩放比例
        # scale = min(1.0, 1024 / max(height, width))
        scale = 1
        # 缩放图片
        img_resized = cv2.resize(img, None, fx=scale, fy=scale)
        # 创建窗口并显示图片
        cv2.namedWindow('image', 0)
        # cv2.namedWindow('picture', 0)
        cv2.imshow('image', img_resized)
        # 初始化框选区域列表
        rect_list = []
        # 循环框选区域
        while True:
            # 等待用户框选区域
            rect = cv2.selectROI('image', img_resized, False)
            # 计算缩放后的框选区域
            rect_resized = [int(x / scale) for x in rect]
            # 如果没有框选区域，则退出循环
            if rect == (0, 0, 0, 0):
                break
            # 截取选中区域并保存
            crop_img = img[rect_resized[1]:rect_resized[1] + rect_resized[3],
                       rect_resized[0]:rect_resized[0] + rect_resized[2]]
            rect_list.append(rect)
            # print(rect_list)
            cv2.imwrite(os.path.join(save_path, save_name), crop_img)

    selectRec = (rect_list[0][0], rect_list[0][1], rect_list[0][2] + rect_list[0][0], rect_list[0][3] + rect_list[0][1])
    # crop_img_save()
    cv2.destroyAllWindows()  # 关闭弹框

    return selectRec


def crop_img_save(path_img, path_crop, x_min, y_min, x_max, y_max):
    # import cv2
    img = cv2.imread(path_img)
    cropped = img[y_min:y_max, x_min:x_max]  # 裁剪坐标为[y0:y1, x0:x1]必须为整数
    cv2.imwrite(path_crop, cropped)
    print("保存图", path_crop)


def ResizeImage(filein, fileout, scale=1):
    """
    改变图片大小
    :param filein: 输入图片
    :param fileout: 输出图片
    :param width: 输出图片宽度
    :param height: 输出图片宽度
    :param type: 输出图片类型（png, gif, jpeg...）
    :return:
    """
    img = Image.open(filein)
    if img.size[0] < 1080 and img.size[1] < 1080:  # 限制图片大小避免过大

        width = int(img.size[0] * scale)
        height = int(img.size[1] * scale)
        type = img.format
        out = img.resize((width, height),
                         Image.LANCZOS)
        # 第二个参数：
        # Image.NEAREST ：低质量
        # Image.BILINEAR：双线性
        # Image.BICUBIC ：三次样条插值
        # Image.ANTIALIAS：高质量
        out.save(fileout, type)


def set_Image_size(filein, fileout):
    """
    改变图片大小
    :param filein: 输入图片
    :param fileout: 输出图片
    :param width: 输出图片宽度
    :param height: 输出图片宽度
    :param type: 输出图片类型（png, gif, jpeg...）
    :return:
    """
    max_length = 1000
    img = Image.open(filein)
    if img.size[0] > img.size[1]:  # 限制图片大小避免过大
        width = max_length
        height = int(img.size[1] * max_length / img.size[0])
    else:
        height = max_length
        width = int(img.size[0] * max_length / img.size[1])
    type = img.format
    out = img.resize((width, height), Image.LANCZOS)
    out = hist(out, show_img_key=0)
    # 第二个参数：
    # Image.NEAREST ：低质量
    # Image.BILINEAR：双线性
    # Image.BICUBIC ：三次样条插值
    # Image.ANTIALIAS：高质量
    out = Image.fromarray(np.uint8(out))
    out.save(fileout, type)


# def set_Image_size(filein, fileout):
#     """
#     改变图片大小
#     :param filein: 输入图片
#     :param fileout: 输出图片
#     :param width: 输出图片宽度
#     :param height: 输出图片宽度
#     :param type: 输出图片类型（png, gif, jpeg...）
#     :return:
#     """
#     max_length = 1000
#     img = Image.open(filein)
#     if img.size[0] > img.size[1]:  # 限制图片大小避免过大
#         width = max_length
#         height = int(img.size[1] * max_length / img.size[0])
#     else:
#         height = max_length
#         width = int(img.size[0] * max_length / img.size[1])
#     img = img.resize((width, height), Image.LANCZOS)
#     img = img.convert('L')
#     cleaned_page_array = cv2.adaptiveThreshold(np.array(img),
#                                                255,
#                                                # cv2.ADAPTIVE_THRESH_MEAN_C, #基于邻域均值的自适应阈值。
#                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 基于邻域加权平均的自适应阈值。
#                                                cv2.THRESH_BINARY,
#                                                19,
#                                                15)
#     new_image = np.array(Image.fromarray(cleaned_page_array))
#     cv2.imwrite(fileout, new_image)


def hist(img, show_img_key):
    # 求出img 的最大最小值
    Maximg = np.max(img)
    Minimg = np.min(img)
    # 输出最小灰度级和最大灰度级
    Omin, Omax = 0, 255
    # 求 a, b
    a = float(Omax - Omin) / (Maximg - Minimg)
    b = Omin - a * Minimg
    # 线性变换
    O = a * img + b
    O = O.astype(np.uint8)
    if show_img_key == 1:
        cv2.imshow('enhance-0', O)
        # cv2.imwrite('hist.png', O, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return O


def hist_auto(img):
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    # 使用全局直方图均衡化
    equa = cv2.equalizeHist(img)
    # 分别显示原图，CLAHE，HE
    # cv.imshow("img", img)
    # cv2.imshow("dst", dst)
    cv2.imwrite('hist_auto.png', dst, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def calcGrayHist(I):
    # 计算灰度直方图
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist


def equalHist(img):
    # import math
    # 灰度图像矩阵的高、宽
    h, w = img.shape
    # 第一步：计算灰度直方图
    grayHist = calcGrayHist(img)
    # 第二步：计算累加灰度直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = math.floor(q)
        else:
            outPut_q[p] = 0
    # 第四步：得到直方图均衡化后的图像
    equalHistImage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalHistImage[i][j] = outPut_q[img[i][j]]
    return equalHistImage


def linear(img):
    # img = cv2.imread(source, 0)
    # 使用自己写的函数实现
    equa = equalHist(img)
    cv2.imshow("equa", equa)
    cv2.imwrite('temp.png', equa, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.waitKey()


def dirlist(path):  # 循环遍历文件夹下的文件并输出list存储所有文件的绝对地址
    allfile = []
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile


def present_pin_map(pin_map):
    # print(pin_map)
    print("################输出bottom视图###############")
    print("pin存在显示'o',不存在以位置信息代替")
    print('     ', end='')
    for i in range(len(pin_map[0])):
        if i <= 8:
            print(i + 1, end='    ')
        if i > 8:
            print(i + 1, end='   ')
    print()
    letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'U', 'V', 'W',
              'Y']
    for i in range(len(pin_map)):
        if (i + 1) <= 20:
            print(letter[i], end='    ')
        if 20 <= i:
            print(letter[int(((i + 1) // 20) - 1)] + letter[int(((i + 1) % 20) - 1)], end='   ')
        for j in range(len(pin_map[i])):
            if pin_map[i][j] == 1:
                print("o", end='    ')
            if pin_map[i][j] == 0 and (i + 1) <= 20:
                if (j + 1) > 9:
                    print(letter[int(i)], j + 1, end='  ', sep='')
                if (j + 1) < 10:
                    print(letter[int(i)], j + 1, end='   ', sep='')
            if pin_map[i][j] == 0 and i > 19:
                if (j + 1) > 9:
                    print(letter[int(((i + 1) // 20) - 1)] + letter[int(((i + 1) % 20) - 1)], j + 1, end=' ',
                          sep='')
                if (j + 1) < 10:
                    print(letter[int(((i + 1) // 20) - 1)] + letter[int(((i + 1) % 20) - 1)], j + 1, end='  ',
                          sep='')
        print()


def do_the_same():  # 判断是否继续检测
    wh_key = True
    while wh_key:
        while_key = input("是否继续检测：y/n:")
        if while_key == 'y' or while_key == 'Y':
            while_num = True
            wh_key = False
        elif while_key == 'n' or while_key == 'N':
            while_num = False
            wh_key = False
        else:
            print("未输入正确，请重新输入：y/n:")
            wh_key = True
    return while_num


# -*- coding: utf-8 -*-


def pd_toExcel(data, fileName):  # pandas库储存数据到excel
    parameter = []
    max = []
    type = []
    min = []
    for i in range(len(data)):
        parameter.append(data[i]["参数"])
        max.append(data[i]["Max"])
        type.append(data[i]["Type"])
        min.append(data[i]["Min"])

    dfData = {  # 用字典设置DataFrame所需数据
        '参数': parameter,
        'Max': max,
        'Type': type,
        'Min': min
    }
    df = pd.DataFrame(dfData)  # 创建DataFrame
    df.to_excel(fileName, index=False)  # 存表，去除原始索引列（0,1,2...）


def make_data(body_x, body_y, pin_x_num, pin_y_num, pitch_x, pitch_y, high, pin_diameter, standoff, pin_map):
    testData = []
    if len(body_x) != 0:
        testData.append({"参数": "实体长(D)", "Max": body_x[0], "Type": body_x[1], "Min": body_x[2]})
    else:
        testData.append({"参数": "实体长(D)", "Max": 0, "Type": 0, "Min": 0})
    if len(body_y) != 0:
        testData.append({"参数": "实体宽(E)", "Max": body_y[0], "Type": body_y[1], "Min": body_y[2]})
    else:
        testData.append({"参数": "实体宽(E)", "Max": 0, "Type": 0, "Min": 0})
    if len(high) != 0:
        testData.append({"参数": "实体高(A)", "Max": high[0], "Type": high[1], "Min": high[2]})
    else:
        testData.append({"参数": "实体高(A)", "Max": 0, "Type": 0, "Min": 0})
    if len(standoff) != 0:
        for i in range(len(standoff)):
            testData.append(
                {"参数": "支撑高(A1)", "Max": standoff[i][0], "Type": standoff[i][1], "Min": standoff[i][2]})
    else:
        testData.append({"参数": "支撑高(A1)", "Max": 0, "Type": 0, "Min": 0})
    if len(pin_diameter) != 0:
        for i in range(len(pin_diameter)):
            testData.append(
                {"参数": "球直径(b)", "Max": pin_diameter[i][0], "Type": pin_diameter[i][1], "Min": pin_diameter[i][2]})
    else:
        testData.append({"参数": "球直径(b)", "Max": 0, "Type": 0, "Min": 0})
    testData.append({"参数": "行数(n_x)", "Max": "-", "Type": pin_x_num, "Min": "-"})
    testData.append({"参数": "列数(n_x)", "Max": "-", "Type": pin_y_num, "Min": "-"})
    if len(pitch_x) != 0:
        testData.append({"参数": "行Pitch(e)", "Max": "-", "Type": pitch_x, "Min": "-"})
    else:
        testData.append({"参数": "行Pitch(e)", "Max": "-", "Type": 0, "Min": "-"})
    if len(pitch_y) != 0:
        testData.append({"参数": "列Pitch(e1)", "Max": "-", "Type": pitch_y, "Min": "-"})
    else:
        testData.append({"参数": "列Pitch(e1)", "Max": "-", "Type": 0, "Min": "-"})
    if (pin_map == 1).all():
        testData.append({"参数": "缺PIN否", "Max": "否", "Type": "-", "Min": "-"})
    else:

        letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'U', 'V', 'W',
                  'Y']
        lost_pin = list([])

        for i in range(len(pin_map)):
            for j in range(len(pin_map[i])):
                if pin_map[i][j] == 0 and (i + 1) <= 20:
                    string = letter[int(i)] + str(j + 1)
                    string = list(string)
                    string = ''.join(string)
                    lost_pin.append(string)
                if pin_map[i][j] == 0 and i > 19:
                    string = letter[int(((i + 1) // 20) - 1)] + letter[int(((i + 1) % 20) - 1)] + str(j + 1)
                    string = list(string)
                    string = ''.join(string)
                    lost_pin.append(string)
        testData.append({"参数": "缺PIN否", "Max": lost_pin, "Type": "-", "Min": "-"})

    return testData


def output_excel(body_x, body_y, pin_x_num, pin_y_num, pitch_x, pitch_y, high, pin_diameter, standoff, pin_map,
                 pdf_name, page):
    test_data = make_data(body_x, body_y, pin_x_num, pin_y_num, pitch_x, pitch_y, high, pin_diameter, standoff, pin_map)
    fileName = f'{pdf_name}第{page}页检测结果.xlsx'
    pd_toExcel(test_data, 'output/' + fileName)


def resize_bga(bga_path):
    img = Image.open(bga_path)
    if 700 < img.size[0] < 2600 and 700 < img.size[1] < 2600:
        print("BGA图片大小合适")
    else:

        width = int(1500)
        height = int(img.size[1] * width / img.size[0])
        type = img.format
        out = img.resize((width, height), Image.ANTIALIAS)
        # 第二个参数：
        # Image.NEAREST ：低质量
        # Image.BILINEAR：双线性
        # Image.BICUBIC ：三次样条插值
        # Image.ANTIALIAS：高质量
        out.save(bga_path, type)


def get_pinmap():
    # 由用户决定是否使用人工框选pinmap
    # 创建文件夹

    # shutil.copy(f'data/bottom.jpg', f'bga_bottom/bottom.jpg')
    folder_path = "bga_bottom"
    save_path = f'data_bottom_crop'
    save_name = f'pinmap.jpg'

    # wh_key1 = True
    # while wh_key1:
    #     auto_key = input("是否手动框选pin图:y/n:")
    #     if auto_key == 'y' or auto_key == 'Y':
    #         auto_bool = False
    #         wh_key1 = False
    #     elif auto_key == 'n' or auto_key == 'N':
    #         auto_bool = True
    #         wh_key1 = False
    #     else:
    #         print("未输入正确，请重新输入：y/n:")
    #         wh_key1 = True
    # if auto_bool == True:

    # from output_bottom_pin_location import begain_output_bottom_pin_location
    pin_map_limation_1 = begain_output_bottom_pin_location()
    path_img = 'bga_bottom/bottom.jpg'
    # filter_black_point(path_img, path_img)
    path_crop = 'data_bottom_crop/pinmap.jpg'
    x_min = int(pin_map_limation_1[0][0])
    y_min = int(pin_map_limation_1[0][1])
    x_max = int(pin_map_limation_1[0][2])
    y_max = int(pin_map_limation_1[0][3])
    crop_img_save(path_img, path_crop, x_min, y_min, x_max, y_max)

    # print('pin_map_limation_1', pin_map_limation_1)
    # else:
    #
    #     print("框选pinmap")
    #     pin_map_limation = np.array([[1, 1, 1, 1]])
    #     pin_map_limation = manual_get_boxes(folder_path, save_path, save_name)
    #
    #     pin_map_limation = np.asarray(pin_map_limation)
    #     pin_map_limation_1 = np.zeros((0, 4))
    #     pin_map_limation_1 = np.r_[pin_map_limation_1, [pin_map_limation]]
    #     print('pin_map_limation_1', pin_map_limation_1)

    np.savetxt('yolox_data/pin_map_limation.txt', pin_map_limation_1, delimiter=',')
    # 清空文件夹


def Is_Loss_Pin(pin_map, pin_1_location, color):
    '''
    # pin_1_location = [X, Y],X = 0:横向用数字标记序号，纵向用字母标记序号；X= 1，横向用字母标记序号，纵向用数字标记序号
    # pin_1_location = [X, Y],Y = 0 = 左上角,1= 右上角，2 = 右下角，3 = 左下角
    '''
    # pin_map = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'U', 'V', 'W',
              'Y']
    lost_pin = list([])
    lost_color = list([])
    num_y = pin_map.shape[0]  # 行数
    num_x = pin_map.shape[1]  # 列数
    # print(pin_map)
    if pin_1_location[0] == 0 or pin_1_location[0] == -1:
        if pin_1_location[1] == 0 or pin_1_location[1] == -1:
            for i in range(len(pin_map)):
                for j in range(len(pin_map[i])):
                    if pin_map[i][j] == 0 and (i + 1) <= 20:
                        string = letter[int(i)] + str(j + 1)
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
                    if pin_map[i][j] == 0 and i > 19:
                        string = letter[int(((i + 1) // 20) - 1)] + letter[int(((i + 1) % 20) - 1)] + str(j + 1)
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
        elif pin_1_location[1] == 1:
            for i in range(len(pin_map)):
                for j in range(len(pin_map[i])):
                    if pin_map[i][j] == 0 and (i + 1) <= 20:
                        string = letter[int(i)] + str(num_x - j)
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
                    if pin_map[i][j] == 0 and i > 19:
                        string = letter[int(((i + 1) // 20) - 1)] + letter[int(((i + 1) % 20) - 1)] + str(num_x - j)
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
        elif pin_1_location[1] == 2:
            for i in range(len(pin_map)):
                for j in range(len(pin_map[i])):
                    if pin_map[i][j] == 0 and (i + 1) <= 20:
                        string = letter[int(num_y - i - 1)] + str(num_x - j)
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
                    if pin_map[i][j] == 0 and i > 19:
                        string = letter[int(((num_y - i) // 20) - 1)] + letter[int(((i + 1) % 20) - 1)] + str(num_x - j)
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
        elif pin_1_location[1] == 3:
            for i in range(len(pin_map)):
                for j in range(len(pin_map[i])):
                    if pin_map[i][j] == 0 and (num_y - i - 1) < 20:
                        # print(num_y - i - 1)
                        string = letter[int(num_y - i - 1)] + str(j + 1)
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
                    if pin_map[i][j] == 0 and (num_y - i - 1) > 19:
                        string = letter[int(((num_y - i) // 20) - 1)] + letter[int(((num_y - (i + 1)) % 20) )] + str(j + 1)
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])

    if pin_1_location[0] == 1:
        if pin_1_location[1] == 0 or pin_1_location[1] == -1:
            for i in range(len(pin_map)):
                for j in range(len(pin_map[i])):
                    if pin_map[i][j] == 0 and (i + 1) <= 20:
                        string = str(i + 1) + letter[int(j)]
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
                    if pin_map[i][j] == 0 and i > 19:
                        string = str(i + 1) + letter[int(((j + 1) // 20) - 1)] + letter[int(((j + 1) % 20) - 1)]
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
        elif pin_1_location[1] == 1:
            for i in range(len(pin_map)):
                for j in range(len(pin_map[i])):
                    if pin_map[i][j] == 0 and (i + 1) <= 20:
                        string = str(num_x - i) + letter[int(j)]
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
                    if pin_map[i][j] == 0 and i > 19:
                        string = str(num_x - i) + letter[int(((j + 1) // 20) - 1)] + letter[int(((j + 1) % 20) - 1)]
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
        elif pin_1_location[1] == 2:
            for i in range(len(pin_map)):
                for j in range(len(pin_map[i])):
                    if pin_map[i][j] == 0 and (i + 1) <= 20:
                        string = str(num_x - i) + letter[int(num_y - j - 1)]
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
                    if pin_map[i][j] == 0 and i > 19:
                        string = str(num_x - i) + letter[int(((num_y - j) // 20) - 1)] + letter[int(((j + 1) % 20) - 1)]
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
        elif pin_1_location[1] == 3:
            for i in range(len(pin_map)):
                for j in range(len(pin_map[i])):
                    if pin_map[i][j] == 0 and (i + 1) <= 20:
                        string = str(i + 1) + letter[int(num_y - j - 1)]
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
                    if pin_map[i][j] == 0 and i > 19:
                        string = str(i + 1) + letter[int(((num_y - j) // 20) - 1)] + letter[int(((j + 1) % 20) - 1)]
                        string = list(string)
                        string = ''.join(string)
                        lost_pin.append(string)
                        lost_color.append(color[i][j])
    print(lost_pin)
    print(lost_color)

    return lost_pin,  lost_color


def Output_list(body_x, body_y, pin_x_num, pin_y_num, pitch_x, pitch_y, high, pin_diameter, standoff, pin_map,
                pin_1_location, color):
    list1 = []
    # 初始化缺Pin相关变量
    loss_pin1 = []
    loss_color = []  # <<< 在 try 外部初始化

    body_x = f'{body_x}'
    body_y = f'{body_y}'
    high = f'{high}'

    pin_list1 = []
    pin_list2 = []
    top_len_list = []
    top_wid_list = []
    pitch_len_list = []
    pitch_len_list1 = []
    side_hi_list = []

    # pin_list1.append('Nx')
    pin_list1.append('-')
    pin_list1.append(f'{pin_x_num}')
    pin_list1.append('-')

    # pin_list2.append('Ny')
    pin_list2.append('-')
    pin_list2.append(f'{pin_y_num}')
    pin_list2.append('-')

    # top_len_list.append('body_x+tol')
    body_x = [float(item) for item in body_x.strip('[]').split()]
    if len(body_x) == 0:
        top_len_list.append(f'Extract error')
        top_len_list.append(f'None')
    elif len(body_x) == 1:
        top_len_list.append(f'{body_x[0]}')
        top_len_list.append('0')
    elif len(body_x) == 2:
        top_len_list.append(f'{body_x[1]}')
        top_len_list.append(f'{round(body_x[0] - body_x[1], 2)}')
    elif len(body_x) == 3:
        top_len_list.append(f'{body_x[0]}')
        top_len_list.append(f'{body_x[1]}')
        # top_len_list.append(f'{round(body_x[0]-body_x[1],2)},{round(body_x[2]-body_x[1],2)}')
        top_len_list.append(f'{body_x[2]}')

    # top_wid_list.append('body_y+tol')
    body_y = [float(item) for item in body_y.strip('[]').split()]
    if len(body_y) == 0:
        top_wid_list.append(f'Extract error')
        top_wid_list.append(f'None')
    elif len(body_y) == 1:
        top_wid_list.append(f'{body_y[0]}')
        top_wid_list.append('0')
    elif len(body_y) == 2:
        top_wid_list.append(f'{body_y[1]}')
        top_wid_list.append(f'{round(body_y[0] - body_y[1], 2)}')
    elif len(body_y) == 3:
        top_wid_list.append(f'{body_y[0]}')
        top_wid_list.append(f'{body_y[1]}')
        # top_wid_list.append(f'{round(body_y[0]-body_y[1],2)},{round(body_y[2]-body_y[1],2)}')
        top_wid_list.append(f'{body_y[2]}')
    # top_wid_list.append(f'{body_y[0],body_y[1],body_y[2]}')
    # top_wid_list.append('None')

    # pitch_len_list.append('pitch_x')
    if type(pitch_x) == 'int':
        pitch_x = [pitch_x]
    if type(pitch_y) == 'int':
        pitch_y = [pitch_y]
    if len(pitch_x) == 0:
        # pitch_len_list.append('Extract error')
        pitch_len_list.append('-')
        pitch_len_list.append('None')
        pitch_len_list.append('-')
    else:
        pitch_len_list.append('-')
        pitch_len_list.append(f'{pitch_x[0]}')
        pitch_len_list.append('-')

    # pitch_len_list.append('None')

    # pitch_len_list1.append('pitch_y')
    if len(pitch_y) == 0:
        # pitch_len_list1.append('Extract error')
        pitch_len_list1.append('-')
        pitch_len_list1.append('None')
        pitch_len_list1.append('-')
    else:
        pitch_len_list1.append('-')
        pitch_len_list1.append(f'{pitch_y[0]}')
        pitch_len_list1.append('-')

    # pitch_len_list1.append('None')

    # side_hi_list.append('package_height')
    high = [float(item) for item in high.strip('[]').split()]
    if len(high) == 0:
        # out of range的根源
        side_hi_list.append('-')
        side_hi_list.append('未检测出')
        side_hi_list.append('-')
    elif len(high) == 1:
        side_hi_list.append(f'{high[0]}')
        side_hi_list.append('0')
    elif len(high) == 2:
        side_hi_list.append(f'{high[1]}')
        side_hi_list.append(f'{round(high[0] - high[1], 2)}')
    elif len(high) == 3:
        side_hi_list.append(f'{high[0]}')
        side_hi_list.append(f'{high[1]}')
        # side_hi_list.append(f'{round(high[0]-high[1],2)},{round(high[2]-high[1],2)}')
        side_hi_list.append(f'{high[2]}')
    # loss_pin, loss_color = Is_Loss_Pin(pin_map, pin_1_location, color)
    try:
        loss_pin, loss_color = Is_Loss_Pin(pin_map, pin_1_location, color)
    except:
        loss_pin = []
        loss_color = []
    if len(loss_pin) == 0:
        loss_pin1 = 'None'
    else:
        loss_pin1 = loss_pin
    list1.append(top_len_list)  # 实体长
    list1.append(top_wid_list)  # 实体宽
    list1.append(side_hi_list)  # 实体高
    if len(standoff) != 0:
        list1.append([f'{standoff[0][0]}', f'{standoff[0][1]}', f'{standoff[0][2]}'])  # 支撑高
    else:
        list1.append(['-', '未检测出', '-'])
    if len(pin_diameter) != 0:
        list1.append([f'{pin_diameter[0][0]}', f'{pin_diameter[0][1]}', f'{pin_diameter[0][2]}'])  # 球直径
    else:
        list1.append(['-', '未检测出', '-'])
    list1.append(pin_list2)  # 行数
    list1.append(pin_list1)  # 列数

    list1.append(pitch_len_list)  # 行Pitch
    list1.append(pitch_len_list1)
    list1.append([f'{loss_color}', f'{loss_pin1}', '-'])  # 缺pin否

    return list1


def filter_black_point(filename, save_path):
    img = cv2.imread(filename, 0)
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=1)
    cv2.imwrite(save_path, dilate)
    # img = Image.open(filename)
    # img = img.convert('L')
    # cleaned_page_array = cv2.adaptiveThreshold(np.array(img),
    #                                            255,
    #                                            # cv2.ADAPTIVE_THRESH_MEAN_C, #基于邻域均值的自适应阈值。
    #                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 基于邻域加权平均的自适应阈值。
    #                                            cv2.THRESH_BINARY,
    #                                            19,
    #                                            15)
    # new_image = np.array(Image.fromarray(cleaned_page_array))
    # cv2.imwrite(save_path, new_image)


def BGA_clear_creat():

    empty_folder(BGA_BOTTOM)
    try:
        os.makedirs(BGA_BOTTOM)
    except:
        print("文件夹bga_bottom已存在")
    empty_folder(PINMAP)

    filein = f'{DATA}/top.jpg'
    fileout = filein
    set_Image_size(filein, fileout)
    # filter_black_point(filein, filein)
    filein = f'{DATA}/bottom.jpg'

    shutil.copy(filein, f'{BGA_BOTTOM}/bottom.jpg')
    fileout = filein
    set_Image_size(filein, fileout)
    # filter_black_point(filein, filein)
    filein = f'{DATA}/side.jpg'
    fileout = filein
    set_Image_size(filein, fileout)
    # filter_black_point(filein, filein)

    empty_folder(ONNX_OUTPUT)
    # 创建文件夹
    os.makedirs(ONNX_OUTPUT)
    empty_folder(DATA_COPY)
    # 创建文件夹
    os.makedirs(DATA_COPY)
    empty_folder(DATA_BOTTOM_CROP)
    # 创建文件夹
    os.makedirs(DATA_BOTTOM_CROP)
    filePath = DATA
    file_name_list = os.listdir(filePath)
    for file_name in file_name_list:
        shutil.copy(f'{DATA}/{file_name}', f'{DATA_COPY}/{file_name}')


def long_running_task(result_queue, bottom_border):
    print()
    print("***/开始检测pin/***")
    result, color_map = find_pin(bottom_border)
    # try:
    #     result = find_pin()
    # except:
    #     print("pinmap没有正常读取，请记录pdf并反馈")
    #     result = np.ones((10, 10))
    result_queue.put(result, color_map)
    print("***/结束检测pin/***")
    print()


def time_save_find_pinmap(bottom_border):
    result_queue = queue.Queue()
    thread = threading.Thread(target=long_running_task, args=(result_queue, bottom_border))
    thread.start()

    thread.join(timeout=6)  # 设置超时时间为5秒
    if thread.is_alive():
        print("读取pinmap进程花费时间过长，跳过")
        pin_map = np.ones((10, 10))
        # 记录pin的行列数
        pin_num_x_y = np.array([0, 0])
        pin_num_x_y = pin_num_x_y.astype(int)
        path = f'{YOLOX_DATA}\pin_num.txt'
        np.savetxt(path, pin_num_x_y)
    else:
        try:
            pin_map, color_map = result_queue.get_nowait()  # 尝试获取结果
            # print("Result:", pin_map)
        except queue.Empty:
            print("Queue is empty, no result available.")
            pin_map = np.ones((10, 10))
            color_map = np.full_like(pin_map, fill_value=2)
            # 记录pin的行列数
            pin_num_x_y = np.array([0, 0])
            pin_num_x_y = pin_num_x_y.astype(int)
            path = f'{YOLOX_DATA}\pin_num.txt'
            np.savetxt(path, pin_num_x_y)
    # 记录pin的行列数
    pin_num_x_y = np.array([pin_map.shape[1], pin_map.shape[0]])
    pin_num_x_y = pin_num_x_y.astype(int)
    path = f'{YOLOX_DATA}\pin_num.txt'
    np.savetxt(path, pin_num_x_y)

    return pin_map, color_map


def extract_BGA(page_num, letter_or_number, table_dic):
    # 检测top、bottom、side视图是否存在，不存在则创建一张纯白图代替
    # path = 'data'
    # detect_img_top_bottom_side(path)
    print("table_dic:\n", table_dic)
    test_mode = 0
    # 清除缓存并创建文件夹
    BGA_clear_creat()

    pin_output = 1  #（1表示在该函数中输出了正确的pinmap）

    # 6.引线方法直接提取参数
    body_x_yinXian, body_y_yinXian, pitch_x_yinXian, pitch_y_yinXian, high_yinXian, pin_diameter_yinXian, standoff_yinXian, \
        pin_num_x_serial, pin_num_y_serial, pin_1_location, \
        yolox_pairs_top_copy, yolox_pairs_bottom_copy, yolox_pairs_side_copy, \
        letter_or_number, top_ocr_data, bottom_ocr_data, side_ocr_data, pin_map, color = \
        yinXian_begain_get_data_present(test_mode, letter_or_number, table_dic)

    if letter_or_number == 'number':
        # 8.漏缺参数用数据比较法筛选出
        # body_x, body_y, pin_x_num, pin_y_num, pitch_x, pitch_y, high, pin_diameter, standoff, pin_map_present = begain_get_pairs_data_present2()
        body_x, body_y, pin_x_num, pin_y_num, pitch_x, pitch_y, high, pin_diameter, standoff, pin_map_present = begain_get_pairs_data_present2(
            body_x_yinXian, body_y_yinXian, pitch_x_yinXian, pitch_y_yinXian, high_yinXian, pin_diameter_yinXian,
            standoff_yinXian, pin_num_x_serial, pin_num_y_serial, yolox_pairs_top_copy, yolox_pairs_bottom_copy,
            yolox_pairs_side_copy, pin_1_location, test_mode, top_ocr_data, bottom_ocr_data, side_ocr_data)
        print("********************总结********************")
        print("body_x_yinXian, body_y_yinXian(max_medium_min)\n", body_x_yinXian, body_y_yinXian)
        print("pin_num_x_serial, pin_num_y_serial, pin_1_location\n", pin_num_x_serial, pin_num_y_serial,
              pin_1_location)
        print("pitch_x_yinXian, pitch_y_yinXian(max_medium_min)\n", pitch_x_yinXian, pitch_y_yinXian)
        print("high_yinXian(max_medium_min)\n", high_yinXian)
        print("pin_diameter_yinXian(max_medium_min)\n", pin_diameter_yinXian)
        print("standoff_yinXian(max_medium_min)\n", standoff_yinXian)
        print("**********************************************")
        print("body_x,body_y(max_medium_min)\n", body_x, body_y)
        print("pin_x_num,pin_y_num\n", pin_x_num, pin_y_num)
        print("pitch_x,pitch_y\n", pitch_x, pitch_y)
        print("high(max_medium_min)", high)
        print("pin_diameter(max_medium_min)\n", pin_diameter)
        print("standoff\n", standoff)
        if pin_output == 1:
            try:
                present_pin_map(pin_map)
            except:
                print("未识别到pin")
        if pin_output != 1:
            try:
                present_pin_map(pin_map_present)
            except:
                print("未识别到pin")
        print("****************************************************")
        # 修正输出的格式
        if pin_diameter.ndim == 1:
            zer = np.zeros((0, 3))
            pin_diameter = np.r_[zer, [pin_diameter]]
        if standoff.ndim == 1:
            zer = np.zeros((0, 3))
            standoff = np.r_[zer, [standoff]]
        # output_excel(body_x, body_y, pin_x_num, pin_y_num, pitch_x, pitch_y, high, pin_diameter, standoff, pin_map_present, pdf_name, pageNumbers)
    if letter_or_number == 'table':
        print("********************总结********************")
        print("body_x_yinXian, body_y_yinXian(max_medium_min)\n", body_x_yinXian, body_y_yinXian)
        print("pin_num_x_serial, pin_num_y_serial, pin_1_location\n", pin_num_x_serial, pin_num_y_serial,
              pin_1_location)
        print("pitch_x_yinXian, pitch_y_yinXian(max_medium_min)\n", pitch_x_yinXian, pitch_y_yinXian)
        print("high_yinXian(max_medium_min)\n", high_yinXian)
        print("pin_diameter_yinXian(max_medium_min)\n", pin_diameter_yinXian)
        print("standoff_yinXian(max_medium_min)\n", standoff_yinXian)
        body_x = body_x_yinXian
        body_y = body_y_yinXian
        pin_x_num = pin_num_x_serial
        pin_y_num = pin_num_y_serial
        pitch_x = pitch_x_yinXian
        standoff = standoff_yinXian
        pitch_y = pitch_y_yinXian
        high = high_yinXian
        pin_diameter = pin_diameter_yinXian
        if pin_output != 1:
            pin_map_present = get_pinmap_table()
        if pin_output == 1:
            try:
                present_pin_map(pin_map)
            except:
                print("未识别到pin")
        if pin_output != 1:
            try:
                present_pin_map(pin_map_present)
            except:
                print("未识别到pin")
        print("****************************************************")
        # 修正输出的格式
        if pin_diameter.ndim == 1:
            zer = np.zeros((0, 3))
            pin_diameter = np.r_[zer, [pin_diameter]]
        if standoff.ndim == 1:
            zer = np.zeros((0, 3))
            standoff = np.r_[zer, [standoff]]
    body_x = np.round(body_x, 3)
    body_y = np.round(body_y, 3)
    high = np.round(high, 3)
    pin_diameter = np.round(pin_diameter, 3)
    standoff = np.round(standoff, 3)
    pin_x_num = round(pin_x_num)
    pin_y_num = round(pin_y_num)

    if pin_output != 1:
        result_list = Output_list(body_x, body_y, pin_x_num, pin_y_num, pitch_x, pitch_y, high, pin_diameter,
                                  standoff, pin_map_present, pin_1_location, color)
    elif pin_output == 1:
        result_list = Output_list(body_x, body_y, pin_x_num, pin_y_num, pitch_x, pitch_y, high, pin_diameter,
                                  standoff, pin_map, pin_1_location, color)

    result_list = [sublist[::-1] for sublist in result_list]
    result_list = [[0] + sublist if i < len(result_list) - 1 else sublist for i, sublist in enumerate(result_list)]

    # ['实体长D', '实体宽E', '实体高A', '支撑高A1', '球直径b', '球行数', '球列数', '行球间距e', '列球间距e', '缺PIN否']
    print(result_list)
    # 20250722添加
    # 指定要查找的 page_num
    target_page_num = page_num
    json_file = 'output.json'
    result = []
    # 读取 JSON 文件
    print("正在读取JSON文件...")
    # with open(json_file, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    with open(json_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content:
            try:
                data = json.loads(content)
                print("JSON解析成功")
                for item in data:
                    if item['page_num'] == target_page_num:
                        result.append(item['pin'])
                        result.append(item['length'])
                        result.append(item['width'])
                        result.append(item['height'])
                        result.append(item['horizontal_pin'])
                        result.append(item['vertical_pin'])
                print("json文件读取完毕")
                print("json:", result)
            except json.JSONDecodeError as e:
                print("JSON 解析失败:", e)
        else:
            print("文件为空")
    # 遍历列表，查找匹配的条目


    if result != []:
        if result[0] != None:
            if result[4] != None and result[5] != None:
                if abs(result[4] * result[5] - result[0]) < 1e-9 and abs(result_list[5][2] * result_list[6][2] - result[4] * result[5]) > 1e-9:
                    result_list[5][2] = result[4]
                    result_list[6][2] = result[5]
                if result_list[5][2] == 0 and result[4] != None:
                    result_list[5][2] = result[4]
                if result_list[6][2] == 0 and result[5] != None:
                    result_list[6][2] = result[5]
        print("result_list", result_list)

        try:
            length = float(result_list[0][2])
        except:
            print("无法转化为浮点数length", result_list[0][2])
        try:
            weight = float(result_list[1][2])
        except:
            print("无法转化为浮点数weight", result_list[1][2])
        try:
            height = float(result_list[2][2])
        except:
            print("无法转化为浮点数height", result_list[2][2])

        if result[1] != None and result[1] != length and (result[1] != weight and result[2] != length):
            result_list[0][1] = ''
            result_list[0][2] = result[1]
            result_list[0][3] = ''
        if result[2] != None and result[2] != weight and (result[1] != weight and result[2] != length):
            result_list[1][1] = ''
            result_list[1][2] = result[2]
            result_list[1][3] = ''
        if result[3] != None and result[3] != height:
            result_list[2][1] = ''
            result_list[2][2] = result[3]
            result_list[2][3] = ''
    # 文本输出列表：
    # # 20250723改变顺序
    new_result_list = []
    new_result_list.append(result_list[7])
    new_result_list.append(result_list[8])
    new_result_list.append(result_list[6])
    new_result_list.append(result_list[5])
    new_result_list.append(result_list[2])
    new_result_list.append(result_list[3])
    new_result_list.append(result_list[0])
    new_result_list.append(result_list[1])
    new_result_list.append([0,'-','-','-'])
    new_result_list.append(result_list[4])
    new_result_list.append(result_list[9])
    return new_result_list


if __name__ == "__main__":
    extract_BGA(letter_or_number='number', table_dic=[])
