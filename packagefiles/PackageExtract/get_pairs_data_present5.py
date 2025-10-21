import math
from math import sqrt
import cv2
import numpy as np
import shutil
import time
import os
import operator
import re
from random import randint
import queue
import threading
from packagefiles.PackageExtract.QFP_extract import *
# from yolox_onnx_py.onnx_QFP_pairs_data_location2 import begain_output_QFP_pairs_data_location
# from yolox_onnx_py.onnx_output_other_location import begain_output_other_location
from packagefiles.PackageExtract.yolox_onnx_py.onnx_output_serial_number_letter_location import begain_output_serial_number_letter_location
from packagefiles.PackageExtract.yolox_onnx_py.onnx_output_side_body_standoff_location import begain_output_side_body_standoff_location
from packagefiles.PackageExtract.yolox_onnx_py.onnx_output_pin_yinXian_find_pitch import begain_output_pin_location
from packagefiles.PackageExtract.yolox_onnx_py.onnx_output_top_body_location import begain_output_top_body_location
from packagefiles.PackageExtract.yolox_onnx_py.onnx_output_bottom_body_location import begain_output_bottom_body_location
from packagefiles.PackageExtract.BGA_cal_pin import *
# from packagefiles.PackageExtract.BGA_extract_old import time_save_find_pinmap
# from output_QFP_pairs_data_location2 import begain_output_QFP_pairs_data_location
# from output_other_location import begain_output_other_location
# from output_serial_number_letter_location import begain_output_serial_number_letter_location
# from output_side_body_standoff_location import begain_output_side_body_standoff_location
# from output_pin_yinXian_find_pitch import begain_output_pin_location
# from output_top_body_location import begain_output_top_body_location
# from output_bottom_body_location import begain_output_bottom_body_location
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
page_path = 'Result/Package_view/page'

from packagefiles.PackageExtract.onnx_use import Run_onnx_det
from packagefiles.PackageExtract.onnx_use import Run_onnx


# from system_test import Dbnet_Inference

# from output_body import output_body


def choose_x(binary):
    """在二值图像中筛选水平长条轮廓并绘制对应直线。"""
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
    """在二值图像中筛选竖直长条轮廓并绘制对应直线。"""
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


def output_body(img_path, name):
    """通过形态学处理提取外框轮廓并输出矩形坐标。"""
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
    file_name = 'opencv_output/' + name + 'output_waikuang.jpg'
    cv2.imwrite(file_name, rec)
    # cv2.namedWindow('body', 0)
    # cv2.imshow('body', rec)
    # cv2.waitKey(0)
    try:
        location = np.array([[x, y, x + w - 3, y + h - 3]])
    except:
        location = np.array([])
        print("opencv函数找不到外框")
    return location


def find_all_lines(img_path, test_mode):
    """结合形态学与线段检测获取图像全部水平和竖直线段。"""
    # img_path = r'data_copy/bottom.jpg'
    src_img = cv2.imread(img_path)
    src_img1 = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    # src_img1 = cv2.GaussianBlur(src_img1, (3, 3), 0)
    thresh, AdaptiveThreshold = cv2.threshold(src_img1, 240, 255, 0)
    AdaptiveThreshold = cv2.bitwise_not(AdaptiveThreshold)

    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()

    horizontalSize = int(horizontal.shape[1] / 25)  # 默认40
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    verticalSize = int(vertical.shape[0] / 25)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    syz_heng = []
    syz_shu = []

    contours1, hierarchy1 = cv2.findContours(horizontal, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in contours1:
        x, y, w, h = cv2.boundingRect(cnt)
        xq = x
        yq = int(y + (h / 2))
        xz = x + w
        yz = int(y + (h / 2))
        syz_heng.append([min(xq, xz), min(yq, yz), max(xq, xz), max(yq, yz)])
        # [min(xq, xz), min(yq, yz), max(xq, xz), max(yq, yz)]

    contours2, hierarchy2 = cv2.findContours(vertical, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in contours2:
        x, y, w, h = cv2.boundingRect(cnt)
        xq = int(x + (w / 2))
        yq = y
        xz = int(x + (w / 2))
        yz = y + h
        syz_shu.append([min(xq, xz), min(yq, yz), max(xq, xz), max(yq, yz)])
    syz_heng = np.array(syz_heng)
    syz_shu = np.array(syz_shu)

    if test_mode == 1:
        for point in syz_heng:
            cv2.line(src_img, [point[0], point[1]], [point[2], point[3]], (0, 0, 255), 2)
        for point in syz_shu:
            cv2.line(src_img, [point[0], point[1]], [point[2], point[3]], (0, 0, 255), 2)
        cv2.namedWindow('all_line', 0)
        cv2.imshow("all_line", src_img)
        cv2.waitKey(0)

    return syz_heng, syz_shu


def get_rotate_crop_image(img, points):  # 图片分割，在ultil中的原有函数,from utils import get_rotate_crop_image
    """按照点集坐标仿射裁剪出旋转纠正后的图像块。"""
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE,
                                  flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    # if dst_img_height * 1.0 / dst_img_width >= 1:
    #     dst_img = np.rot90(dst_img)
    return dst_img


def ocr_get_data(image_path,
    """封装 OCR 推理流程，按批处理图片并统计耗时。"""
                 yolox_pairs):  # 输入yolox输出的pairs坐标和匹配的data坐标以及图片地址，ocr识别文本后输出data内容按序保存在data_list_np（numpy二维数组）
    show_img_key = 0  # 是否显示过程中ocr待检测图片 0 = 不显示，1 = 显示
    yolox_pairs = np.array(yolox_pairs)
    ocr = PaddleOCR(use_angle_cls=True,
                    lang="en",
                    det_model_dir="ppocr_model/det/en/en_PP-OCRv3_det_infer",
                    rec_model_dir='ppocr_model/rec/en/en_PP-OCRv3_rec_infer',
                    cls_model_dir='ppocr_model/cls/ch_ppocr_mobile_v2.0_cls_infer',
                    use_gpu=False)  # 导入模型， 禁用gpu
    with open(image_path, 'rb') as f:
        np_arr = np.frombuffer(f.read(), dtype=np.uint8)
        # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    #以彩图读取
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取
    data_list = []  # 按序存储pairs的data
    error_times = 0  # 记录ocr识别为空的次数
    no = 1  # ocr识别图片的序号
    # 针对yolox_pairs中的data在图片中的坐标，裁剪出图片区域用来给ocr检测
    for i in range(len(yolox_pairs)):
        # 裁剪识别区域的时候需要扩展一圈，以防yolox极限检测框导致某些数据边缘没有被检测
        # 只可能横着的裁剪成竖着的

        # 方案：横着的一定图片方向正确，扩展一圈识别，如果没有识别到，等比例扩大再识别；竖着的图片看宽长比是否小于0.7，小于则顺时针旋转90，如果识别不到，先认为是文本方向错误，逆时针90转回来识别。如果还识别不到，则等比例放大重复上述
        print("*********************************************************", yolox_pairs[i])
        box = np.array([[yolox_pairs[i][8], yolox_pairs[i][9]], [yolox_pairs[i][10], yolox_pairs[i][9]],
                        [yolox_pairs[i][10], yolox_pairs[i][11]], [yolox_pairs[i][8], yolox_pairs[i][11]]], np.float32)
        box_img = get_rotate_crop_image(img, box)  # yolox检测的原始data区域
        height, width = box_img.shape[0], box_img.shape[1]
        # ****************
        box_img = cv2.resize(box_img, (320, int(height * 320 / width)))  # 等比例放缩成320，适合ocr识别
        # ****************
        # print(yolox_pairs[i][12])
        if show_img_key == 1:
            cv2.imshow('no_correct_img', box_img)  # 显示当前ocr的识别区域
            cv2.waitKey(0)
        if yolox_pairs[i][12] == 0:
            KuoZhan_ratio = 0.25  # 扩展的比例
            KuoZhan_x = KuoZhan_ratio * abs(yolox_pairs[i][10] - yolox_pairs[i][8]) * (0.5)
            KuoZhan_y = KuoZhan_ratio * abs(yolox_pairs[i][11] - yolox_pairs[i][9]) * (0.5)
            box = np.array([[yolox_pairs[i][8] - KuoZhan_x, yolox_pairs[i][9] - KuoZhan_y],
                            [yolox_pairs[i][10] + KuoZhan_x, yolox_pairs[i][9] - KuoZhan_y],
                            [yolox_pairs[i][10] + KuoZhan_x, yolox_pairs[i][11] + KuoZhan_y],
                            [yolox_pairs[i][8] - KuoZhan_x, yolox_pairs[i][11] + KuoZhan_y]], np.float32)

            box_img = get_rotate_crop_image(img, box)

        if yolox_pairs[i][12] == 1:  # 竖着的data需要旋转90,如果识别不到那就是本来文本是横着结果截取范围是竖着的
            KuoZhan_ratio = 0.25
            KuoZhan_x = KuoZhan_ratio * abs(yolox_pairs[i][10] - yolox_pairs[i][8]) * (0.5)
            KuoZhan_y = KuoZhan_ratio * abs(yolox_pairs[i][11] - yolox_pairs[i][9]) * (0.5)
            box = np.array([[yolox_pairs[i][8] - KuoZhan_x, yolox_pairs[i][9] - KuoZhan_y],
                            [yolox_pairs[i][10] + KuoZhan_x, yolox_pairs[i][9] - KuoZhan_y],
                            [yolox_pairs[i][10] + KuoZhan_x, yolox_pairs[i][11] + KuoZhan_y],
                            [yolox_pairs[i][8] - KuoZhan_x, yolox_pairs[i][11] + KuoZhan_y]], np.float32)
            box_img = get_rotate_crop_image(img, box)
            print("(yolox_pairs[i][10] - yolox_pairs[i][8])/(yolox_pairs[i][11]-yolox_pairs[i][9])",
                  (yolox_pairs[i][10] - yolox_pairs[i][8]) / (yolox_pairs[i][11] - yolox_pairs[i][9]))
            rotate_key = 0
            length_to_weight = 0.7  # 长宽比 小于1
            if (yolox_pairs[i][10] - yolox_pairs[i][8]) / (yolox_pairs[i][11] - yolox_pairs[i][9]) < length_to_weight:
                rotate_key = 1
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90

        print("正在OCR识别第", no, "个pairs匹配的data")
        no += 1
        ############################################################
        height, width = box_img.shape[0], box_img.shape[1]
        box_img = cv2.resize(box_img, (320, int(height * 320 / width)))  # 等比例放缩成320，适合ocr识别
        ################################################################
        if show_img_key == 1:
            cv2.imshow('origin_img', box_img)  # 显示当前ocr的识别区域
            cv2.waitKey(0)

        result = ocr.ocr(box_img, cls=True, det=True, rec=True)  # 问题：有可能yolox检测不到data，需要在yolox检测方面解决

        if (result == [None] or result == [[]]) and yolox_pairs[i][12] == 1 and rotate_key == 0:
            box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
            rotate_key = 1
            if show_img_key == 1:
                cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                cv2.waitKey(0)
            result = ocr.ocr(box_img, cls=True, det=True, rec=True)

        if (result == [None] or result == [[]]) and yolox_pairs[i][12] == 1 and rotate_key == 1:
            box_img = cv2.rotate(box_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90
            if show_img_key == 1:
                cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                cv2.waitKey(0)
            result = ocr.ocr(box_img, cls=True, det=True, rec=True)

        if result == [None] or result == [[]]:  # 如果识别不到，那么把图片”等比例“放大，不等比例会导致OCR识别失败
            if yolox_pairs[i][12] == 0:
                KuoZhan_ratio = 0.25
                KuoZhan_x = KuoZhan_ratio * abs(yolox_pairs[i][10] - yolox_pairs[i][8]) * (0.5)
                KuoZhan_y = KuoZhan_ratio * abs(yolox_pairs[i][11] - yolox_pairs[i][9]) * (0.5)
                box = np.array([[yolox_pairs[i][8] - KuoZhan_x, yolox_pairs[i][9] - KuoZhan_y],
                                [yolox_pairs[i][10] + KuoZhan_x, yolox_pairs[i][9] - KuoZhan_y],
                                [yolox_pairs[i][10] + KuoZhan_x, yolox_pairs[i][11] + KuoZhan_y],
                                [yolox_pairs[i][8] - KuoZhan_x, yolox_pairs[i][11] + KuoZhan_y]], np.float32)

                box_img = get_rotate_crop_image(img, box)

            if yolox_pairs[i][12] == 1:  # 竖着的data需要旋转90,如果识别不到那就是本来文本是横着结果截取范围是竖着的
                KuoZhan_ratio = 0.25
                KuoZhan_x = KuoZhan_ratio * abs(yolox_pairs[i][10] - yolox_pairs[i][8]) * (0.5)
                KuoZhan_y = KuoZhan_ratio * abs(yolox_pairs[i][11] - yolox_pairs[i][9]) * (0.5)
                box = np.array([[yolox_pairs[i][8] - KuoZhan_x, yolox_pairs[i][9] - KuoZhan_y],
                                [yolox_pairs[i][10] + KuoZhan_x, yolox_pairs[i][9] - KuoZhan_y],
                                [yolox_pairs[i][10] + KuoZhan_x, yolox_pairs[i][11] + KuoZhan_y],
                                [yolox_pairs[i][8] - KuoZhan_x, yolox_pairs[i][11] + KuoZhan_y]], np.float32)
                box_img = get_rotate_crop_image(img, box)
                print("(yolox_pairs[i][10] - yolox_pairs[i][8])/(yolox_pairs[i][11]-yolox_pairs[i][9])",
                      (yolox_pairs[i][10] - yolox_pairs[i][8]) / (yolox_pairs[i][11] - yolox_pairs[i][9]))
                rotate_key = 0
                if (yolox_pairs[i][10] - yolox_pairs[i][8]) / (yolox_pairs[i][11] - yolox_pairs[i][9]) < 0.75:
                    rotate_key = 1
                    box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
            # box_img = cv2.resize(box_img, (160, 80),cv2.INTER_AREA)
            box_img = img_resize(box_img)  # 等比例放大
            box_img = hist(box_img, show_img_key)  # 图像增强

            if show_img_key == 1:
                cv2.imshow('enhance_origin_img', box_img)  # 显示当前ocr的识别区域
                cv2.waitKey(0)

            result = ocr.ocr(box_img, cls=True, det=True, rec=True)  # 问题：有可能yolox检测不到data，需要在yolox检测方面解决

            if (result == [None] or result == [[]]) and yolox_pairs[i][12] == 1 and rotate_key == 0:
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
                rotate_key = 1
                if show_img_key == 1:
                    cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                    cv2.waitKey(0)
                result = ocr.ocr(box_img, cls=True, det=True, rec=True)

            if (result == [None] or result == [[]]) and yolox_pairs[i][12] == 1 and rotate_key == 1:
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90
                if show_img_key == 1:
                    cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                    cv2.waitKey(0)
                result = ocr.ocr(box_img, cls=True, det=True, rec=True)

        print("OCR识别data的result\n", result)
        # 将ocr识别的文本提取数据
        try:
            if yolox_pairs[i][4] == 1:
                data_list_pairs = [yolox_pairs[i][4],
                                   yolox_pairs[i][3] - yolox_pairs[i][1], yolox_pairs[i][8], yolox_pairs[i][9],
                                   yolox_pairs[i][10], yolox_pairs[i][11], ]  # [识别箭头方向,pairs的长度,data坐标,识别内容1，识别内容2，...]
            if yolox_pairs[i][4] == 0 or yolox_pairs[i][4] == 0.5:
                data_list_pairs = [yolox_pairs[i][4],
                                   yolox_pairs[i][2] - yolox_pairs[i][0], yolox_pairs[i][8], yolox_pairs[i][9],
                                   yolox_pairs[i][10], yolox_pairs[i][11], ]  # [识别箭头方向,识别内容1，data坐标，识别内容2，...]

            # data_list_pairs = [yolox_pairs[i][4], ]  # [识别箭头方向,识别内容1，识别内容2，...]
            for i in range(len(result[0])):
                # result[0][i][1][0] = comma_inter_point((result[0][i][1][0]))
                if get_data_and_del_en(comma_inter_point(result[0][i][1][0])) != '':
                    if get_data_and_del_en(comma_inter_point(result[0][i][1][0])) == []:
                        continue
                    else:
                        data_list_pairs_one, data_list_pairs_another = get_data_and_del_en(
                            comma_inter_point(result[0][i][1][0]))
                        print("len(data_list_pairs_one),data_list_pairs_one\n", len(data_list_pairs_one), ",",
                              data_list_pairs_one)
                        print("len(data_list_pairs_another),data_list_pairs_another\n", len(data_list_pairs_another),
                              ",",
                              data_list_pairs_another)
                        if len(data_list_pairs_another) == 0 and data_list_pairs_one != [''] and len(
                                data_list_pairs_one) != 0:
                            # data_list_pairs.append(get_data_and_del_en(comma_inter_point(result[0][i][1][0])))
                            data_list_pairs.append(data_list_pairs_one)
                            print("data_list_pairs\n", data_list_pairs)
                        # if len(data_list_pairs_another) != 0:
                        if data_list_pairs_another != [''] and len(
                                data_list_pairs_another) != 0:  # 识别到类似"12+0.1"的文本时，get_data_and_del_en会返回12和0.1,之后判断大小来得到max，medium，min
                            data_list_pairs.append(data_list_pairs_one)
                            # print("data_list_pairs", data_list_pairs)
                            data_list_pairs.append(data_list_pairs_another)
                            print("data_list_pairs\n", data_list_pairs)
        except Exception as r:
            print('OCR未识别到内容，报错： %s' % (r))
            error_times += 1

        data_list.append(data_list_pairs)
    # 对data_list中只存在min和max的值，以及只有一个median的，扩展为max，median和min
    print("data_list：一张img中的所有文本，按行OCR识别的结果，可以有多行的输出结果:\n", data_list)
    compire_ratio = 0.5
    data_list_np = np.empty((0, 9))
    # 将ocr识别出的数据按照max，medium，min保存在data_list_np中
    for i in range(len(data_list)):
        # try:
        m_m_m = np.asarray(data_list[i])
        m_m_m = m_m_m.astype(np.float_)  # 识别出来的数据格式不对就会报错
        if len(m_m_m) == 6:
            continue
        if len(m_m_m) == 8:
            if m_m_m[6] > m_m_m[7] and m_m_m[7] / m_m_m[6] > compire_ratio:
                m_m_m = np.append(m_m_m, (m_m_m[6] + m_m_m[7]) / 2)
                min = m_m_m[7]
                m_m_m[7] = m_m_m[8]
                m_m_m[8] = min
            elif m_m_m[6] < m_m_m[7] and m_m_m[6] / m_m_m[7] > compire_ratio:
                max = m_m_m[7]
                m_m_m[7] = m_m_m[6]
                m_m_m[6] = max
                m_m_m = np.append(m_m_m, (m_m_m[6] + m_m_m[7]) / 2)
            elif m_m_m[6] > m_m_m[7] and m_m_m[7] / m_m_m[6] < compire_ratio:
                max = m_m_m[6] + m_m_m[7]
                min = m_m_m[6] - m_m_m[7]
                medium = m_m_m[6]
                m_m_m[6] = max
                m_m_m[7] = medium
                m_m_m = np.append(m_m_m, min)
            elif m_m_m[6] < m_m_m[7] and m_m_m[6] / m_m_m[7] < compire_ratio:
                max = m_m_m[7] + m_m_m[6]
                min = m_m_m[7] - m_m_m[6]
                medium = m_m_m[7]
                m_m_m[6] = max
                m_m_m[7] = medium
                m_m_m = np.append(m_m_m, min)
            elif math.isclose(m_m_m[7], m_m_m[6]):
                m_m_m = np.append(m_m_m, m_m_m[6])
        if len(m_m_m) == 7:
            m_m_m = np.append(m_m_m, m_m_m[6])
            m_m_m = np.append(m_m_m, m_m_m[6])
        #
        if len(m_m_m) == 9:  # 规定m_m_m的格式，过滤掉OCR识别出来的非数字项
            y = m_m_m[6:]
            # x.sort()  # [1 2 3 4 6 8]
            # x = abs(np.sort(-x))  # [8 6 4 3 2 1] 先取相反数排序，再加上绝对值得到原数组的降序
            y = np.sort(y)
            y = abs(np.sort(-y))
            m_m_m[6:] = y
            # data_list_np = np.c_[data_list_np, m_m_m]  # 添加行
        print("中间量data_list_np\n", data_list_np)
        print("即将添加到data_list_np的m_m_m\n", m_m_m)
        data_list_np = np.r_[data_list_np, [m_m_m]]
        print("添加了m_m_m之后的data_list_np\n", data_list_np)
        # except Exception as e:
        #     print("*******报错*******\n",e)
    # data_list_np = data_list_np.T

    print(
        "该视图下最终结果data_list_np：[[方向,pairs的长度,data坐标,max,medium,min],[方向,pairs的长度,data坐标，max,medium,min],...]:\n",
        data_list_np)
    # data_list_np：[[方向, max, medium, min], [方向, max, medium, min], ...]

    # print("data_list",data_list)
    if error_times != 0:
        print("在该视图下有", error_times, "个data是OCR识别为空的,请检查并优化识别效果")
    # print("在该视图下有",error_times,"个data是OCR识别为空的,请检查并优化识别效果")
    return data_list, data_list_np


def img_resize(image):
    """将图像缩放至指定尺寸，便于统一处理。"""
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 160
    height_new = 80
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new


def img_clear(img):
    """对输入图像进行滤波降噪以提升识别质量。"""
    img = cv2.bilateralFiler(img, 9, 75, 75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    ret, thresh = cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MOEPH_RECT, (3, 3))
    img = cv2.dilate(thresh, kernel)
    return img


def comma_inter_point(str_data):  # 将字符串中的comma转换为point
    """把字符串中的逗号替换为小数点，兼容不同格式。"""
    str_data = list(str_data)  # str不可修改，转换成list可以修改元素
    for i in range(len(str_data)):
        if str_data[i] == ',':
            str_data[i] = '.'
    str_data = ''.join(str_data)  # 将列表转换成字符串
    return str_data


def jump_inter_comma(str_data):
    """删除字符串中多余的逗号字符。"""
    str_data = list(str_data)
    for i in range(len(str_data)):
        if str_data[i] == ' ':
            str_data[i] = ','
    str_data = ''.join(str_data)
    return str_data


def get_data_and_del_en(string):  # 将输入字符串，从中提取数字（含小数点），删除中英文
    """提取字符串里的数值并清理中英文字符。"""
    # import re
    # string = "轻型车：共有198家企业4747个车型（12305个信息公开编号）15498915辆车进行了轻型车国六环保信息公开，与上周汇总环比增加105个车型、386379辆车。其中，国内生产企业177家、4217个车型、14645390辆，国外生产企业21家、530个车型、853525辆；轻型汽油"
    # 打印结果：['198', '4747', '12305', '15498915', '105', '386379', '177', '4217', '14645390', '21', '530', '853525']
    # 最终打印str:"198474712305..."
    str_data = []
    str_data_another = []
    # str_data = re.findall("\d+\.?\d*", string)  # 正则表达式:小数或者整数
    str_data = re.findall("\d*\.?\d*", string)  # 正则表达式:小数或者整数 + .40
    # 问题：如果string是6.250.10时怎么解决：
    # 方法1：根据小数点数量判断是#公差类型#还是#后面括号包含英寸#
    # 实际方法：对正则输出两个数的默认为公差类型，直接将公差化为小数点后一位的数量级
    str_data = [x.strip() for x in str_data if x.strip() != '']  # 将字符串列表中空项删除

    # print("str_data,存储一个data里面一行的所有文本里面的数字(字符串列表)\n", str_data)
    # list转化为字符串numpy数组，再转化为数字numpy数组
    str_data = np.asarray(str_data)
    str_data = str_data.astype(np.float_)
    #####################################
    # print("str_data（数字数组）\n", str_data)
    if len(str_data) == 2:  # 一个data里面的一行文本里面如果有两个数字则判断为公差形式，则判断第二个数字是公差，将公差数量级降为第一个数字的相匹配的数量级
        # 数字大于等于1时，公差为0.1型，数字为0.1型，公差为0.01型
        if str_data[0] >= 1:
            while str_data[1] >= 1:
                str_data[1] = str_data[1] * 0.1
        if 0.1 <= str_data[0] < 1 and str_data[1] >= 1:
            while str_data[1] >= 0.1:
                str_data[1] = str_data[1] * 0.1

    str_data = str_data.astype(str).tolist()  # 数字numpy数组转换为字符串numpy数组再转化为字符串list
    # print("str_data\n", str_data)

    if len(str_data) != 1:
        if len(str_data) == 2:
            str_data_another = str_data[1]
            str_data = str_data[0]

    str_data = ''.join(str_data)  # 将列表转换成字符串
    str_data_another = ''.join(str_data_another)
    # print("str_data（字符串）\n", str_data)
    # print("str_data_another（字符串）\n", str_data_another)
    return str_data, str_data_another


def get_np_array_in_txt(file_path):  # 提取txt中保存的数组，要求：浮点数且用逗号隔开
    """从文本文件中读取数组数据并转换为 numpy 格式。"""
    # import numpy as np
    with open(file_path) as f:
        line = f.readline()
        data_array = []
        while line:
            # num = list(map(int, line.split(',')))
            num = list(map(float, line.split(',')))
            data_array.append(num)
            line = f.readline()
        data_array = np.array(data_array)

    # print(data_array[0][:])
    # print('*' * 50)
    # print(data_array)
    return data_array


def get_path_in_txt(image_txt_path):  # 提取txt中保存的地址作为字符串输出
    """读取文本文件中的路径字符串。"""
    with open(image_txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def Square_or_Rectangular(top_data, bottom_data, side_data):  # 判断矩形长宽是否一样
    """根据顶部和底部数据判断封装是否为正方或长方形。"""
    key = -1  # 1是正方形，0是长方形
    down_top_data = top_data.sort(func=None, key=None, reverse=True)
    down_bottom_data = bottom_data.sort(func=None, key=None, reverse=True)
    print(down_bottom_data)
    down_side_data = side_data.sort(func=None, key=None, reverse=True)
    #####################################长宽同屏##############
    data = down_top_data
    if len(data) >= 2:
        if data[0] == data[1]:
            key = 1
    data = down_bottom_data
    if len(data) >= 2:
        if data[0] == data[1]:
            key = 1
    data = down_side_data
    if len(data) >= 2:
        if data[0] == data[1]:
            key = 1
    #######################################
    # if down_top_data[0] == down_bottom_data[0]:


def get_body_x_y(top_data):
    """从顶部检测结果中推算封装 X/Y 尺寸。"""
    限定长宽的范围2，40
    '''
    # 缺少的长或者宽按照最长尺寸线和尺寸数字规则寻找
    body_x = []
    body_y = []
    pairs_num = len(top_data)  # 总pairs数量
    if pairs_num == 0:
        print("在top view没有找到pairs和data")
        return body_x, body_y
    pairs_x_num = 0  # 横向pairs
    pairs_y_num = 0
    pairs_no_match_num = 0
    for i in range(pairs_num):
        if top_data[i][0] == 0:
            pairs_x_num += 1
        if top_data[i][0] == 1:
            pairs_y_num += 1
        if top_data[i][0] == 0.5:
            pairs_no_match_num += 1
    stand_x = 0
    stand_y = 0
    j = k = 0
    standard_ratio = 0.5
    if pairs_x_num != 0 and pairs_y_num != 0:
        for i in range(pairs_num):
            if (top_data[i][0] == 0) and top_data[i][1] > stand_x and top_data[i][1] < 30 and top_data[i][
                3] > 2:  # 经验值，长和宽少于30
                stand_x = top_data[i][1]
                j = i
            if (top_data[i][0] == 1) and top_data[i][1] > stand_y and top_data[i][1] < 30 and top_data[i][3] > 2:
                stand_y = top_data[i][1]
                k = i
        if stand_x == stand_y:
            print("通过两个相等pairs判断为正方形，长宽相等")
            body_x = top_data[j, 1:]
            body_y = top_data[k, 1:]
            return body_x, body_y
        else:
            if stand_x > stand_y:
                if stand_y / stand_x <= standard_ratio:
                    print("横pairs相比竖pairs过长，仅有横pairs为有用数据，因此判断为正方形")
                    body_x = top_data[j, 1:]
                    body_y = top_data[j, 1:]
                    return body_x, body_y
                if stand_y / stand_x > standard_ratio:
                    print("通过长度相近但是不等的横竖pairs判断为长方形")
                    body_x = top_data[j, 1:]
                    body_y = top_data[k, 1:]
                    return body_x, body_y
            if stand_x < stand_y:

                if stand_x / stand_y <= standard_ratio:
                    print("竖pairs相比横pairs过长，仅有竖pairs为有用数据，因此判断为正方形")
                    body_x = top_data[k, 1:]
                    body_y = top_data[k, 1:]
                    return body_x, body_y
                if stand_x / stand_y > standard_ratio:
                    print("通过长度相近但是不等的横竖pairs判断为长方形")
                    body_x = top_data[j, 1:]
                    body_y = top_data[k, 1:]
                    return body_x, body_y
    if pairs_x_num == 0 or pairs_y_num == 0:
        print("仅横或竖pairs有效，判断为正方形")
        stand_x = 0
        stand_y = 0
        j = k = 0
        if pairs_y_num == 0:
            for i in range(pairs_num):
                if top_data[i][0] == 0 and top_data[i][1] > stand_x and top_data[i][1] < 30 and top_data[i][3] > 2:
                    stand_x = top_data[i][1]
                    j = i
            body_x = top_data[j, 1:]
            body_y = top_data[j, 1:]
            return body_x, body_y
        if pairs_x_num == 0:
            for i in range(pairs_num):
                if top_data[i][0] == 1 and top_data[i][1] > stand_y and top_data[i][1] < 30 and top_data[i][3] > 2:
                    stand_y = top_data[i][1]
                    k = i
            body_x = top_data[k, 1:]
            body_y = top_data[k, 1:]
            return body_x, body_y


def get_pitch_x_y(bottom_data_np, pin_num_x, pin_num_y, body_x, body_y, bottom_ocr_data):  # 算出行和列的pitch值

    """依据 bottom 侧数据及引脚数量计算行列间距。"""
    ############################1.分别在行和列pairs的data中通过不等式：长/2 < 行pitch*（行pin数量-1）< 长 来初步筛选行pitch值（有力竞争者：pin值）
    ############################2.分别在行和列pairs的data中通过不等式：长/2 < 行pitch总长 < 长 来初步筛选 行pitch总长
    ############################3.通过两组筛选的数值进行严格匹配： 行pitch值*（行pin数量-1） = 行pitch总长 来输出行pitc和pitch总长这一个数值对
    print("***/开始检测pitch参数/***")
    print("pin_num_x,pin_num_y", pin_num_x, pin_num_y)

    pitch_x = np.empty((0,))
    pitch_y = np.empty((0,))
    for i in range(len(bottom_data_np)):
        if bottom_data_np[i][1] == bottom_data_np[i][2] == bottom_data_np[i][3]:
            if 0.4 <= bottom_data_np[i][2] < 2:
                if bottom_data_np[i][0] == 0:
                    # if body_x[1] / 3 < (bottom_data_np[i][2]) * (pin_num_x - 1) < body_x[1] and bottom_data_np[i][
                    #     2] * 1.5 not in pitch_x:
                    pitch_x = np.append(pitch_x, bottom_data_np[i][2])
                if bottom_data_np[i][0] == 1:
                    # if body_y[1] / 3 < (bottom_data_np[i][2]) * (pin_num_y - 1) < body_y[1] and bottom_data_np[i][
                    #     2] * 1.5 not in pitch_y:
                    pitch_y = np.append(pitch_y, bottom_data_np[i][2])
                # print(bottom_data_np[i][2])
    if len(pitch_x) == 0 and len(pitch_y) == 0:
        for i in range(len(bottom_data_np)):
            if bottom_data_np[i][1] == bottom_data_np[i][2] == bottom_data_np[i][3]:
                if 0.4 <= bottom_data_np[i][2] < 2:
                    if body_x[1] / 2 < (bottom_data_np[i][2]) * (pin_num_x - 1) < body_x[1] and bottom_data_np[i][
                        2] not in pitch_x:
                        pitch_x = np.append(pitch_x, bottom_data_np[i][2])
                    if body_y[1] / 2 < (bottom_data_np[i][2]) * (pin_num_y - 1) < body_y[1] and bottom_data_np[i][
                        2] not in pitch_y:
                        pitch_y = np.append(pitch_y, bottom_data_np[i][2])
    # print("pitch_x, pitch_y", pitch_x, pitch_y)
    print("不等式判断可能是pitch的值[行][列]：\n", pitch_x, pitch_y)
    right_key = 0
    pitch_x_true = np.empty((0,))
    toltal_pitch_x = np.empty((0,))
    pitch_y_true = np.empty((0,))
    toltal_pitch_y = np.empty((0,))
    pitch_true = np.empty((0,))
    toltal_pitch = np.empty((0,))
    for i in range(len(bottom_data_np)):
        if bottom_data_np[i][0] == 0 and len(pitch_x_true) != 1:
            if body_x[1] / 2 < bottom_data_np[i][2] < body_x[1]:
                for j in range(len(pitch_x)):
                    if math.isclose(pitch_x[j] * (pin_num_x - 1), bottom_data_np[i][2], rel_tol=1e-09, abs_tol=0.0):
                        print("找到行pitch与其匹配的行pitch总长：", pitch_x[j], " * (", pin_num_x, "- 1 ) =",
                              bottom_data_np[i][2])
                        # 标记为绝对正确的pitch
                        for x in range(len(bottom_ocr_data)):
                            if bottom_ocr_data[x]['max_medium_min'][0] == pitch_x[j] and \
                                    bottom_ocr_data[x]['max_medium_min'][1] == pitch_x[j] and \
                                    bottom_ocr_data[x]['max_medium_min'][2] == pitch_x[j]:
                                bottom_ocr_data[x]['Absolutely'] = 'pitch_x'
                                break
                        # 标记为绝对正确的pin行列数
                        bottom_ocr_data.append(
                            {'location': [], 'ocr_strings': [], 'key_info': [],
                             'matched_pairs_location': [], 'matched_pairs_outside_or_inside': [],
                             'matched_pairs_yinXian': [], 'Absolutely': 'pin_num_x',
                             'max_medium_min': [pin_num_x, pin_num_x, pin_num_x]})

                        right_key += 1
                        pitch_x_true = np.append(pitch_x_true, pitch_x[j])
                        toltal_pitch_x = np.append(toltal_pitch_x, bottom_data_np[i][2])
                        break
        if bottom_data_np[i][0] == 1 and len(pitch_y_true) != 1:
            if body_y[1] / 2 < bottom_data_np[i][2] < body_y[1]:
                for k in range(len(pitch_y)):
                    if math.isclose(pitch_y[k] * (pin_num_y - 1), bottom_data_np[i][2], rel_tol=1e-09, abs_tol=0.0):
                        print("找到列pitch与其匹配的列pitch总长", pitch_y[k], "*(", pin_num_y, "-1)=",
                              bottom_data_np[i][2])
                        # 标记为绝对正确的pitch
                        for x in range(len(bottom_ocr_data)):
                            if bottom_ocr_data[x]['max_medium_min'][0] == pitch_y[k] and \
                                    bottom_ocr_data[x]['max_medium_min'][1] == pitch_y[k] and \
                                    bottom_ocr_data[x]['max_medium_min'][2] == pitch_y[k]:
                                bottom_ocr_data[x]['Absolutely'] = 'pitch_y'
                                break
                        # 标记为绝对正确的pin行列数

                        bottom_ocr_data.append(
                            {'location': [], 'ocr_strings': [], 'key_info': [],
                             'matched_pairs_location': [], 'matched_pairs_outside_or_inside': [],
                             'matched_pairs_yinXian': [], 'Absolutely': 'pin_num_y',
                             'max_medium_min': [pin_num_y, pin_num_y, pin_num_y]})
                        right_key += 1
                        pitch_y_true = np.append(pitch_y_true, pitch_y[k])
                        toltal_pitch_y = np.append(toltal_pitch_y, bottom_data_np[i][2])
                        break

    if right_key == 0:  # 行和列分别匹配却匹配不到时，不分行列混在一起匹配pitch和总pitch
        for i in range(len(bottom_data_np)):
            # print(bottom_data_np[i][2])
            # if body_x[1] * 0.5 < bottom_data_np[i][2] < body_x[1]:
            for j in range(len(pitch_y)):
                if math.isclose(pitch_y[j] * (pin_num_y - 1), bottom_data_np[i][2], rel_tol=1e-09, abs_tol=0.0) and \
                        pitch_y[j] not in pitch_true:
                    print("找到列pitch与其匹配的行pitch总长：", pitch_y[j], " * (", pin_num_y, "- 1 ) =",
                          bottom_data_np[i][2])
                    # 标记为绝对正确的pitch
                    for x in range(len(bottom_ocr_data)):
                        if bottom_ocr_data[x]['max_medium_min'][0] == pitch_y[j] and \
                                bottom_ocr_data[x]['max_medium_min'][1] == pitch_y[j] and \
                                bottom_ocr_data[x]['max_medium_min'][2] == pitch_y[j]:
                            bottom_ocr_data[x]['Absolutely'] = 'pitch_y'
                            break
                    # 标记为绝对正确的pin行列数

                    bottom_ocr_data.append(
                        {'location': [], 'ocr_strings': [], 'key_info': [],
                         'matched_pairs_location': [], 'matched_pairs_outside_or_inside': [],
                         'matched_pairs_yinXian': [], 'Absolutely': 'pin_num_y',
                         'max_medium_min': [pin_num_y, pin_num_y, pin_num_y]})
                    pitch_true = np.append(pitch_true, pitch_y[j])
                    toltal_pitch = np.append(toltal_pitch, bottom_data_np[i][2])

            # if body_y[1] / 2 < bottom_data_np[i][2] < body_y[1]:
            for k in range(len(pitch_x)):
                # if pitch_x[k] * (pin_num_x - 1) == bottom_data_np[i][2]:
                if math.isclose(pitch_x[k] * (pin_num_x - 1), bottom_data_np[i][2], rel_tol=1e-09, abs_tol=0.0) and \
                        pitch_x[k] not in pitch_true:
                    print("找到行pitch与其匹配的列pitch总长：", pitch_x[k], " * (", pin_num_x, "- 1 ) =",
                          bottom_data_np[i][2])
                    # 标记为绝对正确的pitch
                    for x in range(len(bottom_ocr_data)):
                        if bottom_ocr_data[x]['max_medium_min'][0] == pitch_x[k] and \
                                bottom_ocr_data[x]['max_medium_min'][1] == pitch_x[k] and \
                                bottom_ocr_data[x]['max_medium_min'][2] == pitch_x[k]:
                            bottom_ocr_data[x]['Absolutely'] = 'pitch_x'
                            break
                    # 标记为绝对正确的pin行列数
                    bottom_ocr_data.append(
                        {'location': [], 'ocr_strings': [], 'key_info': [],
                         'matched_pairs_location': [], 'matched_pairs_outside_or_inside': [],
                         'matched_pairs_yinXian': [], 'Absolutely': 'pin_num_x',
                         'max_medium_min': [pin_num_x, pin_num_x, pin_num_x]})
                    pitch_true = np.append(pitch_true, pitch_x[k])
                    toltal_pitch = np.append(toltal_pitch, bottom_data_np[i][2])

        if len(pitch_true) == 0:
            pitch_true_y = pitch_true
            print(
                "没有满足不等式的数值对（pitch和总pitch长）")  # 很可能是行列pin数错误，可以通过bottom视图下的最长的两个data和其他data整除，并把倍数与pin数比较，如果相近则把pitch和行列pin数作为可能值输出
            print("可能行列pin数错误，尝试修正")
            ################################################################################
            key = 0
            ratio = 0.5  # 等式推断出新行列数对比原来的行列数占比 的上限
            for i in range(len(bottom_data_np)):
                if bottom_data_np[i][0] == 0:
                    for j in range(len(bottom_data_np)):
                        if body_x[2] > bottom_data_np[i][2] >= bottom_data_np[j][2] and bottom_data_np[j][
                            2] not in pitch_x_true:
                            # print(bottom_data_np[i][2],bottom_data_np[j][2],bottom_data_np[i][2]%bottom_data_np[j][2])
                            if (bottom_data_np[i][2] % bottom_data_np[j][2] == 0 or math.isclose(
                                    (bottom_data_np[i][2] % bottom_data_np[j][2]), bottom_data_np[j][2], rel_tol=1e-09,
                                    abs_tol=0.0)) and (abs(1 -
                                                           (bottom_data_np[i][2] / bottom_data_np[j][
                                                               2]) / pin_num_x)) < ratio and bottom_data_np[j][
                                2] not in pitch_x_true:
                                pin_num_x = round(bottom_data_np[i][2] / bottom_data_np[j][2]) + 1
                                print("找到疑似行pitch与其匹配的行pitch总长：", bottom_data_np[j][2], " * (", pin_num_x,
                                      "- 1 ) =",
                                      bottom_data_np[i][2])
                                # 标记为绝对正确的pitch
                                for x in range(len(bottom_ocr_data)):
                                    if bottom_ocr_data[x]['max_medium_min'][0] == bottom_data_np[j][2] and \
                                            bottom_ocr_data[x]['max_medium_min'][1] == bottom_data_np[j][2] and \
                                            bottom_ocr_data[x]['max_medium_min'][2] == bottom_data_np[j][2]:
                                        bottom_ocr_data[x]['Absolutely'] = 'pitch_x'
                                        break
                                # 标记为绝对正确的pin行列数

                                bottom_ocr_data.append(
                                    {'location': [], 'ocr_strings': [], 'key_info': [],
                                     'matched_pairs_location': [], 'matched_pairs_outside_or_inside': [],
                                     'matched_pairs_yinXian': [], 'Absolutely': 'pin_num_x',
                                     'max_medium_min': [pin_num_x, pin_num_x, pin_num_x]})
                                key += 1
                                pitch_x_true = np.append(pitch_x_true, bottom_data_np[j][2])

                if bottom_data_np[i][0] == 1:
                    for j in range(len(bottom_data_np)):
                        if body_y[2] > bottom_data_np[i][2] >= bottom_data_np[j][2] and bottom_data_np[i][
                            2] not in pitch_y_true:
                            if (bottom_data_np[i][2] % bottom_data_np[j][2] == 0 or math.isclose(
                                    (bottom_data_np[i][2] % bottom_data_np[j][2]), bottom_data_np[j][2], rel_tol=1e-09,
                                    abs_tol=0.0)) and (abs(1 -
                                                           (bottom_data_np[i][2] / bottom_data_np[j][
                                                               2]) / pin_num_y)) < ratio and bottom_data_np[j][
                                2] not in pitch_y_true:
                                pin_num_y = round(bottom_data_np[i][2] / bottom_data_np[j][2]) + 1
                                print("找到疑似列pitch与其匹配的列pitch总长：", bottom_data_np[j][2], " * (", pin_num_y,
                                      "- 1 ) =",
                                      bottom_data_np[i][2])
                                # 标记为绝对正确的pitch
                                for x in range(len(bottom_ocr_data)):
                                    if bottom_ocr_data[x]['max_medium_min'][0] == bottom_data_np[j][2] and \
                                            bottom_ocr_data[x]['max_medium_min'][1] == bottom_data_np[j][2] and \
                                            bottom_ocr_data[x]['max_medium_min'][2] == bottom_data_np[j][2]:
                                        bottom_ocr_data[x]['Absolutely'] = 'pitch_y'
                                        break
                                # 标记为绝对正确的pin行列数

                                bottom_ocr_data.append(
                                    {'location': [], 'ocr_strings': [], 'key_info': [],
                                     'matched_pairs_location': [], 'matched_pairs_outside_or_inside': [],
                                     'matched_pairs_yinXian': [], 'Absolutely': 'pin_num_y',
                                     'max_medium_min': [pin_num_y, pin_num_y, pin_num_y]})
                                key += 1
                                pitch_y_true = np.append(pitch_y_true, bottom_data_np[j][2])

            if key == 0 or key > 1:
                # if len(pitch_x_true) == 0:
                #     pin_num_x = pin_num_y
                # if len(pitch_y_true) == 0:
                #     pin_num_y = pin_num_x
                return pitch_x_true, pitch_y_true, pin_num_x, pin_num_y, bottom_ocr_data
            if key == 1:
                if len(pitch_x_true) != 0:
                    pitch_y_true = pitch_x_true
                    # pin_num_y = pin_num_x
                    return pitch_x_true, pitch_y_true, pin_num_x, pin_num_y, bottom_ocr_data
                if len(pitch_y_true) != 0:
                    pitch_x_true = pitch_y_true
                    # pin_num_x = pin_num_y
                    return pitch_x_true, pitch_y_true, pin_num_x, pin_num_y, bottom_ocr_data

            ################################################################################

            return pitch_x_true, pitch_y_true, pin_num_x, pin_num_y, bottom_ocr_data
        elif len(pitch_true) == 1:
            pitch_true_y = pitch_true
            return pitch_true, pitch_true_y, pin_num_x, pin_num_y, bottom_ocr_data
        else:
            pitch_true_y = pitch_true
            return pitch_true, pitch_true_y, pin_num_x, pin_num_y, bottom_ocr_data  ############存在问题：应该分地i更仔细

    if right_key >= 1:
        if len(pitch_x_true) != 0 and len(pitch_y_true) != 0:
            return pitch_x_true, pitch_y_true, pin_num_x, pin_num_y, bottom_ocr_data
        elif len(pitch_x_true) != 0:
            pitch_y_true = pitch_x_true
            return pitch_x_true, pitch_y_true, pin_num_x, pin_num_y, bottom_ocr_data
        else:
            pitch_x_true = pitch_y_true
            return pitch_x_true, pitch_y_true, pin_num_x, pin_num_y, bottom_ocr_data


def get_pitch_when_lone(bottom_data_np, pin_num_x, pin_num_y, body_x,
    """在行列缺失时根据已有数据估算间距。"""
                        body_y):  # 当用等式匹配pitch，把行列pin作为参考量重新匹配都失败后，判断应该是没有总pitch值，此时只能用不等式匹配然后输出可能的pitch值

    pitch_x = np.empty((0,))
    pitch_y = np.empty((0,))
    for i in range(len(bottom_data_np)):
        if bottom_data_np[i][1] == bottom_data_np[i][2] == bottom_data_np[i][3]:
            if 0.4 < bottom_data_np[i][2] < 2:
                if (bottom_data_np[i][0] == 0 or bottom_data_np[i][0] == 0.5) and bottom_data_np[i][2] not in pitch_x:
                    if body_x[1] / 2 < (bottom_data_np[i][2]) * (pin_num_x - 1) < body_x[1]:
                        pitch_x = np.append(pitch_x, bottom_data_np[i][2])
                if (bottom_data_np[i][0] == 1 or bottom_data_np[i][0] == 0.5) and bottom_data_np[i][2] not in pitch_y:
                    if body_y[1] / 2 < (bottom_data_np[i][2]) * (pin_num_y - 1) < body_y[1]:
                        pitch_y = np.append(pitch_y, bottom_data_np[i][2])
    if len(pitch_x) == 0 and len(pitch_y) == 0:
        for i in range(len(bottom_data_np)):
            if bottom_data_np[i][1] == bottom_data_np[i][2] == bottom_data_np[i][3]:
                if (bottom_data_np[i][0] == 0 or bottom_data_np[i][0] == 0.5) and bottom_data_np[i][2] not in pitch_x:
                    if 0.4 < bottom_data_np[i][2] < 2:
                        pitch_x = np.append(pitch_x, bottom_data_np[i][2])
                if (bottom_data_np[i][0] == 1 or bottom_data_np[i][0] == 0.5) and bottom_data_np[i][2] not in pitch_y:
                    if 0.4 < bottom_data_np[i][2] < 2:
                        pitch_y = np.append(pitch_y, bottom_data_np[i][2])
    if len(pitch_x) != 0 and len(pitch_y) != 0:
        print("判断不存在总pitch值的情况下，找到可能的pitch值", pitch_x, pitch_y)
    if len(pitch_x) != 0 and len(pitch_y) == 0:
        pitch_y = pitch_x
        print("判断不存在总pitch值的情况下，找到可能的pitch值", pitch_x, pitch_y)
    if len(pitch_x) == 0 and len(pitch_y) != 0:
        pitch_x = pitch_y
        print("判断不存在总pitch值的情况下，找到可能的pitch值", pitch_x, pitch_y)
    if len(pitch_x) == 0 and len(pitch_y) == 0:
        print("判断不存在总pitch值的情况下，没有找到可能的pitch值", pitch_x, pitch_y)
    # bottom中数量多的为正确率高的pitch
    if len(pitch_x) > 1:
        pitch_x_no = np.zeros((len(pitch_x)))
        for i in range(len(pitch_x)):
            if pitch_x[i] in bottom_data_np[:, 2]:
                pitch_x_no[i] += 1
        max_no = 0
        for i in range(len(pitch_x_no)):
            if max_no < pitch_x_no[i]:
                max_no = pitch_x_no[i]
        new_pitch_x = np.empty((0,))
        for i in range(len(pitch_x_no)):
            if pitch_x_no[i] == max_no:
                new_pitch_x = np.append(new_pitch_x, pitch_x[i])
        pitch_x = new_pitch_x
    if len(pitch_y) > 1:
        pitch_y_no = np.zeros((len(pitch_y)))
        for i in range(len(pitch_y)):
            if pitch_y[i] in bottom_data_np[:, 2]:
                pitch_y_no[i] += 1
        max_no = 0
        for i in range(len(pitch_y_no)):
            if max_no < pitch_y_no[i]:
                max_no = pitch_y_no[i]
        new_pitch_y = np.empty((0,))
        for i in range(len(pitch_y_no)):
            if pitch_y_no[i] == max_no:
                new_pitch_y = np.append(new_pitch_y, pitch_y[i])
        pitch_y = new_pitch_y

    return pitch_x, pitch_y


def get_pitch_x_y_when_last_plan(yolox_num_data, pin_num_x, pin_num_y, image_path):
    """在最后方案阶段以 YOLO 数字框估算行列间距。"""
    yolox_num_data = np.array(yolox_num_data)
    ocr = PaddleOCR(use_angle_cls=True,
                    lang="en",
                    det_model_dir="ppocr_model/det/en/en_PP-OCRv3_det_infer",
                    rec_model_dir='ppocr_model/rec/en/en_PP-OCRv3_rec_infer',
                    cls_model_dir='ppocr_model/cls/ch_ppocr_mobile_v2.0_cls_infer',
                    use_gpu=False)  # 禁用gpu

    with open(image_path, 'rb') as f:
        np_arr = np.frombuffer(f.read(), dtype=np.uint8)
        # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)#以彩图读取
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取
    # print(img)
    data_list = []  # 按序存储pairs的data
    no = 1
    for i in range(len(yolox_num_data)):
        box = np.array([[yolox_num_data[i][0], yolox_num_data[i][1]], [yolox_num_data[i][2], yolox_num_data[i][1]],
                        [yolox_num_data[i][2], yolox_num_data[i][3]], [yolox_num_data[i][0], yolox_num_data[i][3]]],
                       np.float32)

        box_img = get_rotate_crop_image(img, box)
        # box_img = cv2.resize(box_img, (320, 320))
        # cv2.imwrite('G:\PaddleOCR-release-2.7/1.jpg', box_img)#保存图片
        print("正在OCR识别第", no, "个bottom中的data")
        no += 1
        # cv2.imshow('cropped image', box_img)  # 显示当前ocr的识别区域
        cv2.waitKey(0)
        result = ocr.ocr(box_img, cls=True, det=True, rec=True)  # 问题：有可能yolox检测不到data，需要在yolox检测方面解决
        # 如果OCR输出NONE，旋转再识别
        if result == [None]:
            box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
            # cv2.imshow('cropped image', box_img)  # 显示当前ocr的识别区域
            # cv2.waitKey(0)
            result = ocr.ocr(box_img, cls=True, det=True, rec=True)
            if result == [None]:
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90
                # cv2.waitKey(0)
                result = ocr.ocr(box_img, cls=True, det=True, rec=True)
        if result == [None]:  # 如果识别不到，那么把图片放大
            box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
            box_img = cv2.resize(box_img, (320, 320))
            result = ocr.ocr(box_img, cls=True, det=True, rec=True)
        # 如果OCR输出NONE，旋转再识别
        if result == [None]:
            box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
            # cv2.imshow('cropped image', box_img)  # 显示当前ocr的识别区域
            # cv2.waitKey(0)
            result = ocr.ocr(box_img, cls=True, det=True, rec=True)
            if result == [None]:
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90
                # cv2.imshow('cropped image', box_img)  # 显示当前ocr的识别区域
                # cv2.waitKey(0)
                result = ocr.ocr(box_img, cls=True, det=True, rec=True)

        # print("OCR识别data的result", result)
        data_list_pairs = [0, ]  # [识别箭头方向,识别内容1，识别内容2，...]
        for i in range(len(result[0])):
            # result[0][i][1][0] = comma_inter_point((result[0][i][1][0]))
            if get_data_and_del_en(comma_inter_point(result[0][i][1][0])) != '':
                if get_data_and_del_en(comma_inter_point(result[0][i][1][0])) == []:
                    continue
                else:
                    data_list_pairs_one, data_list_pairs_another = get_data_and_del_en(
                        comma_inter_point(result[0][i][1][0]))
                    # print("len(data_list_pairs_one),data_list_pairs_one", len(data_list_pairs_one),
                    #       data_list_pairs_another)
                    # print("len(data_list_pairs_another),data_list_pairs_another", len(data_list_pairs_another),
                    #       data_list_pairs_another)
                    if len(data_list_pairs_another) == 0 and data_list_pairs_one != [''] and len(
                            data_list_pairs_one) != 0:
                        # data_list_pairs.append(get_data_and_del_en(comma_inter_point(result[0][i][1][0])))
                        data_list_pairs.append(data_list_pairs_one)
                        # print("data_list_pairs", data_list_pairs)
                    # if len(data_list_pairs_another) != 0:
                    if data_list_pairs_another != [''] and len(
                            data_list_pairs_another) != 0:  # 识别到类似"12+0.1"的文本时，get_data_and_del_en会返回12和0.1,之后判断大小来得到max，medium，min
                        data_list_pairs.append(data_list_pairs_one)
                        # print("data_list_pairs", data_list_pairs)
                        data_list_pairs.append(data_list_pairs_another)
                        # print("data_list_pairs", data_list_pairs)

                    # if data_list_pairs[1]
            # data_list_pairs.append(get_data_and_del_en(comma_inter_point(result[0][i][1][0] )))
        # result[0][0][0] = comma_inter_point(result[0][0][0])
        data_list.append(data_list_pairs)  # 将data的','替换为'.'，并仅提取数字，删除中英文按序保存
    # 对data_list中只存在min和max的值，以及只有一个median的，扩展为max，median和min
    # print("data_list", data_list)
    compire_ratio = 0.5
    data_list_np = np.empty((4, 0))
    for i in range(len(data_list)):
        m_m_m = np.asarray(data_list[i])
        m_m_m = m_m_m.astype(np.float_)
        if len(m_m_m) == 3:
            if m_m_m[1] > m_m_m[2] and m_m_m[2] / m_m_m[1] > compire_ratio:
                m_m_m = np.append(m_m_m, (m_m_m[1] + m_m_m[2]) / 2)
                min = m_m_m[2]
                m_m_m[2] = m_m_m[3]
                m_m_m[3] = min
            elif m_m_m[1] < m_m_m[2] and m_m_m[1] / m_m_m[2] > compire_ratio:
                max = m_m_m[2]
                m_m_m[2] = m_m_m[1]
                m_m_m[1] = max
                m_m_m = np.append(m_m_m, (m_m_m[1] + m_m_m[2]) / 2)
            elif m_m_m[1] > m_m_m[2] and m_m_m[2] / m_m_m[1] < compire_ratio:
                max = m_m_m[1] + m_m_m[2]
                min = m_m_m[1] - m_m_m[2]
                medium = m_m_m[1]
                m_m_m[1] = max
                m_m_m[2] = medium
                m_m_m = np.append(m_m_m, min)
            elif m_m_m[1] < m_m_m[2] and m_m_m[1] / m_m_m[2] < compire_ratio:
                max = m_m_m[2] + m_m_m[1]
                min = m_m_m[2] - m_m_m[1]
                medium = m_m_m[2]
                m_m_m[1] = max
                m_m_m[2] = medium
                m_m_m = np.append(m_m_m, min)
        if len(m_m_m) == 2:
            m_m_m = np.append(m_m_m, m_m_m[1])
            m_m_m = np.append(m_m_m, m_m_m[1])
        # data_list_np = np.append(data_list_np,m_m_m)
        # print(data_list_np)
        # print(m_m_m)
        data_list_np = np.c_[data_list_np, m_m_m]  # 添加行
    data_list_np = data_list_np.T

    # print("data_list_np", data_list_np)
    #############################################################ocr识别并输出所有的data数据，之后套用等式找到pitch值
    pitch_x_true = np.empty((0,))
    pitch_y_true = np.empty((0,))
    for i in range(len(data_list_np)):
        for j in range(len(data_list_np)):
            # if data_list_np[i][2] * (pin_num_x - 1) == data_list_np[j][2] and i != j :
            if data_list_np[i][2] * (pin_num_x - 1) == data_list_np[j][2]:
                print("找到行pitch与其匹配的行pitch总长：", data_list_np[i][2], " * (", pin_num_x, "- 1 ) =",
                      data_list_np[j][2])
                pitch_x_true = np.append(pitch_x_true, data_list_np[i][2])
                break

    for i in range(len(data_list_np)):
        for j in range(len(data_list_np)):

            # if data_list_np[i][2] * (pin_num_y - 1) == data_list_np[j][2] and i != j:
            if data_list_np[i][2] * (pin_num_y - 1) == data_list_np[j][2]:
                print("找到列pitch与其匹配的列pitch总长：", data_list_np[i][2], " * (", pin_num_y, "- 1 ) =",
                      data_list_np[j][2])
                pitch_y_true = np.append(pitch_y_true, data_list_np[i][2])
                break
    if len(pitch_x_true) == len(pitch_y_true) == 0:
        print("找不到pitch值，请检查图片中是否存在该数据")
        return pitch_x_true, pitch_y_true
    elif len(pitch_x_true) == 0:
        print("找到列pitch值没有找到行pitch值，判断二者相等")
        pitch_x_true = pitch_y_true
        return pitch_x_true, pitch_y_true
    elif len(pitch_y_true) == 0:
        print("找到行pitch值没有找到列pitch值，判断二者相等")
        pitch_y_true = pitch_x_true
        return pitch_x_true, pitch_y_true
    else:
        print("找到行pitch和列pitch")
        return pitch_x_true, pitch_y_true


def get_high_pin_high_max_1(side_data, body_x, body_y):
    """根据侧视图数据统计焊球高度范围。"""
    # print("side_data",side_data)
    high = np.zeros(3)
    if len(side_data) == 0:
        high = np.zeros(3)
        print("side视图未找到data")
        return high
    if len(side_data) != 0:

        high_max = 0
        key_no = 0

        for i in range(len(side_data)):
            if (side_data[i][1:4] < body_x).all() and (side_data[i][1:4] < body_y).all():
                if side_data[i][1] > high_max:
                    high_max = side_data[i][1]
                    high = side_data[i, 1:]
        if high[0] == high[1] == 0:
            for i in range(len(side_data)):
                if (side_data[i][1:4] < body_x).all() and (side_data[i][1:4] < body_y).all():
                    if side_data[i][1] > high_max:
                        high_max = side_data[i][1]
                        high = side_data[i, 1:]

    return high


def get_high_pin_high_max(side_data):
    """整合侧视图信息估算封装高度及最大值。"""
    # print("side_data",side_data)
    high = np.zeros(3)
    if len(side_data) == 0:
        high = np.zeros(3)
        print("side视图未找到data")
        return high
    if len(side_data) != 0:

        high_max = 0
        key_no = 0

        for i in range(len(side_data)):
            # if side_data[i][0] != 0.5:
            if side_data[i][1] > high_max:
                high_max = side_data[i][1]
                high = side_data[i, 1:]
        if high[0] == high[1] == 0:
            for i in range(len(side_data)):
                # if side_data[i][0] != 0.5:
                if side_data[i][1] > high_max:
                    high_max = side_data[i][1]
                    high = side_data[i, 1:]

    return high


def get_pin_diameter(pitch_x, pitch_y, pin_x_number, pin_y_number, body_x, body_y, bottom_data_list_np,
    """综合三视图数据估算 PIN 直径。"""
                     side_data_list_np,
                     top_data_list_np):  # 从三视图中找pin直径，方法是看data的最大值和行列数减一相乘是否小于长和宽，最小值和行列数减一是否大于长和宽的一半
    pin_diameter = np.zeros((1, 3))  # 存储可能的pin直径值，
    # print(bottom_data_list_np,pin_diameter,bottom_data_list_np[1][1:4])
    for i in range(len(bottom_data_list_np)):

        if (bottom_data_list_np[i][1] < pitch_x).any() and (
                bottom_data_list_np[i][1] < pitch_y).any():  # 如果data最大值比pitch值小
            if bottom_data_list_np[i][1] * pin_x_number < body_x[2] and bottom_data_list_np[i][1] * pin_y_number < \
                    body_y[2]:
                if bottom_data_list_np[i][1] * (pin_x_number) > body_x[0] * 0.5 and bottom_data_list_np[i][1] * (
                        pin_y_number) > body_y[0] * 0.5 and bottom_data_list_np[i, 1:3] not in pin_diameter:
                    pin_diameter = np.row_stack((pin_diameter, bottom_data_list_np[i, 1:4]))
                    # print(pin_diameter)
    for i in range(len(side_data_list_np)):
        if (side_data_list_np[i][1] < pitch_x).any() and (side_data_list_np[i][1] < pitch_y).any():
            if side_data_list_np[i][1] * pin_x_number < body_x[2] and side_data_list_np[i][1] * pin_y_number < body_y[
                2]:
                if side_data_list_np[i][1] * (pin_x_number) > body_x[0] * 0.5 and side_data_list_np[i][1] * (
                        pin_y_number) > body_y[0] * 0.5 and side_data_list_np[i, 1:3] not in pin_diameter:
                    pin_diameter = np.row_stack((pin_diameter, side_data_list_np[i, 1:4]))
    for i in range(len(top_data_list_np)):
        if (top_data_list_np[i][1] < pitch_x).any() and (top_data_list_np[i][1] < pitch_y).any():
            if top_data_list_np[i][1] * pin_x_number < body_x[2] and top_data_list_np[i][1] * pin_y_number < body_y[2]:
                if top_data_list_np[i][1] * (pin_x_number) > body_x[0] * 0.5 and top_data_list_np[i][1] * (
                        pin_y_number) > body_y[0] * 0.5 and top_data_list_np[i, 1:3] not in pin_diameter:
                    pin_diameter = np.row_stack((pin_diameter, top_data_list_np[i, 1:4]))
    pin_diameter = pin_diameter[1:]
    # 洗去重复项
    if len(pin_diameter) > 1:
        pin_diameter_2 = np.zeros((0, 3))
        for i in range(len(pin_diameter)):
            if pin_diameter[i] not in pin_diameter_2:
                pin_diameter_2 = np.r_[pin_diameter_2, [pin_diameter[i]]]
        pin_diameter = pin_diameter_2
    # 进一步筛选0.45 - 0.85
    pin_diameter_3 = np.zeros((0, 3))
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if ((pin_diameter[i] / pitch_x[0]) > 0.85).any() or ((pin_diameter[i] / pitch_x[0]) < 0.45).any():
                print("filter")
            else:
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        pin_diameter = pin_diameter_3
    # 进一步筛选0.45 - 0.8
    pin_diameter_3 = np.zeros((0, 3))
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if ((pin_diameter[i] / pitch_x[0]) > 0.80).any() or ((pin_diameter[i] / pitch_x[0]) < 0.45).any():
                print("filter")
            else:
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        if len(pin_diameter_3) != 0 and len(pin_diameter_3) != len(pin_diameter):
            pin_diameter = pin_diameter_3
    # 进一步筛选0.45 - 0.75
    pin_diameter_3 = np.zeros((0, 3))
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if ((pin_diameter[i] / pitch_x[0]) >= 0.75).any() or ((pin_diameter[i] / pitch_x[0]) < 0.45).any():
                print("filter")
            else:
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        if len(pin_diameter_3) != 0 and len(pin_diameter_3) != len(pin_diameter):
            pin_diameter = pin_diameter_3
    # 进一步筛选如果多个尺寸数字中仅有一个含误差的，确定含误差为直径
    pin_diameter_3 = np.zeros((0, 3))
    j = 0
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if pin_diameter[i][0] == pin_diameter[i][1] == pin_diameter[i][2]:
                print("filter")
            else:
                j += 1
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        if j == 1:
            pin_diameter = pin_diameter_3

    return pin_diameter  # numpy二维数组，可能不止一行


def pin_diameter_1(pitch_x, pitch_y, pin_x_number, pin_y_number, body_x, body_y, bottom_data_list_np, side_data_list_np,
    """以多视图信息补充 PIN 直径的冗余计算。"""
                   top_data_list_np, standoff, high):  # 从三视图中找pin直径，方法是看data的最大值和行列数减一相乘是否小于长和宽，最小值和行列数减一是否大于长和宽的一半
    pin_diameter = np.zeros((1, 3))  # 存储可能的pin直径值，
    # print(bottom_data_list_np,pin_diameter,bottom_data_list_np[1][1:4])
    for i in range(len(bottom_data_list_np)):

        if (bottom_data_list_np[i][1] < pitch_x).any() and (bottom_data_list_np[i][1] < pitch_y).any() and (
                bottom_data_list_np[i][-3:] > standoff).all():  # 如果data最大值比pitch值小
            if bottom_data_list_np[i][1] * pin_x_number < body_x[2] and bottom_data_list_np[i][1] * pin_y_number < \
                    body_y[2]:
                if bottom_data_list_np[i][1] * pin_x_number > body_x[0] * 0.5 and bottom_data_list_np[i][
                    1] * pin_y_number > body_y[0] * 0.5 and bottom_data_list_np[i,
                                                            1:3] not in pin_diameter and bottom_data_list_np[i,
                                                                                         1:3] not in high:
                    pin_diameter = np.row_stack((pin_diameter, bottom_data_list_np[i, 1:4]))
                    # print(pin_diameter)
    for i in range(len(side_data_list_np)):
        if (side_data_list_np[i][1] < pitch_x).any() and (side_data_list_np[i][1] < pitch_y).any() and (
                side_data_list_np[i][-3:] > standoff).all():
            if side_data_list_np[i][1] * pin_x_number < body_x[2] and side_data_list_np[i][1] * pin_y_number < body_y[
                2]:
                if side_data_list_np[i][1] * pin_x_number > body_x[0] * 0.5 and side_data_list_np[i][1] * pin_y_number > \
                        body_y[0] * 0.5 and side_data_list_np[i, 1:3] not in pin_diameter and bottom_data_list_np[i,
                                                                                              1:3] not in high:
                    pin_diameter = np.row_stack((pin_diameter, side_data_list_np[i, 1:4]))
    for i in range(len(top_data_list_np)):
        if (top_data_list_np[i][1] < pitch_x).any() and (top_data_list_np[i][1] < pitch_y).any() and (
                top_data_list_np[i][-3:] > standoff).all():
            if top_data_list_np[i][1] * pin_x_number < body_x[2] and top_data_list_np[i][1] * pin_y_number < body_y[2]:
                if top_data_list_np[i][1] * pin_x_number > body_x[0] * 0.5 and top_data_list_np[i][1] * pin_y_number > \
                        body_y[0] * 0.5 and top_data_list_np[i, 1:3] not in pin_diameter and bottom_data_list_np[i,
                                                                                             1:3] not in high:
                    pin_diameter = np.row_stack((pin_diameter, top_data_list_np[i, 1:4]))
    pin_diameter = pin_diameter[1:]
    # 洗去重复项
    if len(pin_diameter) > 1:
        pin_diameter_2 = np.zeros((0, 3))
        for i in range(len(pin_diameter)):
            if pin_diameter[i] not in pin_diameter_2:
                pin_diameter_2 = np.r_[pin_diameter_2, [pin_diameter[i]]]
        pin_diameter = pin_diameter_2
    # 进一步筛选0.45 - 0.85
    pin_diameter_3 = np.zeros((0, 3))
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if ((pin_diameter[i] / pitch_x[0]) > 0.85).any() or ((pin_diameter[i] / pitch_x[0]) < 0.45).any():
                print("filter")
            else:
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        pin_diameter = pin_diameter_3
    # 进一步筛选0.45 - 0.8
    pin_diameter_3 = np.zeros((0, 3))
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if ((pin_diameter[i] / pitch_x[0]) > 0.80).any() or ((pin_diameter[i] / pitch_x[0]) < 0.45).any():
                print("filter")
            else:
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        if len(pin_diameter_3) != 0 and len(pin_diameter_3) != len(pin_diameter):
            pin_diameter = pin_diameter_3
    # 进一步筛选0.45 - 0.75
    pin_diameter_3 = np.zeros((0, 3))
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if ((pin_diameter[i] / pitch_x[0]) >= 0.75).any() or ((pin_diameter[i] / pitch_x[0]) < 0.45).any():
                print("filter")
            else:
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        if len(pin_diameter_3) != 0 and len(pin_diameter_3) != len(pin_diameter):
            pin_diameter = pin_diameter_3
    # 进一步筛选如果多个尺寸数字中仅有一个含误差的，确定含误差为直径
    pin_diameter_3 = np.zeros((0, 3))
    j = 0
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if pin_diameter[i][0] == pin_diameter[i][1] == pin_diameter[i][2]:
                print("filter")
            else:
                j += 1
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        if j == 1:
            pin_diameter = pin_diameter_3

    return pin_diameter  # numpy二维数组，可能不止一行


def find_standoff(side_data_np, pin_diagonal, high):
    """通过侧视图数据估计 standoff 高度。"""
    支撑高的max_medium_min的每一个数都不超过pin_diameter，且支撑高一定不等于pin_diameter
    '''
    standoff = np.zeros((0, 3))
    for j in range(len(pin_diagonal)):
        for i in range(len(side_data_np)):
            if (pin_diagonal[j] >= side_data_np[i, 1:4]).all() and (pin_diagonal[j] != side_data_np[i, 1:4]).any() and (
                    pin_diagonal[j] * 0.5 <= side_data_np[i, 1:4]).all() and side_data_np[i,
                                                                             1:4] not in standoff and side_data_np[i,
                                                                                                      1:4] not in pin_diagonal and (
                    (0.1 <= side_data_np[i, 1:4]) & (side_data_np[i, 1:4] < 0.65)).all():
                standoff = np.row_stack((standoff, side_data_np[i, 1:4]))
    print("standoff", standoff)

    if len(standoff) == 0:
        standoff_1 = np.zeros((0, 3))  # 存储可能的pin直径值，

        for i in range(len(side_data_np)):
            if (side_data_np[i][1] < pin_diagonal).any() and (
                    side_data_np[i][1] < pin_diagonal).any() and side_data_np[i, 1:4] not in high:  # 如果data最大值比pitch值小
                standoff_1 = np.r_[standoff_1, [side_data_np[i, 1:4]]]

        standoff_1 = standoff_1[np.argsort(-standoff_1[:, 0])]
        if len(standoff_1) != 0:
            standoff = np.r_[standoff, [standoff_1[0]]]

    return standoff


def select_best_stanoff(standoff):
    """在多个 standoff 候选中挑选最优值。"""
    当standoff不止一个，挑选最合适的
    '''
    try:
        if standoff.ndim == 2:
            if len(standoff) > 1:
                new_standoff = np.zeros((0, 3))
                acc_standoff = []  # 记录每一组standoff的置信度
                for i in range(len(standoff)):
                    acc_standoff.append(0)
                    if standoff[i][0] != standoff[i][1] or standoff[i][1] != standoff[i][2]:
                        acc_standoff[i] += 1
                    if standoff[i][0] == standoff[i][1] == standoff[i][2] == 0:
                        acc_standoff[i] = -1
                max_value = max(acc_standoff)  # 求列表最大值
                max_idx = acc_standoff.index(max_value)  # 求最大值对应索引
                new_standoff = np.r_[new_standoff, [standoff[max_idx]]]
                standoff = new_standoff
    except:
        pass
    return standoff


def get_pitch_x_y_when_absence_pin(bottom_data_np, pin_num_x, pin_num_y):  # 先不引入长和宽
    """在缺失引脚的情况下估算行列间距。"""
    if pin_num_x == 0:  # 缺整列pin时，输出列pitch值是准确的

        pitch_y_true = np.empty((0,))
        toltal_pitch_y = np.empty((0,))

        for i in range(len(bottom_data_np)):  # 找列pitch值，可以借助等式

            if bottom_data_np[i][0] == 1:

                for k in range(len(bottom_data_np)):

                    # if k != i and bottom_data_np[k][0] == 1:
                    if bottom_data_np[k][0] == 1:  # 照顾到pin_num = 2的情况，k可以等于i

                        if bottom_data_np[k][1] * (pin_num_y - 1) == bottom_data_np[i][2]:  # pitch值肯定没有公差因此用最大值

                            print("找到列pitch与其匹配的列pitch总长", bottom_data_np[k][1], "*(", pin_num_y, "-1)=",
                                  bottom_data_np[i][2])

                            pitch_y_true = np.append(pitch_y_true, bottom_data_np[k][1])
                            toltal_pitch_y = np.append(toltal_pitch_y, bottom_data_np[i][2])
                            break

        # 将横向箭头的数据的最大值按从大到小排序并取最大的两个进行循环配对
        pairs_x_max = np.empty((0,))
        for i in range(2):
            if bottom_data_np[i][0] == 0:
                pairs_x_max = np.append(pairs_x_max, bottom_data_np[i][1])

        pairs_x_max.sort()
        pairs_x_max = abs(np.sort(-pairs_x_max))  # 先取相反数排序，再加上绝对值得到原数组的降序
        # print(pairs_x_max)
        pitch_x_true = np.empty((0,))
        toltal_pitch_x = np.empty((0,))
        for i in range(len(pairs_x_max)):  # 找行pitch值，只能看有没有数值对可以完成整除操作，且大的数值只能取横向数据的最大的两个值

            for k in range(len(bottom_data_np)):

                if bottom_data_np[k][0] == 0 and bottom_data_np[k][0] != pairs_x_max[i]:

                    if pairs_x_max[i] > bottom_data_np[k][1] and pairs_x_max[i] % bottom_data_np[k][
                        1] == 0:  # pitch值肯定没有公差因此用最大值
                        pin_num_x = int((pairs_x_max[i] / bottom_data_np[k][1]) + 1)
                        print("找到行pitch与其匹配的行pitch总长", bottom_data_np[k][1], "*(", pin_num_x, "-1)=",
                              pairs_x_max[i])

                        pitch_x_true = np.append(pitch_x_true, bottom_data_np[k][1])
                        toltal_pitch_x = np.append(toltal_pitch_x, pairs_x_max[i])
                        break

    if pin_num_y == 0:
        pitch_x_true = np.empty((0,))
        toltal_pitch_x = np.empty((0,))

        for i in range(len(bottom_data_np)):  # 找行pitch值，可以借助等式

            if bottom_data_np[i][0] == 0:

                for k in range(len(bottom_data_np)):

                    if k != i and bottom_data_np[k][0] == 0:

                        if bottom_data_np[k][1] * (pin_num_x - 1) == bottom_data_np[i][2]:  # pitch值肯定没有公差因此用最大值

                            print("找到行pitch与其匹配的行pitch总长", bottom_data_np[k][1], "*(", pin_num_x, "-1)=",
                                  bottom_data_np[i][2])

                            pitch_x_true = np.append(pitch_x_true, bottom_data_np[k][1])
                            toltal_pitch_x = np.append(toltal_pitch_x, bottom_data_np[i][2])
                            break

        # 将竖向箭头的数据的最大值按从大到小排序并取最大的两个进行循环配对
        pairs_y_max = np.empty((0,))
        for i in range(2):
            if bottom_data_np[i][0] == 1:
                pairs_y_max = np.append(pairs_y_max, bottom_data_np[i][1])

        pairs_y_max.sort()
        pairs_y_max = abs(np.sort(-pairs_y_max))  # 先取相反数排序，再加上绝对值得到原数组的降序
        # print(pairs_y_max)
        pitch_y_true = np.empty((0,))
        toltal_pitch_y = np.empty((0,))
        for i in range(len(pairs_y_max)):  # 找列pitch值，只能看有没有数值对可以完成整除操作，且大的数值只能取竖向数据的最大的两个值

            for k in range(len(bottom_data_np)):

                if bottom_data_np[k][0] == 1 and bottom_data_np[k][0] != pairs_y_max[i]:

                    if pairs_y_max[i] > bottom_data_np[k][1] and pairs_y_max[i] % bottom_data_np[k][
                        1] == 1:  # pitch值肯定没有公差因此用最大值
                        pin_num_y = int((pairs_y_max[i] / bottom_data_np[k][1]) + 1)
                        print("找到列pitch与其匹配的列pitch总长", bottom_data_np[k][1], "*(", pin_num_y, "-1)=",
                              pairs_y_max[i])

                        pitch_y_true = np.append(pitch_y_true, bottom_data_np[k][1])
                        toltal_pitch_y = np.append(toltal_pitch_y, pairs_y_max[i])
                        break
    return pitch_x_true, pitch_y_true, pin_num_x, pin_num_y


def show_lost_pin(pin, pin_set, average_x_pitch, average_y_pitch, key, pin_num_x, pin_num_y):  # 先不尝试修正零散ball的影响
    """可视化缺失焊球位置并统计对应行列。"""
    try:
        # 由ball的行列最大数量建立矩阵，1表示该位置有ball，0表示没有
        # 找到pin中心位置的x和y坐标最小值作为基础x轴和y轴，其他pin中心偏离轴多少个pitch值就算偏离几个基本单位
        pin_map = np.zeros((pin_num_y, pin_num_x), dtype=int)
        min_x = 9999999
        min_y = 9999999
        if key == 1:  # 缺整列
            for i in range(len(pin)):
                if (pin[i][0] + pin[i][2]) * 0.5 < min_x:
                    min_x = (pin[i][0] + pin[i][2]) * 0.5
                if (pin[i][1] + pin[i][3]) * 0.5 < min_y:
                    min_y = (pin[i][1] + pin[i][3]) * 0.5
        standard_ratio = 0.5  # 能接受的偏离程度
        for i in range(len(pin)):
            x = y = -1
            test_x = round(((pin[i][0] + pin[i][2]) * 0.5 - min_x) / average_x_pitch)
            test_y = round(((pin[i][1] + pin[i][3]) * 0.5 - min_y) / average_y_pitch)
            if abs((pin[i][0] + pin[i][2]) * 0.5 - min_x - test_x * average_x_pitch) / average_x_pitch < standard_ratio:
                x = round(((pin[i][0] + pin[i][2]) * 0.5 - min_x) / average_x_pitch)
            if abs((pin[i][1] + pin[i][3]) * 0.5 - min_y - test_y * average_y_pitch) / average_y_pitch < standard_ratio:
                y = round(((pin[i][1] + pin[i][3]) * 0.5 - min_y) / average_y_pitch)
            if x != -1 and y != -1:
                pin_map[y][x] = 1
        return pin_map
    except Exception as e:
        print('错误类型是', e.__class__.__name__)
        print('错误明细是', e)


def show_lost_pin_when_full(pin, pin_num_x, pin_num_y, average_x_pitch, average_y_pitch):
    """在满阵列情况下标注缺失焊球。"""
    # try:
    pin_map = np.zeros((int(pin_num_y), int(pin_num_x))).astype(int)
    min_x = 9999999
    min_y = 9999999
    for i in range(len(pin)):
        if (pin[i][0] + pin[i][2]) * 0.5 < min_x:
            min_x = (pin[i][0] + pin[i][2]) * 0.5
        if (pin[i][1] + pin[i][3]) * 0.5 < min_y:
            min_y = (pin[i][1] + pin[i][3]) * 0.5

    standard_ratio = 0.5  # 能接受的偏离程度
    for i in range(len(pin)):
        x = y = -1
        test_x = round(((pin[i][0] + pin[i][2]) * 0.5 - min_x) / average_x_pitch)
        test_y = round(((pin[i][1] + pin[i][3]) * 0.5 - min_y) / average_y_pitch)
        if abs((pin[i][0] + pin[i][2]) * 0.5 - min_x - test_x * average_x_pitch) / average_x_pitch < standard_ratio:
            x = round(((pin[i][0] + pin[i][2]) * 0.5 - min_x) / average_x_pitch)
        if abs((pin[i][1] + pin[i][3]) * 0.5 - min_y - test_y * average_y_pitch) / average_y_pitch < standard_ratio:
            y = round(((pin[i][1] + pin[i][3]) * 0.5 - min_y) / average_y_pitch)
        if x != -1 and y != -1 and x < pin_num_x and y < pin_num_y:
            pin_map[y][x] = 1
        # print(pin_map)#问题：可能存在累计误差
        # print("################输出bottom视图###############")
        # print("pin存在显示'o',不存在以位置信息代替")
        #
        # print('   ', end='')
        # for i in range(len(pin_map[0])):
        #     print(i + 1, end='  ')
        # print()
        #
        # for i in range(len(pin_map)):
        #     print(chr(65 + i), end='  ')
        #     for j in range(len(pin_map[i])):
        #         if pin_map[i][j] == 1:
        #             print("o", end='  ')
        #         if pin_map[i][j] == 0:
        #             print(chr(65 + i), j + 1, end=' ', sep='')
        #     print()
    return pin_map
    # except Exception as e:
    #     print('错误类型是', e.__class__.__name__)
    #     print('错误明细是', e)


###################################图像增强

def gamma(img, out):
    """执行 gamma 校正增强图像对比度。"""
    # img = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
    # 归1
    Cimg = img / 255
    # 伽玛变换
    gamma = 0.7
    O = np.power(Cimg, gamma)
    O = O * 255
    # 效果
    cv2.imwrite(out, O, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def hist(img, show_img_key):
    """绘制灰度直方图并根据配置决定是否展示。"""
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
    """自动计算直方图阈值进行图像增强。"""
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
    """手动计算灰度直方图数据。"""
    # 计算灰度直方图
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist


def equalHist(img):
    """对图像进行直方图均衡化。"""
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
    """执行线性灰度变换增强图像。"""
    # img = cv2.imread(source, 0)
    # 使用自己写的函数实现
    equa = equalHist(img)
    cv2.imshow("equa", equa)
    cv2.imwrite('temp.png', equa, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.waitKey()


def correct_top_data(data_list_np):
    """修正顶部视图的检测框坐标。"""
    if len(data_list_np) == 0:
        new_data_list_np = data_list_np
    else:
        # 1.如果data大于100，判断为漏掉小数点，一直除10直到小于100
        for i in range(len(data_list_np)):
            while data_list_np[i][2] >= 100:
                data_list_np[i, 2:] = data_list_np[i, 2:] / 10

        print("纠正之后的top_data\n", data_list_np)
        # 2.滤除大于25的没公差的整数
        # 3.筛选出整数且没有误差的data：
        count_array = np.ones((len(data_list_np)))
        new_data_list_np = np.zeros((0, len(data_list_np[0])))
        for i in range(len(data_list_np)):
            if not np.any(data_list_np[i][len(data_list_np[i]) - 3:]):  # 将没检测出数字的筛出
                count_array[i] = 0
                continue
            if np.any(data_list_np[i][len(data_list_np[i]) - 3:] - data_list_np[i][len(data_list_np[i]) - 3].astype(
                    int)):  # 将非整数筛出
                count_array[i] = 0
                continue
            if (not math.isclose(data_list_np[i][len(data_list_np[i]) - 3],
                                 data_list_np[i][len(data_list_np[i]) - 2])) or (
                    not math.isclose(data_list_np[i][len(data_list_np[i]) - 2],
                                     data_list_np[i][len(data_list_np[i]) - 1])):  # 将有误差的数筛出
                count_array[i] = 0
        # 4.筛除大于25的整数
        for i in range(len(data_list_np)):
            if count_array[i] == 1:
                if data_list_np[i][2] >= 25:
                    count_array[i] = 2
        # 5.组合新的data
        for i in range(len(count_array)):
            if count_array[i] != 2:
                new_data_list_np = np.r_[new_data_list_np, [data_list_np[i]]]

    return new_data_list_np


####################################
def correct_bottom_side_data(top_data_list_np, bottom_data_list_np):
    """用顶部数据修正底部和侧面框体。"""
    key = 0
    if len(top_data_list_np) == 0:
        print("top图中找不到数据")
    else:
        correct_ratio_1 = 0.5
        correct_ratio_10 = 0.15
        length = 0
        length_no = -1
        for i in range(len(top_data_list_np)):
            if top_data_list_np[i][2] > length:
                length = top_data_list_np[i][2]
                length_no = i
        for i in range(len(bottom_data_list_np)):
            if bottom_data_list_np[i][2] >= 100 and bottom_data_list_np[i][0] != 0.5:
                print("找到异常数据，尝试修正")
                key += 1
                if correct_ratio_1 < bottom_data_list_np[i][1] / top_data_list_np[length_no][1] < 1:  # 数量级相等
                    bottom_data_list_np[i, 2:] = bottom_data_list_np[i, 2:] / 10
                if correct_ratio_10 < bottom_data_list_np[i][1] / top_data_list_np[length_no][
                    1] < correct_ratio_1:  # 数量级差一位
                    bottom_data_list_np[i, 2:] = bottom_data_list_np[i, 2:] / 100
                if bottom_data_list_np[i][1] / top_data_list_np[length_no][1] < correct_ratio_10:  # 数量级差两位
                    bottom_data_list_np[i, 2:] = bottom_data_list_np[i, 2:] / 1000
    if key != 0:
        print("修复了", key, "处异常数据")
        print("修复后的数据\n", bottom_data_list_np)
    # 删除大于20的没公差的整数
    count_array = np.ones((len(bottom_data_list_np)))
    new_data_list_np = np.zeros((0, 5))
    for i in range(len(bottom_data_list_np)):
        if not np.any(bottom_data_list_np[i][len(bottom_data_list_np[i]) - 3:]):  # 将没检测出数字的筛出
            count_array[i] = 0
            continue
        if np.any(bottom_data_list_np[i][len(bottom_data_list_np[i]) - 3:] - bottom_data_list_np[i][
            len(bottom_data_list_np[i]) - 3].astype(
            int)):  # 将非整数筛出
            count_array[i] = 0
            continue
        if (not math.isclose(bottom_data_list_np[i][len(bottom_data_list_np[i]) - 3],
                             bottom_data_list_np[i][len(bottom_data_list_np[i]) - 2])) or (
                not math.isclose(bottom_data_list_np[i][len(bottom_data_list_np[i]) - 2],
                                 bottom_data_list_np[i][len(bottom_data_list_np[i]) - 1])):  # 将有误差的数筛出
            count_array[i] = 0
    # 4.筛除大于20的整数
    for i in range(len(bottom_data_list_np)):
        if count_array[i] == 1:
            if bottom_data_list_np[i][2] >= 20:
                count_array[i] = 2
    # 5.组合新的data
    for i in range(len(count_array)):
        if count_array[i] != 2:
            new_data_list_np = np.r_[new_data_list_np, [bottom_data_list_np[i]]]
    return new_data_list_np


def compare_top_bottom(top_data_list_np, bottom_data_list_np):
    """比较顶底视图的矩形框并做筛选。"""
    new_top = np.zeros((0, 4))
    for i in range(len(top_data_list_np)):
        if top_data_list_np[i][0] != 0.5 and top_data_list_np[i][1] < 40:  # 经验值，长和宽少于30
            new_top = np.r_[new_top, [top_data_list_np[i]]]
    new_bottom = np.zeros((0, 4))
    for i in range(len(bottom_data_list_np)):
        if bottom_data_list_np[i][0] != 0.5 and bottom_data_list_np[i][1] < 40:  # 经验值，长和宽少于30
            new_bottom = np.r_[new_bottom, [bottom_data_list_np[i]]]
    try:
        top_max = np.max(new_top[:, 1])
    except:
        top_max = 0
    try:
        bottom_max = np.max(new_bottom[:, 1])
    except:
        bottom_max = 0

    if bottom_max <= top_max:
        return True
    if bottom_max > top_max:
        return False


def compare_top_pairs_data_bottom_pairs_data(top_data_list_np, bottom_data_list_np):
    """对比顶底标尺框匹配度并进行过滤。"""
    new_top = np.zeros((0, len(top_data_list_np[0])))
    for i in range(len(top_data_list_np)):
        if top_data_list_np[i][0] != 0.5 and top_data_list_np[i][1] < 40:  # 经验值，长和宽少于30
            new_top = np.r_[new_top, [top_data_list_np[i]]]
    new_bottom = np.zeros((0, len(bottom_data_list_np[0])))
    for i in range(len(bottom_data_list_np)):
        if bottom_data_list_np[i][0] != 0.5 and bottom_data_list_np[i][1] < 40:  # 经验值，长和宽少于30
            new_bottom = np.r_[new_bottom, [bottom_data_list_np[i]]]
    try:
        top_max = np.max(new_top[:, 1])
    except:
        top_max = 0
    try:
        bottom_max = np.max(new_bottom[:, 1])
    except:
        bottom_max = 0

    if bottom_max <= top_max:
        return True
    if bottom_max > top_max:
        return False


def filt_hanglie(bottom_data_np):
    """将 bottom 侧的行列 OCR 结果进行筛选。"""
    # 1.yolox检测pinmap的坐标
    # from output_pinmap_location import begain_output_pinmap_location
    # pinmap = begain_output_pinmap_location()
    # 2.将bottom中的没公差的整数提取出来并排序
    count_array = np.ones((len(bottom_data_np)))
    new_bottom_data_np = np.zeros((0, len(bottom_data_np[0])))
    for i in range(len(bottom_data_np)):
        if not np.any(bottom_data_np[i][len(bottom_data_np[i]) - 3:]):  # 将没检测出数字的筛出
            count_array[i] = 0
            continue
        if np.any(bottom_data_np[i][len(bottom_data_np[i]) - 3:] - bottom_data_np[i][len(bottom_data_np[i]) - 3].astype(
                int)):  # 将非整数筛出
            count_array[i] = 0
            continue
        if (not math.isclose(bottom_data_np[i][len(bottom_data_np[i]) - 3],
                             bottom_data_np[i][len(bottom_data_np[i]) - 2])) or (
                not math.isclose(bottom_data_np[i][len(bottom_data_np[i]) - 2],
                                 bottom_data_np[i][len(bottom_data_np[i]) - 1])):  # 将有误差的数筛出
            count_array[i] = 0
            continue
        new_bottom_data_np = np.r_[new_bottom_data_np, [bottom_data_np[i]]]
    new_bottom_data_np = new_bottom_data_np[np.argsort(new_bottom_data_np[:, 6])]
    # 将非非误差整数记录到输出中
    output_bottom_data_np = np.zeros((0, len(bottom_data_np[0])))
    for i in range(len(bottom_data_np)):
        if count_array[i] == 0:
            output_bottom_data_np = np.r_[output_bottom_data_np, [bottom_data_np[i]]]
    # 3.如果升序整数中左右相差为一的整数对超过两对，则存在行列标记
    if len(new_bottom_data_np) != 0:
        if new_bottom_data_np[0][6] > 5:
            output_bottom_data_np = bottom_data_np
        else:
            k = 0
            for i in range(len(new_bottom_data_np) - 1):
                if abs(new_bottom_data_np[i][6] - new_bottom_data_np[i + 1][6]) == 1:
                    k += 1
            if k >= 3:
                # 4.按x和y坐标找到多于4个在同一行或者同一列的整数
                for i in range(len(new_bottom_data_np)):
                    l = 0
                    h = 0
                    out_bottom_data_l = np.zeros((len(new_bottom_data_np)))
                    out_bottom_data_h = np.zeros((len(new_bottom_data_np)))
                    out_bottom_data_l[i] = 1
                    out_bottom_data_h[i] = 1
                    for j in range(len(new_bottom_data_np)):
                        if not (new_bottom_data_np[i][2] > new_bottom_data_np[j][4] or new_bottom_data_np[i][4] <
                                new_bottom_data_np[j][2]):  # 两矩形在x坐标上的长有重叠
                            chongdie = new_bottom_data_np[i][4] - new_bottom_data_np[i][2] + new_bottom_data_np[j][4] - \
                                       new_bottom_data_np[j][2] - (
                                               max(new_bottom_data_np[i][4], new_bottom_data_np[j][4]) - min(
                                           new_bottom_data_np[i][2], new_bottom_data_np[j][2]))
                            if chongdie / (new_bottom_data_np[i][4] - new_bottom_data_np[i][2]) > 0.7 and chongdie / (
                                    new_bottom_data_np[j][4] - new_bottom_data_np[j][2]) > 0.7:  # 重叠区域占70%以上
                                l += 1
                                out_bottom_data_l[j] = 1
                        if not (new_bottom_data_np[i][3] > new_bottom_data_np[j][5] or new_bottom_data_np[i][5] <
                                new_bottom_data_np[j][3]):  # 两矩形在x坐标上的长有重叠
                            chongdie = new_bottom_data_np[i][5] - new_bottom_data_np[i][3] + new_bottom_data_np[j][5] - \
                                       new_bottom_data_np[j][3] - (
                                               max(new_bottom_data_np[i][5], new_bottom_data_np[j][5]) - min(
                                           new_bottom_data_np[i][3], new_bottom_data_np[j][3]))
                            if chongdie / (new_bottom_data_np[i][5] - new_bottom_data_np[i][3]) > 0.7 and chongdie / (
                                    new_bottom_data_np[j][5] - new_bottom_data_np[j][3]) > 0.7:  # 重叠区域占70%以上
                                h += 1
                                out_bottom_data_h[j] = 1
                    if h > 4 or l > 4:
                        break
                # 5.将标记行列数的整数剔除得到新的bottom_data

                if h > 4:
                    for a in range(len(new_bottom_data_np)):
                        if out_bottom_data_h[a] == 0:
                            print(output_bottom_data_np, new_bottom_data_np[a])
                            output_bottom_data_np = np.r_[output_bottom_data_np, [new_bottom_data_np[a]]]
                if l > 4:
                    for b in range(len(new_bottom_data_np)):
                        if out_bottom_data_l[b] == 0:
                            output_bottom_data_np = np.r_[output_bottom_data_np, [new_bottom_data_np[b]]]

            else:  # 没有升序差值为一的整数序列，则不存在标记行列数的整数
                output_bottom_data_np = bottom_data_np
    return output_bottom_data_np


def cal_max_medium_min_top(ocr_data):
    """计算顶部 OCR 文本的最大、中值与最小尺寸。"""
    根据key_info计算出max-medium_min
    '''

    # 排查是否存在'Φ'
    for i in range(len(ocr_data)):
        ed = 0
        num = 0
        list = ocr_data[i]['key_info']
        dic = {'+': 0, '-': 0, '±': 0, 'none': []}
        m_3 = np.array([0, 0, 0])
        for j in range(len(list)):
            for k in range(len(list[j])):
                if list[j][k] == 'Φ' and ed != 1:
                    if ocr_data[i]['Absolutely'] == 'mb_pin_diameter':
                        ocr_data[i]['Absolutely'] = 'pin_diameter+'
                        ed = 1
                    else:
                        ocr_data[i]['Absolutely'] = 'pin_diameter'
                        ed = 1
                try:
                    a = float(list[j][k])
                    if k > 0:
                        if list[j][k - 1] == '+':
                            dic['+'] = a
                        elif list[j][k - 1] == '-':
                            dic['-'] = a
                        elif list[j][k - 1] == '±':
                            dic['±'] = a
                        else:
                            dic['none'].append(a)
                    else:
                        dic['none'].append(a)
                    num += 1
                except:
                    a = 0
        compire_ratio = 3
        if len(dic['none']) == num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if y[0] / y[1] > compire_ratio:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - y[2]])
            else:
                m_3 = np.array([y[0], y[1], y[2]])
        elif len(dic['none']) == 2 and num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            elif dic['+'] != 0:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - y[1]])
            else:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 1 and num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            else:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 0 and num == 3:
            m_3 = m_3
        elif len(dic['none']) == 2 and num == 2:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if y[0] / y[1] > compire_ratio:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - y[1]])
            else:
                m_3 = np.array([y[0], (y[0] + y[1]) * 0.5, y[1]])
        elif len(dic['none']) == 1 and num == 2:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            elif dic['+'] != 0:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - dic['+']])
            else:
                m_3 = np.array([y[0] + dic['-'], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 0 and num == 2:
            m_3 = m_3
        elif len(dic['none']) == 1 and num == 1:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            m_3 = np.array([y[0], y[0], y[0]])
        elif len(dic['none']) == 0 and num == 1:
            m_3 = np.array(
                [dic['±'] + dic['+'] + dic['-'], dic['±'] + dic['+'] + dic['-'], dic['±'] + dic['+'] + dic['-']])
        else:
            m_3 = m_3
        ocr_data[i]['max_medium_min'] = m_3
    return ocr_data


def cal_max_medium_min_bottom(ocr_data):
    """计算底部 OCR 文本的最大、中值与最小尺寸。"""
    根据key_info计算出max-medium_min
    '''

    # 排查是否存在'Φ'
    # for i in range(len(ocr_data)):
    #     if np.array(ocr_data[i]['key_info']).ndim == 1:
    #         ocr_data[i]['key_info'] = [ocr_data[i]['key_info']]
    for i in range(len(ocr_data)):
        num = 0
        ed = 0
        list = ocr_data[i]['key_info']
        dic = {'+': 0, '-': 0, '±': 0, 'none': []}
        m_3 = np.array([0, 0, 0])
        for j in range(len(list)):
            for k in range(len(list[j])):
                if list[j][k] == 'Φ' and ed != 1:
                    if ocr_data[i]['Absolutely'] == 'mb_pin_diameter':
                        ocr_data[i]['Absolutely'] = 'pin_diameter+'
                        ed = 1
                    else:
                        ocr_data[i]['Absolutely'] = 'pin_diameter'
                        ed = 1
                try:
                    a = float(list[j][k])
                    if k > 0:
                        if list[j][k - 1] == '+':
                            dic['+'] = a
                        elif list[j][k - 1] == '-':
                            dic['-'] = a
                        elif list[j][k - 1] == '±':
                            dic['±'] = a
                        else:
                            dic['none'].append(a)
                    else:
                        dic['none'].append(a)
                    num += 1
                except:
                    a = 0
        compire_ratio = 3
        if len(dic['none']) == num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if y[0] / y[1] > compire_ratio:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - y[2]])
            else:
                m_3 = np.array([y[0], y[1], y[2]])
        elif len(dic['none']) == 2 and num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            elif dic['+'] != 0:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - y[1]])
            else:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 1 and num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            else:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 0 and num == 3:
            m_3 = m_3
        elif len(dic['none']) == 2 and num == 2:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if y[0] / y[1] > compire_ratio:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - y[1]])
            else:
                m_3 = np.array([y[0], (y[0] + y[1]) * 0.5, y[1]])
        elif len(dic['none']) == 1 and num == 2:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            elif dic['+'] != 0:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - dic['+']])
            else:
                m_3 = np.array([y[0] + dic['-'], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 0 and num == 2:
            m_3 = m_3
        elif len(dic['none']) == 1 and num == 1:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            m_3 = np.array([y[0], y[0], y[0]])
        elif len(dic['none']) == 0 and num == 1:
            m_3 = np.array(
                [dic['±'] + dic['+'] + dic['-'], dic['±'] + dic['+'] + dic['-'], dic['±'] + dic['+'] + dic['-']])
        else:
            m_3 = m_3
        ocr_data[i]['max_medium_min'] = m_3
    return ocr_data


def cal_max_medium_min_side(ocr_data):
    """计算侧面 OCR 文本的最大、中值与最小尺寸。"""
    根据key_info计算出max-medium_min
    '''
    # 排查是否存在唯一'max'

    for i in range(len(ocr_data)):
        max_num = 0
        no_acc = -1
        str = re.findall("[Mm][Aa][Xx]", ocr_data[i]['ocr_strings'])
        if len(str) > 0:
            max_num += len(str)
            no_acc = i
        if max_num > 0:
            ocr_data[no_acc]['Absolutely'] = 'high'

    # 排查是否存在'Φ'
    for i in range(len(ocr_data)):
        num = 0
        ed = 0
        list = ocr_data[i]['key_info']
        dic = {'+': 0, '-': 0, '±': 0, 'none': []}
        m_3 = np.array([0, 0, 0])
        for j in range(len(list)):
            for k in range(len(list[j])):
                if list[j][k] == 'Φ' and ed != 1:
                    if ocr_data[i]['Absolutely'] == 'mb_pin_diameter':
                        ocr_data[i]['Absolutely'] = 'pin_diameter+'
                        ed = 1
                    else:
                        ocr_data[i]['Absolutely'] = 'pin_diameter'
                        ed = 1
                try:
                    a = float(list[j][k])
                    if k > 0:
                        if list[j][k - 1] == '+':
                            dic['+'] = a
                        elif list[j][k - 1] == '-':
                            dic['-'] = a
                        elif list[j][k - 1] == '±':
                            dic['±'] = a
                        else:
                            dic['none'].append(a)
                    else:
                        dic['none'].append(a)
                    num += 1
                except:
                    a = 0
        compire_ratio = 3
        if len(dic['none']) == num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if y[0] / y[1] > compire_ratio:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - y[2]])
            else:
                m_3 = np.array([y[0], y[1], y[2]])
        elif len(dic['none']) == 2 and num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            elif dic['+'] != 0:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - y[1]])
            else:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 1 and num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            else:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 0 and num == 3:
            m_3 = m_3
        elif len(dic['none']) == 2 and num == 2:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if y[0] / y[1] > compire_ratio:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - y[1]])
            else:
                m_3 = np.array([y[0], (y[0] + y[1]) * 0.5, y[1]])
        elif len(dic['none']) == 1 and num == 2:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            elif dic['+'] != 0:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - dic['+']])
            else:
                m_3 = np.array([y[0] + dic['-'], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 0 and num == 2:
            m_3 = m_3
        elif len(dic['none']) == 1 and num == 1:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            m_3 = np.array([y[0], y[0], y[0]])
        elif len(dic['none']) == 0 and num == 1:
            m_3 = np.array(
                [dic['±'] + dic['+'] + dic['-'], dic['±'] + dic['+'] + dic['-'], dic['±'] + dic['+'] + dic['-']])
        else:
            m_3 = m_3
        ocr_data[i]['max_medium_min'] = m_3
    return ocr_data


def bind_data(yolox_num, ocr_data):
    """将 YOLO 数字框与 OCR 文本绑定。"""
    按照特殊yolox的框线将一个或者多个dbnet的框线合并，并把标注合并
    'key_info': [['3.505'], ['3.445']]
    '''
    new_ocr_data = []
    remember_no_arr = np.zeros(len(ocr_data))  # 记录dbnet的ocr是否被合并
    remember_no_arr_yolox = np.zeros(len(yolox_num))  # 记录yolox的框线是否用于合并
    # 1.针对每个yolox检测的data，看是否有两个及其以上的dbnet数据与之相交重叠
    for i in range(len(yolox_num)):
        k = 0
        x = np.zeros((len(ocr_data)))

        for j in range(len(ocr_data)):
            if remember_no_arr[j] == 0:
                if not (yolox_num[i][0] > ocr_data[j]['location'][2] or yolox_num[i][2] < ocr_data[j]['location'][
                    0]):  # 两矩形在x坐标上的长有重叠
                    if not (yolox_num[i][1] > ocr_data[j]['location'][3] or yolox_num[i][3] < ocr_data[j]['location'][
                        1]):  # 两矩形在y坐标上的高有重叠
                        k += 1
                        x[j] = 1
        # 2.如果有的重叠data框，则将yolox的框添加到新的dbnet的data框
        if k > 0:
            new_ocr_data_mid = {'location': yolox_num[i], 'ocr_strings': '', 'key_info': [],
                                'matched_pairs_location': [], 'matched_pairs_outside_or_inside': [],
                                'matched_pairs_yinXian': [], 'Absolutely': [], 'max_medium_min': []}
            for m in range(len(x)):
                if x[m] == 1:
                    new_ocr_data_mid['ocr_strings'] += (' ' + ocr_data[m]['ocr_strings'])
                    new_ocr_data_mid['key_info'].append(ocr_data[m]['key_info'])
                    if ocr_data[m]['Absolutely'] != []:
                        new_ocr_data_mid['Absolutely'] = ocr_data[m]['Absolutely']
            new_ocr_data.append(new_ocr_data_mid)

            remember_no_arr_yolox[i] += 1
            for l in range(len(x)):
                if x[l] == 1:
                    remember_no_arr[l] = 1

    # 3.将剩下的没有与yolox框重合的dbnet的data框添加到新的dbnet的data
    for i in range(len(ocr_data)):
        if remember_no_arr[i] == 0:
            ocr_data[i]['key_info'] = [ocr_data[i]['key_info']]
            new_ocr_data.append(ocr_data[i])

    # 4将剩下的没有与dbnet框重合的yolox框添加到新的dbnet的data
    # for i in range(len(yolox_num)):
    #     if remember_no_arr_yolox[i] == 0:
    #         new_dbnet_data = np.r_[new_dbnet_data, [yolox_num[i]]]
    # # new_dbnet_data = new_dbnet_data[1:,]
    # print(new_ocr_data)
    # for i in range(len(new_ocr_data)):
    #     if np.array(new_ocr_data[i]['key_info']).ndim == 1:
    #         new_ocr_data[i]['key_info'] = [new_ocr_data[i]['key_info']]

    return new_ocr_data


def ocr_get_data_QFP(image_path,
    """针对 QFP/QFN 流程调度 OCR 推理并整理结果。"""
                     yolox_pairs):  # 输入yolox输出的pairs坐标和匹配的data坐标以及图片地址，ocr识别文本后输出data内容按序保存在data_list_np（numpy二维数组）
    show_img_key = 0  # 是否显示过程中ocr待检测图片 0 = 不显示，1 = 显示
    yolox_pairs = np.array(yolox_pairs)
    ocr = PaddleOCR(use_angle_cls=True,
                    lang="en",
                    # det_model_dir="ppocr_model/det/en/en_PP-OCRv3_det_infer",
                    det_model_dir="ppocr_model/det/en/en_PP-OCRv3_sever_det_infer",

                    # rec_model_dir='ppocr_model/rec/en/en_PP-OCRv3_rec_infer',
                    rec_model_dir='ppocr_model/rec/en/en_PP-OCRv3_sever_rec_infer',
                    cls_model_dir='ppocr_model/cls/ch_ppocr_mobile_v2.0_cls_infer',
                    use_gpu=False)  # 导入模型， 禁用gpu
    with open(image_path, 'rb') as f:
        np_arr = np.frombuffer(f.read(), dtype=np.uint8)
        # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    #以彩图读取
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取
    data_list = []  # 按序存储pairs的data
    error_times = 0  # 记录ocr识别为空的次数
    no = 1  # ocr识别图片的序号
    # 针对yolox_pairs中的data在图片中的坐标，裁剪出图片区域用来给ocr检测
    for i in range(len(yolox_pairs)):
        # 裁剪识别区域的时候需要扩展一圈，以防yolox极限检测框导致某些数据边缘没有被检测
        # 只可能横着的裁剪成竖着的
        # 方案：横着的一定图片方向正确，扩展一圈识别，如果没有识别到，等比例扩大再识别；竖着的图片看宽长比是否小于0.7，小于则顺时针旋转90，如果识别不到，先认为是文本方向错误，逆时针90转回来识别。如果还识别不到，则等比例放大重复上述
        box = np.array([[yolox_pairs[i][0], yolox_pairs[i][1]], [yolox_pairs[i][2], yolox_pairs[i][1]],
                        [yolox_pairs[i][2], yolox_pairs[i][3]], [yolox_pairs[i][0], yolox_pairs[i][3]]],
                       np.float32)
        box_img = get_rotate_crop_image(img, box)  # yolox检测的原始data区域
        height, width = box_img.shape[0], box_img.shape[1]
        # ****************
        box_img = cv2.resize(box_img, (320, int(height * 320 / width)))  # 等比例放缩成320，适合ocr识别
        # ****************
        # print(yolox_pairs[i][12])
        if show_img_key == 1:
            cv2.imshow('no_correct_img', box_img)  # 显示当前ocr的识别区域
            cv2.waitKey(0)
        if yolox_pairs[i][2] - yolox_pairs[i][0] > yolox_pairs[i][3] - yolox_pairs[i][1]:
            KuoZhan_ratio = 0.25  # 扩展的比例
            KuoZhan_x = KuoZhan_ratio * abs(yolox_pairs[i][2] - yolox_pairs[i][0]) * (0.5)
            KuoZhan_y = KuoZhan_ratio * abs(yolox_pairs[i][3] - yolox_pairs[i][1]) * (0.5)
            box = np.array([[yolox_pairs[i][0] - KuoZhan_x, yolox_pairs[i][1] - KuoZhan_y],
                            [yolox_pairs[i][2] + KuoZhan_x, yolox_pairs[i][1] - KuoZhan_y],
                            [yolox_pairs[i][2] + KuoZhan_x, yolox_pairs[i][3] + KuoZhan_y],
                            [yolox_pairs[i][0] - KuoZhan_x, yolox_pairs[i][3] + KuoZhan_y]], np.float32)

            box_img = get_rotate_crop_image(img, box)

        if yolox_pairs[i][2] - yolox_pairs[i][0] < yolox_pairs[i][3] - yolox_pairs[i][
            1]:  # 竖着的data需要旋转90,如果识别不到那就是本来文本是横着结果截取范围是竖着的
            KuoZhan_ratio = 0.25
            KuoZhan_x = KuoZhan_ratio * abs(yolox_pairs[i][2] - yolox_pairs[i][0]) * (0.5)
            KuoZhan_y = KuoZhan_ratio * abs(yolox_pairs[i][3] - yolox_pairs[i][1]) * (0.5)
            box = np.array([[yolox_pairs[i][0] - KuoZhan_x, yolox_pairs[i][1] - KuoZhan_y],
                            [yolox_pairs[i][2] + KuoZhan_x, yolox_pairs[i][1] - KuoZhan_y],
                            [yolox_pairs[i][2] + KuoZhan_x, yolox_pairs[i][3] + KuoZhan_y],
                            [yolox_pairs[i][0] - KuoZhan_x, yolox_pairs[i][3] + KuoZhan_y]], np.float32)
            box_img = get_rotate_crop_image(img, box)
            # print("(yolox_pairs[i][10] - yolox_pairs[i][8])/(yolox_pairs[i][11]-yolox_pairs[i][9])",
            #       (yolox_pairs[i][10] - yolox_pairs[i][8]) / (yolox_pairs[i][11] - yolox_pairs[i][9]))
            rotate_key = 0
            length_to_weight = 0.7  # 长宽比 小于1
            if (yolox_pairs[i][2] - yolox_pairs[i][0]) / (
                    yolox_pairs[i][3] - yolox_pairs[i][1]) < length_to_weight:
                rotate_key = 1
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90

        height, width = box_img.shape[0], box_img.shape[1]
        # ****************
        box_img = cv2.resize(box_img, (320, int(height * 320 / width)))  # 等比例放缩成320，适合ocr识别

        print("正在OCR识别第", no, "个pairs匹配的data")
        no += 1
        if show_img_key == 1:
            cv2.imshow('origin_img', box_img)  # 显示当前ocr的识别区域
            cv2.waitKey(0)

        result = ocr.ocr(box_img, cls=True, det=True, rec=True)  # 问题：有可能yolox检测不到data，需要在yolox检测方面解决

        if (result == [None] or result == [[]]) and yolox_pairs[i][2] - yolox_pairs[i][0] < yolox_pairs[i][3] - \
                yolox_pairs[i][1] and rotate_key == 0:
            box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
            rotate_key = 1
            if show_img_key == 1:
                cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                cv2.waitKey(0)
            result = ocr.ocr(box_img, cls=True, det=True, rec=True)

        if (result == [None] or result == [[]]) and yolox_pairs[i][2] - yolox_pairs[i][0] < yolox_pairs[i][3] - \
                yolox_pairs[i][1] and rotate_key == 1:
            box_img = cv2.rotate(box_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90
            if show_img_key == 1:
                cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                cv2.waitKey(0)
            result = ocr.ocr(box_img, cls=True, det=True, rec=True)

        if result == [None] or result == [[]]:  # 如果识别不到，那么把图片”等比例“放大，不等比例会导致OCR识别失败
            if yolox_pairs[i][2] - yolox_pairs[i][0] > yolox_pairs[i][3] - yolox_pairs[i][1]:
                KuoZhan_ratio = 0.25
                KuoZhan_x = KuoZhan_ratio * abs(yolox_pairs[i][2] - yolox_pairs[i][0]) * (0.5)
                KuoZhan_y = KuoZhan_ratio * abs(yolox_pairs[i][3] - yolox_pairs[i][1]) * (0.5)
                box = np.array([[yolox_pairs[i][0] - KuoZhan_x, yolox_pairs[i][1] - KuoZhan_y],
                                [yolox_pairs[i][2] + KuoZhan_x, yolox_pairs[i][1] - KuoZhan_y],
                                [yolox_pairs[i][2] + KuoZhan_x, yolox_pairs[i][3] + KuoZhan_y],
                                [yolox_pairs[i][0] - KuoZhan_x, yolox_pairs[i][3] + KuoZhan_y]], np.float32)

                box_img = get_rotate_crop_image(img, box)

            if yolox_pairs[i][2] - yolox_pairs[i][0] < yolox_pairs[i][3] - yolox_pairs[i][
                1]:  # 竖着的data需要旋转90,如果识别不到那就是本来文本是横着结果截取范围是竖着的
                KuoZhan_ratio = 0.25
                KuoZhan_x = KuoZhan_ratio * abs(yolox_pairs[i][2] - yolox_pairs[i][0]) * (0.5)
                KuoZhan_y = KuoZhan_ratio * abs(yolox_pairs[i][3] - yolox_pairs[i][1]) * (0.5)
                box = np.array([[yolox_pairs[i][0] - KuoZhan_x, yolox_pairs[i][1] - KuoZhan_y],
                                [yolox_pairs[i][2] + KuoZhan_x, yolox_pairs[i][1] - KuoZhan_y],
                                [yolox_pairs[i][2] + KuoZhan_x, yolox_pairs[i][3] + KuoZhan_y],
                                [yolox_pairs[i][0] - KuoZhan_x, yolox_pairs[i][3] + KuoZhan_y]], np.float32)
                box_img = get_rotate_crop_image(img, box)
                # print("(yolox_pairs[i][10] - yolox_pairs[i][8])/(yolox_pairs[i][11]-yolox_pairs[i][9])",
                #       (yolox_pairs[i][10] - yolox_pairs[i][8]) / (yolox_pairs[i][11] - yolox_pairs[i][9]))
                rotate_key = 0
                if (yolox_pairs[i][2] - yolox_pairs[i][0]) / (yolox_pairs[i][3] - yolox_pairs[i][1]) < 0.75:
                    rotate_key = 1
                    box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
            # box_img = cv2.resize(box_img, (160, 80),cv2.INTER_AREA)
            box_img = img_resize(box_img)  # 等比例放大
            box_img = hist(box_img, show_img_key)  # 图像增强

            if show_img_key == 1:
                cv2.imshow('enhance_origin_img', box_img)  # 显示当前ocr的识别区域
                cv2.waitKey(0)

            result = ocr.ocr(box_img, cls=True, det=True, rec=True)  # 问题：有可能yolox检测不到data，需要在yolox检测方面解决

            if (result == [None] or result == [[]]) and yolox_pairs[i][2] - yolox_pairs[i][0] < yolox_pairs[i][3] - \
                    yolox_pairs[i][1] and rotate_key == 0:
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
                rotate_key = 1
                if show_img_key == 1:
                    cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                    cv2.waitKey(0)
                result = ocr.ocr(box_img, cls=True, det=True, rec=True)

            if (result == [None] or result == [[]]) and yolox_pairs[i][2] - yolox_pairs[i][0] < yolox_pairs[i][3] - \
                    yolox_pairs[i][1] and rotate_key == 1:
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90
                if show_img_key == 1:
                    cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                    cv2.waitKey(0)
                result = ocr.ocr(box_img, cls=True, det=True, rec=True)

        print("OCR识别data的result\n", result)
        # 将ocr识别的文本提取数据
        try:

            data_list_pairs = [yolox_pairs[i][0], yolox_pairs[i][1], yolox_pairs[i][2],
                               yolox_pairs[i][3], ]  # [999,识别内容1，识别内容2，...]

            for i in range(len(result[0])):
                # result[0][i][1][0] = comma_inter_point((result[0][i][1][0]))
                if get_data_and_del_en(comma_inter_point(result[0][i][1][0])) != '':
                    if get_data_and_del_en(comma_inter_point(result[0][i][1][0])) == []:
                        continue
                    else:
                        data_list_pairs_one, data_list_pairs_another = get_data_and_del_en(
                            comma_inter_point(result[0][i][1][0]))
                        print("len(data_list_pairs_one),data_list_pairs_one\n", len(data_list_pairs_one), ",",
                              data_list_pairs_one)
                        print("len(data_list_pairs_another),data_list_pairs_another\n",
                              len(data_list_pairs_another),
                              ",",
                              data_list_pairs_another)
                        if len(data_list_pairs_another) == 0 and data_list_pairs_one != [''] and len(
                                data_list_pairs_one) != 0:
                            # data_list_pairs.append(get_data_and_del_en(comma_inter_point(result[0][i][1][0])))
                            data_list_pairs.append(data_list_pairs_one)
                            print("data_list_pairs\n", data_list_pairs)
                        # if len(data_list_pairs_another) != 0:
                        if data_list_pairs_another != [''] and len(
                                data_list_pairs_another) != 0:  # 识别到类似"12+0.1"的文本时，get_data_and_del_en会返回12和0.1,之后判断大小来得到max，medium，min
                            data_list_pairs.append(data_list_pairs_one)
                            # print("data_list_pairs", data_list_pairs)
                            data_list_pairs.append(data_list_pairs_another)
                            print("data_list_pairs\n", data_list_pairs)
        except Exception as r:
            print('OCR未识别到内容，报错： %s' % (r))
            error_times += 1

        data_list.append(data_list_pairs)  # 将data的','替换为'.'，并仅提取数字，删除中英文按序保存
    # 对data_list中只存在min和max的值，以及只有一个median的，扩展为max，median和min
    print("data_list：一张img中的所有文本，按行OCR识别的结果，可以有多行的输出结果:\n", data_list)
    x = 7  # 一个框线中data最终输出包含数据个数
    data_list_np = np.empty((0, x))  # [坐标，max，medium，min]
    # 将ocr识别出的数据按照max，medium，min保存在data_list_np中
    compire_ratio = 0.5
    for i in range(len(data_list)):
        try:
            m_m_m = np.asarray(data_list[i])
            m_m_m = m_m_m.astype(np.float_)  # 识别出来的数据格式不对就会报错
            if (m_m_m[-3:] != 0).all():
                if len(m_m_m) == (x - 1):  # （1）OCR在一个框中识别出两个值（可能是最大+最小，或者中间值+误差）
                    # （2）找到最大值并比较数量级判断是（1）中提到的哪种情况
                    if m_m_m[x - 3] > m_m_m[x - 2] and m_m_m[x - 2] / m_m_m[x - 3] > compire_ratio:  # (2)最大+最小情况
                        m_m_m = np.append(m_m_m, (m_m_m[x - 3] + m_m_m[x - 2]) / 2)
                        min = m_m_m[x - 2]
                        m_m_m[x - 2] = m_m_m[x - 1]
                        m_m_m[x - 1] = min
                    elif m_m_m[x - 3] < m_m_m[x - 2] and m_m_m[x - 3] / m_m_m[x - 2] > compire_ratio:  # (2)最大+最小情况
                        max = m_m_m[x - 2]
                        m_m_m[x - 2] = m_m_m[x - 3]
                        m_m_m[x - 3] = max
                        m_m_m = np.append(m_m_m, (m_m_m[x - 2] + m_m_m[x - 3]) / 2)
                    elif m_m_m[x - 3] > m_m_m[x - 2] and m_m_m[x - 2] / m_m_m[x - 3] < compire_ratio:  # （2）中间值+误差情况
                        max = m_m_m[x - 2] + m_m_m[x - 3]
                        min = m_m_m[x - 3] - m_m_m[x - 2]
                        medium = m_m_m[x - 3]
                        m_m_m[x - 3] = max
                        m_m_m[x - 2] = medium
                        m_m_m = np.append(m_m_m, min)
                    elif m_m_m[x - 3] < m_m_m[x - 2] and m_m_m[x - 3] / m_m_m[x - 2] < compire_ratio:
                        max = m_m_m[x - 2] + m_m_m[x - 3]
                        min = m_m_m[x - 2] - m_m_m[x - 3]
                        medium = m_m_m[x - 2]
                        m_m_m[x - 3] = max
                        m_m_m[x - 2] = medium
                        m_m_m = np.append(m_m_m, min)
                if len(m_m_m) == (x - 2):  # OCR在一个框中识别出一个值
                    m_m_m = np.append(m_m_m, m_m_m[x - 3])
                    m_m_m = np.append(m_m_m, m_m_m[x - 3])
                if len(m_m_m) == x - 3:  # OCR在一个框中没有识别到数字
                    m_m_m = np.append(m_m_m, 0)
                    m_m_m = np.append(m_m_m, 0)
                    m_m_m = np.append(m_m_m, 0)
                if len(m_m_m) == x:  # OCR在一个框中识别出三个值
                    y = m_m_m[x - 3:]
                    # x.sort()  # [1 2 3 4 6 8]
                    # x = abs(np.sort(-x))  # [8 6 4 3 2 1] 先取相反数排序，再加上绝对值得到原数组的降序
                    y = np.sort(y)
                    y = abs(np.sort(-y))
                    m_m_m[x - 3:] = y
                    # data_list_np = np.c_[data_list_np, m_m_m]  # 添加行
                try:
                    data_list_np = np.r_[data_list_np, [m_m_m]]
                except:
                    print("错误，ocr识别到过多数据，错误拼接：\n", m_m_m, data_list_np)
                print("中间量data_list_np\n", data_list_np)
                print("即将添加到data_list_np的m_m_m\n", m_m_m)
                print("添加了m_m_m之后的data_list_np\n", data_list_np)
        except:
            print("ocr识别出双小数点")
        # except Exception as e:
        #     print("*******报错*******\n", e)
    # data_list_np = data_list_np.T

    print(
        "该视图下最终结果data_list_np：[[方向,pairs的长度,max,medium,min],[方向,pairs的长度,max,medium,min],...]:\n",
        data_list_np)
    # data_list_np：[[方向, max, medium, min], [方向, max, medium, min], ...]

    # print("data_list",data_list)
    if error_times != 0:
        print("在该视图下有", error_times, "个data是OCR识别为空的,请检查并优化识别效果")
    # print("在该视图下有",error_times,"个data是OCR识别为空的,请检查并优化识别效果")
    return data_list_np


def ocr_get_data_onnx(image_path,
    """调用 ONNX OCR 模型识别并返回结构化数据。"""
                      yolox_pairs):  # 输入yolox输出的pairs坐标和匹配的data坐标以及图片地址，ocr识别文本后输出data内容按序保存在data_list_np（numpy二维数组）
    ocr_data = []  # 按序存储pairs的data
    dt_boxes = []
    # 裁剪识别区域的时候需要扩展一圈，以防yolox极限检测框导致某些数据边缘没有被检测
    for i in range(len(yolox_pairs)):
        KuoZhan_x = 0
        KuoZhan_y = 0
        dt_boxes_middle = np.array([[yolox_pairs[i][0] - KuoZhan_x, yolox_pairs[i][1] - KuoZhan_y],
                                    [yolox_pairs[i][2] + KuoZhan_x, yolox_pairs[i][1] - KuoZhan_y],
                                    [yolox_pairs[i][2] + KuoZhan_x, yolox_pairs[i][3] + KuoZhan_y],
                                    [yolox_pairs[i][0] - KuoZhan_x, yolox_pairs[i][3] + KuoZhan_y]], dtype=np.float32)
        dt_boxes.append(dt_boxes_middle)
    ocr_data = Run_onnx(image_path, dt_boxes)
    return ocr_data


def ocr_onnx_table_or_number(img_path, dbnet_data):
    """判断文本区域是表格还是数字并分别处理。"""
    dbnet_data = ocr_en_cn_onnx(img_path, dbnet_data)
    # dbnet_data np.(,5)['x1','y1','x2','y2','0.12']
    # 识别出的标注统计英文字母的数量和数字的数量
    letter_num = 0
    number_num = 0
    new_dbnet_data = np.zeros((0, 5))
    for i in range(len(dbnet_data)):
        strings = dbnet_data[i][4]
        if len(strings) < 4:
            new_dbnet_data = np.r_[new_dbnet_data, [dbnet_data[i]]]
            for k in range(len(strings)):
                if strings[k].isdigit():
                    number_num += 1
                if strings[k].isalpha():
                    letter_num += 1
    ratio = 0.4
    print("letter_num, number_num, letter_num / (letter_num + number_num)", letter_num, number_num,
          letter_num / (letter_num + number_num))
    if letter_num / (letter_num + number_num) > ratio:
        key = "table"
    else:
        key = "number"
    return key, new_dbnet_data


def ocr_data(img_path, dbnet_data):
    """统一调度 OCR 推理并合并多模型输出。"""
    # ocr_get_data(img_path,top_yolox_num)
    dbnet_data = ocr_get_data_onnx(img_path, dbnet_data)
    return dbnet_data


def delete_ocr_zeros(data):
    """清理 OCR 结果中误识别的零值或噪声。"""
    new_data = np.zeros((0, data.shape[1]))
    for i in range(len(data)):
        if not (data[i][4] == data[i][5] == data[i][6] == 0):
            new_data = np.r_[new_data, [data[i]]]
    return new_data


def yolox_get_pairs_and_data(img_path):
    """获取 YOLO 标尺对与对应检测数据。"""
    yolox_pairs, yolox_num, other = begain_output_QFP_pairs_data_location(img_path)
    # yolox_pairs np.二维数组[x1,y1,x2,y2,0 = outside 1 = inside]
    # yolox_num np.二维数组[x1,y1,x2,y2]
    return yolox_pairs, yolox_num, other


def get_img_info(img_path):
    """读取图像基本属性信息。"""
    # import cv2
    image = cv2.imread(img_path)
    size = image.shape
    w = size[1]  # 宽度
    h = size[0]  # 高度
    return w, h


def dbnet_get_data(img_path):
    """调用 DBNet 模型获取文本框位置。"""
    # import sys
    # sys.path.append("..")
    # from ocr_onnx.onnx_use import Run_onnx_det
    location_cool = Run_onnx_det(img_path)
    # import numpy as np
    dbnet_data = np.empty((len(location_cool), 4))  # [x1,x2,x3,x4]
    for i in range(len(location_cool)):
        dbnet_data[i][0] = min(location_cool[i][2], location_cool[i][0])
        dbnet_data[i][1] = min(location_cool[i][3], location_cool[i][1])
        dbnet_data[i][2] = max(location_cool[i][2], location_cool[i][0])
        dbnet_data[i][3] = max(location_cool[i][3], location_cool[i][1])
    return dbnet_data


def dbnet_get_data_db(img_path):
    """针对 DB 版本模型读取文本框位置。"""
    # from system_test import Dbnet_Inference
    location_cool = Dbnet_Inference(img_path)
    # import numpy as np
    dbnet_data = np.empty((len(location_cool), 4))  # [x1,x2,x3,x4]
    for i in range(len(location_cool)):
        dbnet_data[i][0] = min(location_cool[i][2], location_cool[i][6])
        dbnet_data[i][1] = min(location_cool[i][3], location_cool[i][7])
        dbnet_data[i][2] = max(location_cool[i][2], location_cool[i][6])
        dbnet_data[i][3] = max(location_cool[i][3], location_cool[i][7])
    return dbnet_data


def dbnet_get_num(img_path):
    """获取 DBNet 输出的数字类文本框。"""
    dbnet_data = dbnet_get_data(img_path)
    return dbnet_data


def get_pairs_data(img_path):
    """整合 YOLO 与 OCR 结果为 pairs 数据结构。"""
    # import time
    yolox_pairs, yolox_num, other = yolox_get_pairs_and_data(img_path)
    start = time.time()
    # dbnet_data = dbnet_get_num(img_path)
    end = time.time()
    dbnet_time = end - start
    # yolox_pairs  np.二维数组[x1,y1,x2,y2,0 = outside 1 = inside]
    # yolox_num  np.二维数组[x1,y1,x2,y2]
    # dbnet_data  np.二维数组[x1,y1,x2,y2]
    return yolox_pairs, yolox_num, other, dbnet_time


def show_data_table(img_path, data):
    """以表格形式可视化检测数据。"""
    data:np(,5)[x1,y1,x2,y2]
    '''
    wh_key1 = True
    while wh_key1:
        auto_key = input("是否展示dbnet框选的标注:y/n:")

        if auto_key == 'y' or auto_key == 'Y':
            auto_bool = False
            wh_key1 = False
        elif auto_key == 'n' or auto_key == 'N':
            auto_bool = True
            wh_key1 = False
        else:
            print("未输入正确，请重新输入：y/n:")
            wh_key1 = True
    if auto_bool == False:
        # import numpy as np
        # import cv2 as cv

        with open(img_path, 'rb') as f:
            np_arr = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 以彩图读取
            # img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取

            for i in range(len(data)):
                # 绘制一个红色矩形
                ptLeftTop = (int(data[i][0]), int(data[i][1]))
                ptRightBottom = (int(data[i][2]), int(data[i][3]))
                point_color = (0, 0, 255)  # BGR
                thickness = 2
                lineType = 8
                cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

        cv2.namedWindow("data(red)", 0)
        cv2.imshow('data(red)', img)
        cv2.waitKey(0)  # 显示 10000 ms 即 10s 后消失
        cv2.destroyAllWindows()


def show_data(img_path, data):
    """在图像上展示检测框和相关信息。"""
    wh_key1 = True
    while wh_key1:
        auto_key = input("是否展示dbnet框选的尺寸数字:y/n:")

        if auto_key == 'y' or auto_key == 'Y':
            auto_bool = False
            wh_key1 = False
        elif auto_key == 'n' or auto_key == 'N':
            auto_bool = True
            wh_key1 = False
        else:
            print("未输入正确，请重新输入：y/n:")
            wh_key1 = True
    if auto_bool == False:
        # import numpy as np
        # import cv2 as cv

        with open(img_path, 'rb') as f:
            np_arr = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 以彩图读取
            # img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取

            for i in range(len(data)):
                # 绘制一个红色矩形
                ptLeftTop = (int(data[i][0]), int(data[i][1]))
                ptRightBottom = (int(data[i][2]), int(data[i][3]))
                point_color = (0, 0, 255)  # BGR
                thickness = 2
                lineType = 8
                cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

        cv2.namedWindow("data(red)", 0)
        cv2.imshow('data(red)', img)
        cv2.waitKey(0)  # 显示 10000 ms 即 10s 后消失
        cv2.destroyAllWindows()


def show_ocr_result(img_path, ocr):
    """在图像上绘制 OCR 识别的文本结果。"""
    # import numpy as np
    # import cv2 as cv
    wh_key1 = True
    while wh_key1:
        auto_key = input("是否展示ocr识别的标注:y/n:")

        if auto_key == 'y' or auto_key == 'Y':
            auto_bool = False
            wh_key1 = False
        elif auto_key == 'n' or auto_key == 'N':
            auto_bool = True
            wh_key1 = False
        else:
            print("未输入正确，请重新输入：y/n:")
            wh_key1 = True
    if auto_bool == False:

        img = cv2.imread(img_path)
        for i in range(len(ocr)):
            # 绘制一个红色矩形
            ptLeftTop = (int(ocr[i]['location'][0]), int(ocr[i]['location'][1]))
            ptRightBottom = (int(ocr[i]['location'][2]), int(ocr[i]['location'][3]))
            point_color = (0, 0, 255)  # BGR
            thickness = 2
            lineType = 8
            cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

            # 调用cv.putText()添加文字
            if ocr[i]['max_medium_min'] != []:
                text = str(ocr[i]['max_medium_min'][0]) + ',' + str(ocr[i]['max_medium_min'][1]) + ',' + str(
                    ocr[i]['max_medium_min'][2])
                # AddText = img.copy()
                cv2.putText(img, text, (int(ocr[i]['location'][0]), int(ocr[i]['location'][1])),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 139), 2)
            # import numpy as np
            # 将原图片和添加文字后的图片拼接起来
            # img = np.hstack([img, AddText])

        cv2.namedWindow("data(red)", 0)
        cv2.imshow('data(red)', img)
        cv2.waitKey(0)  # 显示 10000 ms 即 10s 后消失
        cv2.destroyAllWindows()
    wh_key1 = True
    while wh_key1:
        auto_key = input("是否修改ocr识别的标注:y/n:")

        if auto_key == 'y' or auto_key == 'Y':
            auto_bool = False
            wh_key1 = False
        elif auto_key == 'n' or auto_key == 'N':
            auto_bool = True
            wh_key1 = False
        else:
            print("未输入正确，请重新输入：y/n:")
            wh_key1 = True
    if auto_bool == False:
        for i in range(len(ocr)):
            img = cv2.imread(img_path)
            # 绘制一个红色矩形
            ptLeftTop = (int(ocr[i]['location'][0]), int(ocr[i]['location'][1]))
            ptRightBottom = (int(ocr[i]['location'][2]), int(ocr[i]['location'][3]))
            point_color = (0, 0, 255)  # BGR
            thickness = 2
            lineType = 8
            cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

            cv2.namedWindow("data_to_be_correct", 0)
            cv2.imshow('data_to_be_correct', img)
            cv2.waitKey(0)  # 显示 10000 ms 即 10s 后消失
            cv2.destroyAllWindows()
            print("修改格式: max medium min，正在修改第", i + 1, "个")
            print("ocr识别结果max medium min：", ocr[i])
            right_data = input("请输入修改值，无需修改直接回车:")
            str_list = right_data.split()
            ocr[i]['max_medium_min'] = str_list
    return ocr


def show_ocr_result_table(img_path, data):
    """生成表格视图展示 OCR 结果。"""
    data:np(,5)['x1','y1','x2','y2','A1']
    '''
    data_a = (data[:, 0:4]).astype(float)
    data_b = data
    data = data_a

    # import numpy as np
    # import cv2 as cv
    wh_key1 = True
    while wh_key1:
        auto_key = input("是否展示ocr识别的标注:y/n:")

        if auto_key == 'y' or auto_key == 'Y':
            auto_bool = False
            wh_key1 = False
        elif auto_key == 'n' or auto_key == 'N':
            auto_bool = True
            wh_key1 = False
        else:
            print("未输入正确，请重新输入：y/n:")
            wh_key1 = True
    if auto_bool == False:

        img = cv2.imread(img_path)
        for i in range(len(data)):
            # 绘制一个红色矩形
            ptLeftTop = (int(data[i][0]), int(data[i][1]))
            ptRightBottom = (int(data[i][2]), int(data[i][3]))
            point_color = (0, 0, 255)  # BGR
            thickness = 2
            lineType = 8
            cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

            # 调用cv.putText()添加文字
            if data_b[i, 4] != '':
                text = data_b[i, 4]
                # AddText = img.copy()
                cv2.putText(img, text, (int(data[i][0]), int(data[i][1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 139), 2)
            # import numpy as np
            # 将原图片和添加文字后的图片拼接起来
            # img = np.hstack([img, AddText])

        cv2.namedWindow("data(red)", 0)
        cv2.imshow('data(red)', img)
        cv2.waitKey(0)  # 显示 10000 ms 即 10s 后消失
        cv2.destroyAllWindows()
    wh_key1 = True
    while wh_key1:
        auto_key = input("是否修改ocr识别的标注:y/n:")

        if auto_key == 'y' or auto_key == 'Y':
            auto_bool = False
            wh_key1 = False
        elif auto_key == 'n' or auto_key == 'N':
            auto_bool = True
            wh_key1 = False
        else:
            print("未输入正确，请重新输入：y/n:")
            wh_key1 = True
    if auto_bool == False:
        for i in range(len(data)):
            img = cv2.imread(img_path)
            # 绘制一个红色矩形
            ptLeftTop = (int(data[i][0]), int(data[i][1]))
            ptRightBottom = (int(data[i][2]), int(data[i][3]))
            point_color = (0, 0, 255)  # BGR
            thickness = 2
            lineType = 8
            cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

            cv2.namedWindow("data_to_be_correct", 0)
            cv2.imshow('data_to_be_correct', img)
            cv2.waitKey(0)  # 显示 10000 ms 即 10s 后消失
            cv2.destroyAllWindows()
            print("修改格式: (str)，正在修改第", i + 1, "个")
            print("ocr识别结果(str)：", data_b[i, 4])
            right_data = input("请输入修改值，无需修改直接回车:")

            if len(right_data) > 0:
                data_b[i, 4] = right_data

    return data_b


def match_pairs_data_table(pairs,
    """匹配表格类的标尺对与 OCR 文本。"""
                           data):  # pairs[[0,1,2,3],[0,1,2,3]];data[['0','1','2','3','A1'],['0','1','2','3','B1']]
    #
    data_copy = data.copy()
    data = (data[:, 0: 4]).astype(float)

    matched_pairs_data = np.zeros((0, 8))  # 匹配的尺寸线和尺寸数字存在这里
    middle_arr = np.zeros((8))  # 作为中间量存入matched_pairs_data
    matched_data = np.zeros((len(data)))  # 标记是否该data被匹配
    matched_pairs = np.zeros((len(pairs)))  # 标记是否该pairs被匹配
    # 1.直接匹配和pairs重叠的data，并从data池中删除。完成后删除匹配到的pairs，保证一个pairs可以和多个重叠data匹配
    for i in range(len(pairs)):
        for j in range(len(data)):
            if not (pairs[i][0] > data[j][2]) or pairs[i][2] < data[j][0]:  # 两矩形在x坐标上的长有重叠
                if not (pairs[i][1] > data[j][3] or pairs[i][3] < data[j][1]):  # 两矩形在y坐标上的高有重叠
                    matched_data[j] = 1
                    matched_pairs[i] = 1
                    middle_arr[0:4] = pairs[i]
                    middle_arr[4:8] = data[j]
                    print("匹配有重叠的pairs与data\n", middle_arr)
                    matched_pairs_data = np.r_[matched_pairs_data, [middle_arr]]
    # print(matched_pairs_data)
    new_data = np.zeros((0, 4))  # 过滤已经匹配了的数据的尺寸数字数组
    for i in range(len(data)):
        if matched_data[i] == 0:
            new_data = np.r_[new_data, [data[i]]]
    new_pairs = np.zeros((0, 4))
    for i in range(len(pairs)):
        if matched_pairs[i] == 0:
            new_pairs = np.r_[new_pairs, [pairs[i]]]
    # 2.针对没有重叠的pairs，（1）横向pair只能匹配横向data（2）竖向pair可能匹配横向或者竖向data
    # 将pairs和data重叠的作为标尺，其他匹配的pairs和data比例不能过大
    # 横向pairs 1.重叠的data 2.同方向且中心轴相同的离得近的data（紧贴） 3.横向水平且离得近（欧氏距离不超过两个pairs长）4.欧氏距离
    # 竖向pairs 1.重叠的data 2.同方向且中心轴相同的离得近的data（紧贴） 3.竖向水平且离得近（欧氏距离不超过两个pairs长）4.欧氏距离
    # 每次匹配三个data并排序，如果有条件差不多的data：找相近的pairs按照pairs长度分别将两个data匹配
    # 3.方法：同方向且中心轴相同的离得近的data（紧贴）#####pairs的方向按照长宽比判断，但方向区分度不大的pairs不能判断方向
    # 3.实现版本：找pairs中心轴穿过的data，根据水平距离排序再通过标尺筛选
    # 可以改进：根据ocr识别的图片判断data方向而不是根据data长宽
    #####################################横向
    mid_arr = np.zeros((5))  # 前4位存储位置，第5位存储尺寸数据和尺寸线的最近距离
    matched_new_data = np.zeros((len(new_data)))  # 标记是否该data被匹配
    matched_new_pairs = np.zeros((len(new_pairs)))  # 标记是否该pairs被匹配
    for i in range(len(new_pairs)):
        maybe_match_data = np.zeros((0, 5))  # 存储可能的尺寸数字，前4位存储位置，第5位存储尺寸数据和尺寸线的最近距离
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] > new_pairs[i][3] - new_pairs[i][1]:  # 横向pairs
                for j in range(len(new_data)):
                    if new_data[j][2] - new_data[j][0] > new_data[j][3] - new_data[j][1]:  # 横向data
                        if (new_pairs[i][2] + new_pairs[i][0]) * 0.5 < new_data[j][2] and (
                                new_pairs[i][2] + new_pairs[i][0]) * 0.5 > new_data[j][0]:  # 横向pairs中心轴穿过横向data
                            if (new_pairs[i][3] - new_pairs[i][1]) * 4 > min(abs(new_pairs[i][3] - new_data[j][1]), abs(
                                    new_pairs[i][1] - new_data[j][3])):  # pairs与data距离不超过pairs高度2倍
                                mid_arr[0:4] = new_data[j]
                                mid_arr[4] = min(abs(new_pairs[i][3] - new_data[j][1]),
                                                 abs(new_pairs[i][1] - new_data[j][3]))
                                maybe_match_data = np.r_[maybe_match_data, [mid_arr]]
                maybe_match_data = maybe_match_data[np.argsort(maybe_match_data[:, 4])]  # 按距离从小到大排序
                # 当ruler中data比maybe中data大时，ruler中pairs一定比data中
                middle_arr[0:4] = new_pairs[i]
                if len(maybe_match_data) > 0:
                    middle_arr[4:8] = maybe_match_data[0, 0:4]
                else:
                    middle_arr[4:8] = np.array([0, 0, 0, 0])
                print("匹配同方向且中心轴相同的离得近的data（紧贴）\n", middle_arr)
                matched_pairs_data = np.r_[matched_pairs_data, [middle_arr]]
                matched_new_pairs[i] = 1
                if (middle_arr[4:8] != np.array([0, 0, 0, 0])).any():
                    for i in range(len(new_data)):
                        if (maybe_match_data[0, 0:4] == new_data[i]).all():
                            matched_new_data[i] = 1
                    break
    new_new_pairs = np.zeros((0, 4))
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
    new_pairs = new_new_pairs
    new_new_data = np.zeros((0, 4))
    for i in range(len(new_data)):
        if matched_new_data[i] == 0:
            new_new_data = np.r_[new_new_data, [new_data[i]]]
    new_data = new_new_data
    ###########################竖向
    mid_arr = np.zeros((5))
    matched_new_data = np.zeros((len(new_data)))
    matched_new_pairs = np.zeros((len(new_pairs)))
    for i in range(len(new_pairs)):
        maybe_match_data = np.zeros((0, 5))
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] < new_pairs[i][3] - new_pairs[i][1]:  # 竖向pairs
                for j in range(len(new_data)):
                    if new_data[j][2] - new_data[j][0] < new_data[j][3] - new_data[j][1]:  # 竖向data
                        if (new_pairs[i][3] + new_pairs[i][1]) * 0.5 < new_data[j][3] and (
                                new_pairs[i][3] + new_pairs[i][1]) * 0.5 > new_data[j][1]:  # 竖向pairs中心轴穿过竖向data
                            if (new_pairs[i][2] - new_pairs[i][0]) * 2 > min(abs(new_pairs[i][2] - new_data[j][0]), abs(
                                    new_pairs[i][0] - new_data[j][2])):  # pairs与data距离不超过pairs高度2倍
                                mid_arr[0: 4] = new_data[j]
                                mid_arr[4] = min(abs(new_pairs[i][2] - new_data[j][0]),
                                                 abs(new_pairs[i][0] - new_data[j][2]))
                                maybe_match_data = np.r_[maybe_match_data, [mid_arr]]
                maybe_match_data = maybe_match_data[np.argsort(maybe_match_data[:, 4])]  # 按距离从小到大排序
                # 当ruler中data比maybe中data大时，ruler中pairs一定比data中

                middle_arr[0:4] = new_pairs[i]
                if len(maybe_match_data) > 0:
                    middle_arr[4:8] = maybe_match_data[0, 0:4]
                else:
                    middle_arr[4:8] = np.array([0, 0, 0, 0])
                print("匹配同方向且中心轴相同的离得近的data（紧贴）\n", middle_arr)
                matched_pairs_data = np.r_[matched_pairs_data, [middle_arr]]
                matched_new_pairs[i] = 1
                if (middle_arr[4:8] != np.array([0, 0, 0, 0])).any():
                    for i in range(len(new_data)):
                        if (maybe_match_data[0, 0:4] == new_data[i]).all():
                            matched_new_data[i] = 1
                    break
    new_new_pairs = np.zeros((0, 4))
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
    new_pairs = new_new_pairs
    new_new_data = np.zeros((0, 4))
    for i in range(len(new_data)):
        if matched_new_data[i] == 0:
            new_new_data = np.r_[new_new_data, [new_data[i]]]
    new_data = new_new_data
    # 4.1横向箭头横向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）（横向箭头一般只会匹配横向数据）
    # 实际：横向箭头匹配数据（不管方向）：先在pairs的y坐标上有重叠的data填入待match，再将pairs高度扩充三倍，在y轴上与之相交的data按欧氏距离大小排列为待匹配，用ruler检验
    matched_new_pairs = np.zeros((len(new_pairs)))
    matched_new_data = np.zeros((len(new_data)))
    for i in range(len(new_pairs)):
        maybe_match_data_a = np.zeros((0, 4))
        maybe_match_data_b = np.zeros((0, 5))
        middle = np.zeros((5))
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] > new_pairs[i][3] - new_pairs[i][1]:  # 横向pairs
                for j in range(len(new_data)):
                    if not (new_pairs[i][1] > new_data[j][3] or new_pairs[i][3] < new_data[j][1]):  # 两矩形在y坐标上的高有重叠
                        if min(abs(new_data[j][0] - new_pairs[i][2]), abs(new_data[j][2] - new_pairs[i][0])) < 5 * (
                                new_pairs[i][3] - new_pairs[i][1]):  # pairs和data横向距离很近
                            maybe_match_data_a = np.r_[maybe_match_data_a, [new_data[j]]]
                    if not (new_pairs[i][1] - (new_pairs[i][3] - new_pairs[i][1]) * 2 > new_data[j][3] or new_pairs[i][
                        3] + (new_pairs[i][3] - new_pairs[i][1]) * 2 < new_data[j][1]):  # 两矩形在y坐标上的高有重叠
                        if min(abs(new_data[j][0] - new_pairs[i][2]), abs(new_data[j][2] - new_pairs[i][0])) < 7 * (
                                new_pairs[i][3] - new_pairs[i][1]):
                            if new_data[j] not in maybe_match_data_a:
                                middle[0:4] = new_data[j]
                                middle[4] = min(abs(new_data[j][0] - new_pairs[i][2]),
                                                abs(new_data[j][2] - new_pairs[i][0]))
                                maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                maybe_match_data_b = maybe_match_data_b[np.argsort(maybe_match_data_b[:, 4])]  # 按距离从小到大排序
                maybe_match_data_b = maybe_match_data_b[:, 0:4]
                maybe_match_data = np.append(maybe_match_data_a, maybe_match_data_b, axis=0)

                middle_arr[0:4] = new_pairs[i]
                if len(maybe_match_data) > 0:
                    middle_arr[4:8] = maybe_match_data[0, 0:4]
                else:
                    middle_arr[4:8] = np.array([0, 0, 0, 0])
                print(
                    "横向箭头横向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）（横向箭头一般只会匹配横向数据）\n",
                    middle_arr)
                matched_pairs_data = np.r_[matched_pairs_data, [middle_arr]]
                matched_new_pairs[i] = 1
                if (middle_arr[4:8] != np.array([0, 0, 0, 0])).any():
                    for i in range(len(new_data)):
                        if (maybe_match_data[0, 0:4] == new_data[i]).all():
                            matched_new_data[i] = 1
                    break
    new_new_pairs = np.zeros((0, 4))
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
    new_pairs = new_new_pairs
    new_new_data = np.zeros((0, 4))
    for i in range(len(new_data)):
        if matched_new_data[i] == 0:
            new_new_data = np.r_[new_new_data, [new_data[i]]]
    new_data = new_new_data
    # print("new_pairs,new_data",new_pairs,new_data)
    # print("matched_pairs_data",matched_pairs_data)
    # 4.2竖向箭头竖向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）（竖向箭头可能匹配到两个方向的数据）
    # 实际：竖向箭头匹配数据（不管方向）：先在pairs的x坐标上有重叠的data填入待match，再将pairs宽度扩充三倍，在x轴上与之相交的data按欧氏距离大小排列为待匹配，用ruler检验
    matched_new_pairs = np.zeros((len(new_pairs)))
    matched_new_data = np.zeros((len(new_data)))
    for i in range(len(new_pairs)):
        maybe_match_data_a = np.zeros((0, 4))
        maybe_match_data_b = np.zeros((0, 5))
        middle = np.zeros((5))
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] < new_pairs[i][3] - new_pairs[i][1]:  # 竖向pairs
                for j in range(len(new_data)):
                    if not (pairs[i][0] > data[j][2] or pairs[i][2] < data[j][0]):  # 两矩形在x坐标上的长有重叠
                        if min(abs(new_data[j][1] - new_pairs[i][3]), abs(new_data[j][3] - new_pairs[i][1])) < 5 * (
                                new_pairs[i][2] - new_pairs[i][0]):  # pairs和data横向距离很近
                            maybe_match_data_a = np.r_[maybe_match_data_a, [new_data[j]]]
                    if not (new_pairs[i][0] - (new_pairs[i][2] - new_pairs[i][0]) * 2 > new_data[j][2] or new_pairs[i][
                        2] + (new_pairs[i][2] - new_pairs[i][0]) * 2 < new_data[j][0]):  # 两矩形在y坐标上的高有重叠
                        if min(abs(new_data[j][1] - new_pairs[i][3]), abs(new_data[j][3] - new_pairs[i][1])) < 7 * (
                                new_pairs[i][2] - new_pairs[i][0]):
                            if new_data[j] not in maybe_match_data_a:
                                middle[0:5] = new_data[j]
                                middle[5] = min(abs(new_data[j][1] - new_pairs[i][3]),
                                                abs(new_data[j][3] - new_pairs[i][1]))
                                maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                maybe_match_data_b = maybe_match_data_b[np.argsort(maybe_match_data_b[:, 4])]  # 按距离从小到大排序
                maybe_match_data_b = maybe_match_data_b[:, 0:4]
                maybe_match_data = np.append(maybe_match_data_a, maybe_match_data_b, axis=0)

                middle_arr[0:4] = new_pairs[i]
                if len(maybe_match_data) > 0:
                    middle_arr[4:8] = maybe_match_data[0, 0:4]
                else:
                    middle_arr[4:8] = np.array([0, 0, 0, 0])
                print(
                    "竖向箭头竖向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）\n",
                    middle_arr)
                matched_pairs_data = np.r_[matched_pairs_data, [middle_arr]]
                matched_new_pairs[i] = 1
                if (middle_arr[4:8] != np.array([0, 0, 0, 0])).any():
                    for i in range(len(new_data)):
                        if (maybe_match_data[0, 0:4] == new_data[i]).all():
                            matched_new_data[i] = 1
                    break
    new_new_pairs = np.zeros((0, 4))
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
    new_pairs = new_new_pairs
    new_new_data = np.zeros((0, 4))
    for i in range(len(new_data)):
        if matched_new_data[i] == 0:
            new_new_data = np.r_[new_new_data, [new_data[i]]]
    new_data = new_new_data
    # 5.剩余箭头对按欧式距离匹配
    # from math import sqrt
    right_matched_pairs = np.zeros((0, 8))
    x = len(new_pairs)
    while len(right_matched_pairs) != x and len(new_data) != 0 and len(new_pairs) != 0:
        matched_pairs = np.zeros((len(new_pairs)))
        matched_pairs_len = np.zeros((len(new_pairs)))
        matched_pairs[:] = -1  # 存储匹配到的data在new_data中的序号
        # 5.1.将所有pairs按照最近data匹配，记录匹配data序号和距离
        for i in range(len(new_pairs)):
            min_lenth = 99999
            min_no = -1
            for j in range(len(new_data)):
                lenth = sqrt(
                    (((new_pairs[i][2] + new_pairs[i][0]) * 0.5) - (new_data[j][2] + new_data[j][0]) * 0.5) ** 2 + (
                            ((new_pairs[i][3] + new_pairs[i][1]) * 0.5) - (
                            new_data[j][3] + new_data[j][1]) * 0.5) ** 2)
                if lenth < min_lenth:
                    min_lenth = lenth
                    min_no = j
            if min_no != -1:
                matched_pairs[i] = min_no
                matched_pairs_len[i] = min_lenth
        # 5.2.将相同匹配的pairs中距离大的项清零
        for i in range(len(matched_pairs)):
            if matched_pairs[i] != -1:
                for j in range(len(matched_pairs)):
                    if matched_pairs[j] != -1:
                        if i != j and matched_pairs[i] == matched_pairs[j]:
                            if matched_pairs_len[i] > matched_pairs_len[j]:
                                matched_pairs[i] = -1
                                matched_pairs_len[i] = 0
                            else:
                                matched_pairs[j] = -1
                                matched_pairs_len[j] = 0
        # 5.3.将未匹配的data和pairs分离重复1 2 3直到
        no_matched_pairs = np.zeros((0, 4))
        no_matched_data = np.zeros((0, 4))
        for i in range(len(new_pairs)):
            if matched_pairs[i] == -1:
                no_matched_pairs = np.r_[no_matched_pairs, [new_pairs[i]]]
        for i in range(len(new_data)):
            if i not in matched_pairs:
                no_matched_data = np.r_[no_matched_data, [new_data[i]]]

        middle = np.zeros((8))
        for i in range(len(new_pairs)):
            if matched_pairs[i] != -1:
                middle[0:4] = new_pairs[i]
                middle[4:8] = new_data[int(matched_pairs[i])]
                right_matched_pairs = np.r_[right_matched_pairs, [middle]]
        new_pairs = no_matched_pairs
        new_data = no_matched_data
    # 5.4将剩余data添加pairs为空传到最后结果
    middle = np.zeros((8))
    if len(new_data) != 0:
        for i in range(len(new_data)):
            middle[0:4] = np.array([0, 0, 0, 0])
            middle[4:8] = new_data[i]
            right_matched_pairs = np.r_[right_matched_pairs, [middle]]
    # 输出匹配
    for i in range(len(right_matched_pairs)):
        matched_pairs_data = np.r_[matched_pairs_data, [right_matched_pairs[i]]]
    # 补充最后的标注信息
    new_matched_pairs_data = np.array([['0', '0', '0', '0', '0', '0', '0', '0', '0']])
    middle = np.empty(9, dtype=np.dtype('U10'))
    for i in range(len(matched_pairs_data)):
        for j in range(len(data_copy)):
            mid = (matched_pairs_data[i][4: 8]).astype(str)
            if (mid == data_copy[j][0: 4]).all():
                middle[0: 8] = matched_pairs_data[i].astype(str)

                middle[8] = data_copy[j][4]

                try:
                    new_matched_pairs_data = np.r_[new_matched_pairs_data, [middle]]
                except:
                    print('aaa')
    matched_pairs_data = new_matched_pairs_data[1:, :]
    print("matched_pairs_data", matched_pairs_data)

    return matched_pairs_data


def io_1(ocr_data):
    """将 OCR 数据转换为表格字典结构的第一步。"""
    # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]
    '''
    result = np.zeros((0, 11))
    for i in range(len(ocr_data)):
        mid = np.zeros((11))
        if ocr_data[i]['matched_pairs_location'] != []:
            mid[0: 4] = ocr_data[i]['matched_pairs_location']
        else:
            mid[0: 4] = np.array([0, 0, 0, 0])
        mid[4: 8] = ocr_data[i]['location']
        if ocr_data[i]['max_medium_min'] != []:
            mid[8: 11] = ocr_data[i]['max_medium_min']
        else:
            mid[8: 11] = np.array([0, 0, 0])
        result = np.r_[result, [mid]]
    return result


def io_2(ocr_data):
    """在表格字典上继续整理层级的第二步。"""
    转换ocr_data为[['0','1','2','3','A1'],['0','1','2','3','B1']]格式
    '''
    dbnet_data = np.zeros((0, 5))
    for i in range(len(ocr_data)):
        str1 = np.empty(5, dtype=np.dtype('U10'))
        print(ocr_data[i]['location'])
        str1[0: 4] = ocr_data[i]['location'].astype(str)
        str1[4] = ocr_data[i]['ocr_strings']
        dbnet_data = np.r_[dbnet_data, [str1]]
    return dbnet_data


def io_3(table_dic):
    """完成表格字典最终结构调整的第三步。"""
    删除'data'中的空格
    table_dic中的max_medium_min从字符串列表转为数组
    '''
    for i in range(len(table_dic)):
        str_data = re.sub(" ", '', table_dic[i]['data'])
        table_dic[i]['data'] = str_data
    for i in range(len(table_dic)):
        m_m_m = []
        new_m = np.zeros(3)
        k = 0
        j = 0
        no = -1
        for strings in table_dic[i]['max_medium_min']:

            try:
                a = float(strings)
                j += 1
                no = k
            except:
                a = 0
                pass
            if a != 0:
                m_m_m.append(a)
            k += 1

        if len(m_m_m) == 3:
            m_m_m.sort()
            new_m[0] = m_m_m[2]
            new_m[1] = m_m_m[1]
            new_m[2] = m_m_m[0]

        if len(m_m_m) == 1:
            new_m[0] = m_m_m[0]
            new_m[1] = m_m_m[0]
            new_m[2] = m_m_m[0]
        if len(m_m_m) == 2:
            if m_m_m[0] > m_m_m[1]:
                new_m[0] = m_m_m[0]
                new_m[1] = round((m_m_m[1] + m_m_m[0]) * 0.5, 3)
                new_m[2] = m_m_m[1]
            else:
                new_m[0] = m_m_m[1]
                new_m[1] = round((m_m_m[1] + m_m_m[0]) * 0.5, 3)
                new_m[2] = m_m_m[0]
        table_dic[i]['max_medium_min'] = new_m
    return table_dic


def filter_dic_1(table_dic):
    """对表格字典执行第一阶段过滤逻辑。"""
    new_table = []
    print(table_dic)
    # 删除“bbb”,"字母"+"字母"
    for i in range(len(table_dic)):
        str_data = re.sub("bbb|aaa|ccc|ddd|eee|fff", '', table_dic[i]['data'])
        str_data = re.sub("[A-z][A-z]", '', str_data)
        table_dic[i]['data'] = str_data
    # 筛选data中字符串过长的
    max_len = 6
    for i in range(len(table_dic)):
        new_dic = {'data': [], 'max_medium_min': []}
        if len(table_dic[i]['data']) <= max_len:
            new_table.append(table_dic[i])
    table_dic = new_table
    table_n_x = table_dic.copy()  # 非“字母”+“数字”型和“字母”型从这找

    # 筛选出所有“字母”+“数字”型
    new_table = []
    for i in range(len(table_dic)):

        str_list = re.findall("[A-z][1-9]", table_dic[i]['data'])
        str_data = re.sub("[A-z][1-9]", '', table_dic[i]['data'])
        table_dic[i]['data'] = str_data
        try:
            for j in range(len(str_list)):
                new_dic = {'data': str_list[j], 'max_medium_min': table_dic[i]['max_medium_min']}
                new_table.append(new_dic)
        except:
            pass
    table_A1 = new_table  # “字母”+“数字”型从这里找
    # table_a = table_dic  # “字母”型从这里找
    # 筛选出所有“字母”型
    new_table = []
    for i in range(len(table_dic)):

        str_list = re.findall("[A-z]", table_dic[i]['data'])
        str_data = re.sub("[A-z]", '', table_dic[i]['data'])
        table_dic[i]['data'] = str_data
        try:
            for j in range(len(str_list)):
                new_dic = {'data': str_list[j], 'max_medium_min': table_dic[i]['max_medium_min']}
                new_table.append(new_dic)
        except:
            pass
    table_A = new_table  # “字母”型从这里找
    # 从“字母”+“数字”型中找想要的
    new_table = []
    for i in range(len(table_A1)):
        new_dic = {'data': [], 'max_medium_min': []}
        # 筛选信息
        str_data = re.sub("A1", '', table_A1[i]['data'])
        if str_data != table_A1[i]['data']:
            new_dic['data'] = 'A1'
            table_A1[i]['data'] = str_data
            new_dic['max_medium_min'] = table_A1[i]['max_medium_min']
            new_table.append(new_dic)
        str_data = re.sub("e1", '', table_A1[i]['data'])
        if str_data != table_A1[i]['data']:
            new_dic['data'] = 'e1'
            table_A1[i]['data'] = str_data
            new_dic['max_medium_min'] = table_A1[i]['max_medium_min']
            new_table.append(new_dic)
    table_A1 = new_table
    # 从“字母”型中找想要的
    new_table = []
    for i in range(len(table_A)):
        new_dic = {'data': [], 'max_medium_min': []}
        # 筛选信息
        str_data = re.sub("E", '', table_A[i]['data'])
        if str_data != table_A[i]['data']:
            new_dic['data'] = 'E'
            table_A[i]['data'] = str_data
            new_dic['max_medium_min'] = table_A[i]['max_medium_min']
            new_table.append(new_dic)
        str_data = re.sub("D", '', table_A[i]['data'])
        if str_data != table_A[i]['data']:
            new_dic['data'] = 'D'
            table_A[i]['data'] = str_data
            new_dic['max_medium_min'] = table_A[i]['max_medium_min']
            new_table.append(new_dic)
        str_data = re.sub("e", '', table_A[i]['data'])
        if str_data != table_A[i]['data']:
            new_dic['data'] = 'e'
            table_A[i]['data'] = str_data
            new_dic['max_medium_min'] = table_A[i]['max_medium_min']
            new_table.append(new_dic)
        str_data = re.sub("A", '', table_A[i]['data'])
        if str_data != table_A[i]['data']:
            new_dic['data'] = 'A'
            table_A[i]['data'] = str_data
            new_dic['max_medium_min'] = table_A[i]['max_medium_min']
            new_table.append(new_dic)
        str_data = re.sub("b", '', table_A[i]['data'])
        if str_data != table_A[i]['data']:
            new_dic['data'] = 'b'
            table_A[i]['data'] = str_data
            new_dic['max_medium_min'] = table_A[i]['max_medium_min']
            new_table.append(new_dic)
    table_A = new_table
    # 从非“字母”+“数字”型、“字母”型中找想要的
    new_table = []
    for i in range(len(table_n_x)):
        str_data = re.sub("n_x", '', table_n_x[i]['data'])
        if str_data != table_n_x[i]['data']:
            new_dic['data'] = 'n_x'
            table_n_x[i]['data'] = str_data
            new_dic['max_medium_min'] = table_n_x[i]['max_medium_min']
            new_table.append(new_dic)
        str_data = re.sub("n_y", '', table_n_x[i]['data'])
        if str_data != table_n_x[i]['data']:
            new_dic['data'] = 'n_y'
            table_n_x[i]['data'] = str_data
            new_dic['max_medium_min'] = table_n_x[i]['max_medium_min']
            new_table.append(new_dic)
    table_n_x = new_table
    # 合并
    new_table = table_A1 + table_A + table_n_x
    return new_table


def filter_dic_2(table_dic):
    """对表格字典执行第二阶段过滤逻辑。"""
    1.遇到','将字符串按此前后分为两份
    2.删除非数字部分
    3.删除英寸

    '''
    # 1.
    for i in range(len(table_dic)):
        new_m_list = []
        for j in range(len(table_dic[i]['max_medium_min'])):
            strings = table_dic[i]['max_medium_min'][j]
            strings = re.sub(',', ' ', strings)
            str_list = strings.split()
            new_m_list = new_m_list + str_list
        table_dic[i]['max_medium_min'] = new_m_list

    # 2.
    for i in range(len(table_dic)):
        new_m_list = []
        for j in range(len(table_dic[i]['max_medium_min'])):
            strings = table_dic[i]['max_medium_min'][j]
            str_list = re.findall("\d*\.?\d*", strings)
            new_m_list = new_m_list + str_list
        table_dic[i]['max_medium_min'] = new_m_list
    for i in range(len(table_dic)):
        new_m_list = []
        for j in range(len(table_dic[i]['max_medium_min'])):
            strings = table_dic[i]['max_medium_min'][j]
            if strings != '' and strings != '0' and strings != '0.':
                new_m_list.append(strings)
                new_m_list = list(set(new_m_list))  # 删除重复项
        table_dic[i]['max_medium_min'] = new_m_list

    # 3.
    for i in range(len(table_dic)):
        new_m_list = []
        for j in range(len(table_dic[i]['max_medium_min'])):
            key = 1
            strings_1 = table_dic[i]['max_medium_min'][j]
            for k in range(len(table_dic[i]['max_medium_min'])):
                strings_2 = table_dic[i]['max_medium_min'][k]
                if k != j:
                    try:
                        a = float(strings_1)
                    except:
                        a = 0
                    try:
                        b = float(strings_2)
                    except:
                        b = 0

                    if math.isclose(a * 25.4, b, rel_tol=1e-01, abs_tol=0.0):
                        key = 0
                        break
            if key == 1:
                new_m_list.append(strings_1)
        table_dic[i]['max_medium_min'] = new_m_list

    # # 4.
    # for i in range(len(table_dic)):
    #     new_m_list = []
    #     if len(table_dic[i]['max_medium_min']) == 1:
    #         new_m_list.append(table_dic[i]['max_medium_min'][0])
    #         new_m_list.append(table_dic[i]['max_medium_min'][0])
    #         new_m_list.append(table_dic[i]['max_medium_min'][0])
    #         table_dic[i]['max_medium_min'] = new_m_list
    #     if len(table_dic[i]['max_medium_min']) == 2:
    #         new_m_list.append(table_dic[i]['max_medium_min'][0])
    #         new_m_list.append(table_dic[i]['max_medium_min'][1])
    #         new_m_list.append('0')
    #         table_dic[i]['max_medium_min'] = new_m_list
    #     if len(table_dic[i]['max_medium_min']) == 0:
    #         new_m_list.append('0')
    #         new_m_list.append('0')
    #         new_m_list.append('0')
    #         table_dic[i]['max_medium_min'] = new_m_list
    return table_dic


def get_pairs_info(ocr_data, yolox_pairs_copy):
    """提取标尺对的坐标及文本信息。"""
    0 = outside 1 = inside
    '''
    for i in range(len(yolox_pairs_copy)):
        for j in range(len(ocr_data)):
            if ocr_data[j]['matched_pairs_location'] != []:
                if (yolox_pairs_copy[i][4] == 0):
                    ocr_data[j]['matched_pairs_outside_or_inside'] = 'outside'
                if (yolox_pairs_copy[i][4] == 1):
                    ocr_data[j]['matched_pairs_outside_or_inside'] = 'inside'
    return ocr_data


def get_yinxian_info(ocr_data, yolox_pairs_length):
    """提取引线相关的几何和文本信息。"""
    top_yolox_pairs_length np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
    '''
    for i in range(len(yolox_pairs_length)):
        for j in range(len(ocr_data)):
            if ocr_data[j]['matched_pairs_location'] != []:
                if (yolox_pairs_length[i][0: 4] == ocr_data[j]['matched_pairs_location']).all():
                    ocr_data[j]['matched_pairs_yinXian'] = yolox_pairs_length[4: 12]
    return ocr_data


def match_pairs_data(img_path, pairs, ocr):  # pairs[[0,1,2,3],[0,1,2,3]];data[[0,1,2,3,m,m,m],[0,1,2,3,m,m,m]]
    """综合标尺、文本和引线信息进行匹配。"""
    ocr= {'location': yolox_num[i], 'ocr_strings': '', 'key_info': [],
               'matched_pairs_location': [], 'matched_pairs_outside_or_inside': [],
               'matched_pairs_yinXian': [], 'Absolutely': [], 'max_medium_min': []}
    '''
    print("---开始视图的标注和标尺线的匹配---")
    # 设置最大匹配距离，超过这一距离无法匹配在一起
    w, h = get_img_info(img_path)
    max_length = round(min(w, h) * 0.5, 2)
    matched_ocr = []  # 匹配的尺寸线和尺寸数字存在这里
    middle_arr = np.zeros((11))  # 作为中间量存入matched_pairs_data
    matched_data = np.zeros((len(ocr)))  # 标记是否该data被匹配
    matched_pairs = np.zeros((len(pairs)))  # 标记是否该pairs被匹配
    # 1.直接匹配和标尺线重叠的标注，并从标注池中删除。完成后删除匹配到的标注，保证一个标注可以和多个重叠标注匹配
    for i in range(len(pairs)):
        for j in range(len(ocr)):
            if not (pairs[i][0] > ocr[j]['location'][2] or pairs[i][2] < ocr[j]['location'][0]):  # 两矩形在x坐标上的长有重叠
                if not (pairs[i][1] > ocr[j]['location'][3] or pairs[i][3] < ocr[j]['location'][1]):  # 两矩形在y坐标上的高有重叠
                    matched_data[j] = 1
                    matched_pairs[i] = 1
                    ocr[j]['matched_pairs_location'] = pairs[i]
                    matched_ocr.append(ocr[j])
                    print("匹配有重叠的标尺线与标注\n", ocr[j])

    ruler_match = matched_ocr.copy()
    # print(matched_pairs_data)
    new_ocr = []  # 过滤已经匹配了的数据的尺寸数字数组
    for i in range(len(ocr)):
        if matched_data[i] == 0:
            new_ocr.append(ocr[i])
    new_pairs = np.zeros((0, 4))
    for i in range(len(pairs)):
        if matched_pairs[i] == 0:
            new_pairs = np.r_[new_pairs, [pairs[i]]]
    # 2.针对没有重叠的pairs，（1）横向pair只能匹配横向data（2）竖向pair可能匹配横向或者竖向data
    # 将pairs和data重叠的作为标尺，其他匹配的pairs和data比例不能过大
    # 横向pairs 1.重叠的data 2.同方向且中心轴相同的离得近的data（紧贴） 3.横向水平且离得近（欧氏距离不超过两个pairs长）4.欧氏距离
    # 竖向pairs 1.重叠的data 2.同方向且中心轴相同的离得近的data（紧贴） 3.竖向水平且离得近（欧氏距离不超过两个pairs长）4.欧氏距离
    # 每次匹配三个data并排序，如果有条件差不多的data：找相近的pairs按照pairs长度分别将两个data匹配
    # 3.方法：同方向且中心轴相同的离得近的data（紧贴）#####pairs的方向按照长宽比判断，但方向区分度不大的pairs不能判断方向
    # 3.实现版本：找pairs中心轴穿过的data，根据水平距离排序再通过标尺筛选
    # 可以改进：根据ocr识别的图片判断data方向而不是根据data长宽
    # print('***', new_ocr)
    #####################################横向
    mid_arr = np.zeros((8))  # 前七位存储数据，第八位存储尺寸数据和尺寸线的最近距离
    matched_new_data = np.zeros((len(new_ocr)))  # 标记是否该data被匹配
    matched_new_pairs = np.zeros((len(new_pairs)))  # 标记是否该pairs被匹配
    for i in range(len(new_pairs)):
        # mid_ocr_acc = np.zeros((len(new_ocr)))  # 记录哪些是可能的
        maybe_match_data = np.zeros((0, 8))  # 存储可能的尺寸数字，前七位存储数据，第八位存储尺寸数据和尺寸线的最近距离
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] > new_pairs[i][3] - new_pairs[i][1]:  # 横向pairs
                for j in range(len(new_ocr)):
                    if new_ocr[j]['location'][2] - new_ocr[j]['location'][0] > new_ocr[j]['location'][3] - \
                            new_ocr[j]['location'][1]:  # 横向data
                        if (new_pairs[i][2] + new_pairs[i][0]) * 0.5 < new_ocr[j]['location'][2] and (
                                new_pairs[i][2] + new_pairs[i][0]) * 0.5 > new_ocr[j]['location'][
                            0]:  # 横向pairs中心轴穿过横向data
                            if (new_pairs[i][3] - new_pairs[i][1]) * 4 > min(
                                    abs(new_pairs[i][3] - new_ocr[j]['location'][1]), abs(
                                        new_pairs[i][1] - new_ocr[j]['location'][3])):  # pairs与data距离不超过pairs高度2倍
                                mid_arr[0:4] = new_ocr[j]['location']
                                mid_arr[4: 7] = new_ocr[j]['max_medium_min']
                                mid_arr[7] = min(abs(new_pairs[i][3] - new_ocr[j]['location'][1]),
                                                 abs(new_pairs[i][1] - new_ocr[j]['location'][3]))
                                maybe_match_data = np.r_[maybe_match_data, [mid_arr]]
                                # mid_ocr_acc[j] = len(maybe_match_data)
                maybe_match_data = maybe_match_data[np.argsort(maybe_match_data[:, 7])]  # 按距离从小到大排序
                # 当ruler中data比maybe中data大时，ruler中pairs一定比data中
                for k in range(len(maybe_match_data)):
                    buer = True
                    for l in range(len(ruler_match)):
                        if (max(ruler_match[l]['matched_pairs_location'][2] - ruler_match[l]['matched_pairs_location'][
                            0], ruler_match[l]['matched_pairs_location'][3] - ruler_match[l]['matched_pairs_location'][
                            1]) - max(
                            new_pairs[i][2] - new_pairs[i][0], new_pairs[i][3] - new_pairs[i][1])) * (
                                ruler_match[l]['max_medium_min'][0] - maybe_match_data[k][4]) < 0:
                            buer = False
                            break
                    if buer == True:
                        for m in range(len(new_ocr)):
                            if (new_ocr[m]['location'] == maybe_match_data[k][0:4]).all():
                                new_ocr[m]['matched_pairs_location'] = new_pairs[i]
                        print("匹配同方向且中心轴相同的离得近的标注（紧贴）\n", new_ocr[m])
                        matched_ocr.append(new_ocr[m])
                        matched_new_pairs[i] = 1
                        matched_new_data[m] = 1

    new_new_pairs = np.zeros((0, 4))
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
    new_pairs = new_new_pairs
    new_new_data = []
    for i in range(len(new_ocr)):
        if matched_new_data[i] == 0:
            new_new_data.append(new_ocr[i])
    new_ocr = new_new_data
    # print('***', new_ocr)
    ###########################竖向
    mid_arr = np.zeros((8))
    matched_new_data = np.zeros((len(new_ocr)))
    matched_new_pairs = np.zeros((len(new_pairs)))
    for i in range(len(new_pairs)):
        # mid_ocr_acc = np.zeros((len(new_ocr)))  # 记录哪些是可能的
        maybe_match_data = np.zeros((0, 8))
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] < new_pairs[i][3] - new_pairs[i][1]:  # 竖向pairs
                for j in range(len(new_ocr)):
                    if new_ocr[j]['location'][2] - new_ocr[j]['location'][0] < new_ocr[j]['location'][3] - \
                            new_ocr[j]['location'][1]:  # 竖向data
                        if (new_pairs[i][3] + new_pairs[i][1]) * 0.5 < new_ocr[j]['location'][3] and (
                                new_pairs[i][3] + new_pairs[i][1]) * 0.5 > new_ocr[j]['location'][
                            1]:  # 竖向pairs中心轴穿过竖向data
                            if (new_pairs[i][2] - new_pairs[i][0]) * 2 > min(
                                    abs(new_pairs[i][2] - new_ocr[j]['location'][0]), abs(
                                        new_pairs[i][0] - new_ocr[j]['location'][2])):  # pairs与data距离不超过pairs高度2倍
                                mid_arr[0:4] = new_ocr[j]['location']
                                mid_arr[4: 7] = new_ocr[j]['max_medium_min']
                                mid_arr[7] = min(abs(new_pairs[i][3] - new_ocr[j]['location'][1]),
                                                 abs(new_pairs[i][1] - new_ocr[j]['location'][3]))
                                maybe_match_data = np.r_[maybe_match_data, [mid_arr]]
                                # mid_ocr_acc[j] = len(maybe_match_data)
                maybe_match_data = maybe_match_data[np.argsort(maybe_match_data[:, 7])]  # 按距离从小到大排序
                # 当ruler中data比maybe中data大时，ruler中pairs一定比data中
                for k in range(len(maybe_match_data)):
                    buer = True
                    for l in range(len(ruler_match)):
                        if (max(ruler_match[l]['matched_pairs_location'][2] - ruler_match[l]['matched_pairs_location'][
                            0], ruler_match[l]['matched_pairs_location'][3] - ruler_match[l]['matched_pairs_location'][
                            1]) - max(
                            new_pairs[i][2] - new_pairs[i][0], new_pairs[i][3] - new_pairs[i][1])) * (
                                ruler_match[l]['max_medium_min'][0] - maybe_match_data[k][4]) < 0:
                            buer = False
                            break
                    if buer == True:
                        for m in range(len(new_ocr)):
                            if (new_ocr[m]['location'] == maybe_match_data[k][0:4]).all():
                                new_ocr[m]['matched_pairs_location'] = new_pairs[i]
                        print("匹配同方向且中心轴相同的离得近的data（紧贴）\n", new_ocr[m])
                        matched_ocr.append(new_ocr[m])
                        matched_new_pairs[i] = 1
                        matched_new_data[m] = 1
    new_new_pairs = np.zeros((0, 4))
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
    new_pairs = new_new_pairs
    new_new_data = []
    for i in range(len(new_ocr)):
        if matched_new_data[i] == 0:
            new_new_data.append(new_ocr[i])
    new_ocr = new_new_data
    # print('***', new_ocr)
    # 4.1横向箭头横向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）（横向箭头一般只会匹配横向数据）
    # 实际：横向箭头匹配数据（不管方向）：先在pairs的y坐标上有重叠的data填入待match，再将pairs高度扩充三倍，在y轴上与之相交的data按欧氏距离大小排列为待匹配，用ruler检验
    matched_new_pairs = np.zeros((len(new_pairs)))
    matched_new_data = np.zeros((len(new_ocr)))
    for i in range(len(new_pairs)):
        # mid_ocr_acc = np.zeros((len(new_ocr)))  # 记录哪些是可能的
        maybe_match_data_a = np.zeros((0, 7))
        maybe_match_data_b = np.zeros((0, 8))
        middle = np.zeros((8))
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] > new_pairs[i][3] - new_pairs[i][1]:  # 横向pairs
                for j in range(len(new_ocr)):

                    if not (new_pairs[i][1] > new_ocr[j]['location'][3] or new_pairs[i][3] < new_ocr[j]['location'][
                        1]):  # 两矩形在y坐标上的高有重叠
                        if min(abs(new_ocr[j]['location'][0] - new_pairs[i][2]),
                               abs(new_ocr[j]['location'][2] - new_pairs[i][0])) < 5 * (
                                new_pairs[i][3] - new_pairs[i][1]):  # pairs和data横向距离很近
                            mid = np.zeros(7)
                            mid[0: 4] = new_ocr[j]['location']
                            mid[4: 7] = new_ocr[j]['max_medium_min']
                            maybe_match_data_a = np.r_[maybe_match_data_a, [mid]]
                            # mid_ocr_acc[j] = len(maybe_match_data_a)
                    if not (new_pairs[i][1] - (new_pairs[i][3] - new_pairs[i][1]) * 2 > new_ocr[j]['location'][3] or
                            new_pairs[i][
                                3] + (new_pairs[i][3] - new_pairs[i][1]) * 2 < new_ocr[j]['location'][
                                1]):  # 两矩形在y坐标上的高有重叠
                        if min(abs(new_ocr[j]['location'][0] - new_pairs[i][2]),
                               abs(new_ocr[j]['location'][2] - new_pairs[i][0])) < 7 * (
                                new_pairs[i][3] - new_pairs[i][1]):
                            if new_ocr[j]['location'] not in maybe_match_data_a[0: 4]:
                                middle[0:4] = new_ocr[j]['location']
                                middle[4:7] = new_ocr[j]['max_medium_min']
                                middle[7] = min(abs(new_ocr[j]['location'][0] - new_pairs[i][2]),
                                                abs(new_ocr[j]['location'][2] - new_pairs[i][0]))
                                maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                maybe_match_data_b = maybe_match_data_b[np.argsort(maybe_match_data_b[:, 7])]  # 按距离从小到大排序
                maybe_match_data_b = maybe_match_data_b[:, 0:7]
                maybe_match_data = np.append(maybe_match_data_a, maybe_match_data_b, axis=0)
                for k in range(len(maybe_match_data)):
                    buer = True
                    for l in range(len(ruler_match)):
                        if (max(ruler_match[l]['matched_pairs_location'][2] - ruler_match[l]['matched_pairs_location'][
                            0], ruler_match[l]['matched_pairs_location'][3] - ruler_match[l]['matched_pairs_location'][
                            1]) - max(
                            new_pairs[i][2] - new_pairs[i][0], new_pairs[i][3] - new_pairs[i][1])) * (
                                ruler_match[l]['max_medium_min'][0] - maybe_match_data[k][4]) < 0:
                            buer = False
                            break
                    if buer == True:
                        for m in range(len(new_ocr)):
                            if (new_ocr[m]['location'] == maybe_match_data[k][0:4]).all():
                                new_ocr[m]['matched_pairs_location'] = new_pairs[i]
                        print(
                            "横向箭头横向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）（横向箭头一般只会匹配横向数据）\n",
                            new_ocr[m])
                        matched_ocr.append(new_ocr[m])
                        matched_new_pairs[i] = 1
                        matched_new_data[m] = 1
    new_new_pairs = np.zeros((0, 4))
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
    new_pairs = new_new_pairs
    new_new_data = []
    for i in range(len(new_ocr)):
        if matched_new_data[i] == 0:
            new_new_data.append(new_ocr[i])
    new_ocr = new_new_data
    # print('***', new_ocr)
    # print("new_pairs,new_data",new_pairs,new_data)
    # print("matched_pairs_data",matched_pairs_data)
    # 4.2竖向箭头竖向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）（竖向箭头可能匹配到两个方向的数据）
    # 实际：竖向箭头匹配数据（不管方向）：先在pairs的x坐标上有重叠的data填入待match，再将pairs宽度扩充三倍，在x轴上与之相交的data按欧氏距离大小排列为待匹配，用ruler检验
    matched_new_pairs = np.zeros((len(new_pairs)))
    matched_new_data = np.zeros((len(new_ocr)))
    for i in range(len(new_pairs)):
        maybe_match_data_a = np.zeros((0, 7))
        maybe_match_data_b = np.zeros((0, 8))
        middle = np.zeros((8))
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] < new_pairs[i][3] - new_pairs[i][1]:  # 竖向pairs
                for j in range(len(new_ocr)):
                    if not (pairs[i][0] > new_ocr[j]['location'][2] or pairs[i][2] < new_ocr[j]['location'][
                        0]):  # 两矩形在x坐标上的长有重叠
                        if min(abs(new_ocr[j]['location'][1] - new_pairs[i][3]),
                               abs(new_ocr[j]['location'][3] - new_pairs[i][1])) < 5 * (
                                new_pairs[i][2] - new_pairs[i][0]):  # pairs和data横向距离很近
                            mid = np.zeros(7)
                            mid[0: 4] = new_ocr[j]['location']
                            mid[4: 7] = new_ocr[j]['max_medium_min']
                            maybe_match_data_a = np.r_[maybe_match_data_a, [mid]]
                    if not (new_pairs[i][0] - (new_pairs[i][2] - new_pairs[i][0]) * 2 > new_ocr[j]['location'][2] or
                            new_pairs[i][
                                2] + (new_pairs[i][2] - new_pairs[i][0]) * 2 < new_ocr[j]['location'][
                                0]):  # 两矩形在y坐标上的高有重叠
                        if min(abs(new_ocr[j]['location'][1] - new_pairs[i][3]),
                               abs(new_ocr[j]['location'][3] - new_pairs[i][1])) < 7 * (
                                new_pairs[i][2] - new_pairs[i][0]):
                            if new_ocr[j]['location'] not in maybe_match_data_a[0: 4]:
                                middle[0:4] = new_ocr[j]['location']
                                middle[4:7] = new_ocr[j]['max_medium_min']
                                middle[7] = min(abs(new_ocr[j]['location'][1] - new_pairs[i][3]),
                                                abs(new_ocr[j]['location'][3] - new_pairs[i][1]))
                                maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                maybe_match_data_b = maybe_match_data_b[np.argsort(maybe_match_data_b[:, 7])]  # 按距离从小到大排序
                maybe_match_data_b = maybe_match_data_b[:, 0:7]
                maybe_match_data = np.append(maybe_match_data_a, maybe_match_data_b, axis=0)
                for k in range(len(maybe_match_data)):
                    buer = True
                    for l in range(len(ruler_match)):
                        if (max(ruler_match[l]['matched_pairs_location'][2] - ruler_match[l]['matched_pairs_location'][
                            0], ruler_match[l]['matched_pairs_location'][3] - ruler_match[l]['matched_pairs_location'][
                            1]) - max(
                            new_pairs[i][2] - new_pairs[i][0], new_pairs[i][3] - new_pairs[i][1])) * (
                                ruler_match[l]['max_medium_min'][0] - maybe_match_data[k][4]) < 0:
                            buer = False
                            break
                    if buer == True:
                        for m in range(len(new_ocr)):
                            if (new_ocr[m]['location'] == maybe_match_data[k][0:4]).all():
                                new_ocr[m]['matched_pairs_location'] = new_pairs[i]
                        print(
                            "竖向箭头竖向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）\n",
                            new_ocr[m])
                        matched_ocr.append(new_ocr[m])
                        matched_new_pairs[i] = 1
                        matched_new_data[m] = 1
    new_new_pairs = np.zeros((0, 4))
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
    new_pairs = new_new_pairs
    new_new_data = []
    for i in range(len(new_ocr)):
        if matched_new_data[i] == 0:
            new_new_data.append(new_ocr[i])
    new_ocr = new_new_data
    # print('***', new_ocr)
    # 5.剩余标尺线按欧式距离匹配
    # from math import sqrt
    right_matched_pairs = []
    x = len(new_pairs)
    while_count = 0
    while len(right_matched_pairs) != x and len(new_ocr) != 0 and len(new_pairs) != 0:
        matched_pairs = np.zeros((len(new_pairs)))
        matched_pairs_len = np.zeros((len(new_pairs)))
        matched_pairs[:] = -1  # 存储匹配到的data在new_data中的序号
        # 5.1.将所有pairs按照最近data匹配，记录匹配data序号和距离
        for i in range(len(new_pairs)):
            min_lenth = 99999
            min_no = -1
            for j in range(len(new_ocr)):
                lenth = sqrt(
                    (((new_pairs[i][2] + new_pairs[i][0]) * 0.5) - (
                            new_ocr[j]['location'][2] + new_ocr[j]['location'][0]) * 0.5) ** 2 + (
                            ((new_pairs[i][3] + new_pairs[i][1]) * 0.5) - (
                            new_ocr[j]['location'][3] + new_ocr[j]['location'][1]) * 0.5) ** 2)
                if lenth < min_lenth:
                    min_lenth = lenth
                    min_no = j
            if min_no != -1 and min_lenth < max_length:
                matched_pairs[i] = min_no
                matched_pairs_len[i] = min_lenth
        # 5.2.将相同匹配的pairs中距离大的项清零
        for i in range(len(matched_pairs)):
            if matched_pairs[i] != -1:
                for j in range(len(matched_pairs)):
                    if matched_pairs[j] != -1:
                        if i != j and matched_pairs[i] == matched_pairs[j]:
                            if matched_pairs_len[i] > matched_pairs_len[j]:
                                matched_pairs[i] = -1
                                matched_pairs_len[i] = 0
                            else:
                                matched_pairs[j] = -1
                                matched_pairs_len[j] = 0
        # 5.3.将未匹配的data和pairs分离重复1 2 3直到
        no_matched_pairs = np.zeros((0, 4))
        no_matched_data = []
        for i in range(len(new_pairs)):
            if matched_pairs[i] == -1:
                no_matched_pairs = np.r_[no_matched_pairs, [new_pairs[i]]]
        for i in range(len(new_ocr)):
            if i not in matched_pairs:
                no_matched_data.append(new_ocr[i])

        middle = np.zeros((11))
        for i in range(len(new_pairs)):
            if matched_pairs[i] != -1:
                new_ocr[int(matched_pairs[i])]['matched_pairs_location'] = new_pairs[i]
                right_matched_pairs.append(new_ocr[int(matched_pairs[i])])
        new_pairs = no_matched_pairs
        new_ocr = no_matched_data
        # 限定循环次数，防止死循环（不保证所有标尺线会匹配到标注）
        while_count += 1
        if while_count == 3:
            break
    # 5.4将剩余data添加pairs为空传到最后结果
    if len(new_ocr) != 0:
        for i in range(len(new_ocr)):
            right_matched_pairs.append(new_ocr[i])
    # 5.5将剩余pairs添加data为空传到最后结果
    # middle = np.zeros((11))
    # if len(new_pairs) != 0:
    #     for i in range(len(new_pairs)):
    #         middle[0: 4] = new_pairs[i]
    #         middle[4: 11] = np.array([0, 0, 0, 0, 0, 0, 0])
    #         right_matched_pairs = np.r_[right_matched_pairs, [middle]]
    # 输出匹配
    for i in range(len(right_matched_pairs)):
        matched_ocr.append(right_matched_pairs[i])

    result = []
    for i in range(len(matched_ocr)):
        if i == 0:
            result.append(matched_ocr[i])
            continue
        bool = True
        for j in range(len(result)):
            if operator.eq(matched_ocr[i]['location'], result[j]['location']).all():
                bool = False
        if bool:
            result.append(matched_ocr[i])
    print("---结束视图的标注和标尺线的匹配---")
    return result


def show_matched_pairs_data(img_path, pairs_data):
    """以图像形式展示匹配后的标尺数据。"""
    wh_key1 = True
    while wh_key1:
        auto_key = input("是否展示匹配好的pairs_data:y/n:")

        if auto_key == 'y' or auto_key == 'Y':
            auto_bool = False
            wh_key1 = False
        elif auto_key == 'n' or auto_key == 'N':
            auto_bool = True
            wh_key1 = False
        else:
            print("未输入正确，请重新输入：y/n:")
            wh_key1 = True
    if auto_bool == False:
        # import numpy as np
        # import cv2 as cv

        for i in range(len(pairs_data)):
            with open(img_path, 'rb') as f:
                np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 以彩图读取
                # img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取
            # 矩形左上角和右上角的坐标，绘制一个绿色矩形
            if pairs_data[i]['matched_pairs_location'] != []:
                ptLeftTop = (
                    int(pairs_data[i]['matched_pairs_location'][0]), int(pairs_data[i]['matched_pairs_location'][1]))
                ptRightBottom = (
                    int(pairs_data[i]['matched_pairs_location'][2]), int(pairs_data[i]['matched_pairs_location'][3]))
                point_color = (0, 255, 0)  # BGR
                thickness = 2
                lineType = 4
                cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

            # 绘制一个红色矩形
            ptLeftTop = (int(pairs_data[i]['location'][0]), int(pairs_data[i]['location'][1]))
            ptRightBottom = (int(pairs_data[i]['location'][2]), int(pairs_data[i]['location'][3]))
            point_color = (0, 0, 255)  # BGR
            thickness = 2
            lineType = 8
            cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

            cv2.namedWindow("pairs(green)_data(red)", 0)
            cv2.imshow('pairs(green)_data(red)', img)
            cv2.waitKey(0)  # 显示 10000 ms 即 10s 后消失
            cv2.destroyAllWindows()


def show_matched_pairs_data_table(img_path, pairs_data):
    """生成表格展示匹配后的标尺数据。"""
    pairs_data = (pairs_data[:, 0: 8]).astype(float)

    wh_key1 = True
    while wh_key1:
        auto_key = input("是否展示匹配好的pairs_data:y/n:")

        if auto_key == 'y' or auto_key == 'Y':
            auto_bool = False
            wh_key1 = False
        elif auto_key == 'n' or auto_key == 'N':
            auto_bool = True
            wh_key1 = False
        else:
            print("未输入正确，请重新输入：y/n:")
            wh_key1 = True
    if auto_bool == False:
        # import numpy as np
        # import cv2 as cv

        for i in range(len(pairs_data)):
            with open(img_path, 'rb') as f:
                np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 以彩图读取
                # img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取
            # 矩形左上角和右上角的坐标，绘制一个绿色矩形
            ptLeftTop = (int(pairs_data[i][0]), int(pairs_data[i][1]))
            ptRightBottom = (int(pairs_data[i][2]), int(pairs_data[i][3]))
            point_color = (0, 255, 0)  # BGR
            thickness = 2
            lineType = 4
            cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

            # 绘制一个红色矩形
            ptLeftTop = (int(pairs_data[i][4]), int(pairs_data[i][5]))
            ptRightBottom = (int(pairs_data[i][6]), int(pairs_data[i][7]))
            point_color = (0, 0, 255)  # BGR
            thickness = 2
            lineType = 8
            cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

            cv2.imshow('pairs(green)_data(red)', img)
            cv2.waitKey(0)  # 显示 10000 ms 即 10s 后消失
            cv2.destroyAllWindows()


def BGA_side_filter(side_ocr_data):
    """过滤 BGA 侧视图中过度冗余的 OCR 信息。"""
    # 1.side中有用的尺寸数字应该小于side_max_limate
    new_side_ocr_data = []
    side_max_limate = 15
    for i in range(len(side_ocr_data)):
        try:
            if (side_ocr_data[i]['max_medium_min'] < side_max_limate).all():
                new_side_ocr_data.append(side_ocr_data[i])
        except:
            a = 0
    return new_side_ocr_data


def filter_dbnet(dbnet_data):
    """根据规则过滤 DBNet 的文本框结果。"""
    x_l = np.zeros((len(dbnet_data)))
    y_l = np.zeros((len(dbnet_data)))
    new_dbnet_data = np.zeros((0, 4))
    for i in range(len(dbnet_data)):
        x_l[i] = dbnet_data[i][2] - dbnet_data[i][0]
        y_l[i] = dbnet_data[i][3] - dbnet_data[i][1]
        if x_l[i] > 40 or y_l[i] > 40:
            new_dbnet_data = np.r_[new_dbnet_data, [dbnet_data[i]]]

    return new_dbnet_data


def filter_bottom_dbnet(dbnet_data):
    """针对底部视图的 DBNet 文本框进行二次筛选。"""
    pin_map_limation = get_np_array_in_txt('yolox_data/pin_map_limation.txt')
    # a = np.array([[1, 1, 1, 1]])
    # if (pin_map_limation != a).any():
    print(pin_map_limation)
    new_dbnet_data = np.zeros((0, 4))
    for j in range(len(dbnet_data)):
        if (pin_map_limation[0][0] > dbnet_data[j][2] or pin_map_limation[0][2] < dbnet_data[j][0]) or (
                pin_map_limation[0][1] > dbnet_data[j][3] or pin_map_limation[0][3] < dbnet_data[j][
            1]):  # 两矩形在x坐标上的长无重叠 或者 两矩形在y坐标上的高无重叠
            new_dbnet_data = np.r_[new_dbnet_data, [dbnet_data[j]]]
    dbnet_data = new_dbnet_data
    # pin_map_limation = a
    # np.savetxt('yolox_data/pin_map_limation.txt', pin_map_limation, delimiter=',')
    return dbnet_data


def find_pairs_length(img_path, pairs, test_mode):
    """通过线段长度估计标尺对的实际尺寸。"""
    功能：检测标尺线附近成对的引线
    pairs np.二维数组[x1,y1,x2,y2,0 = outside 1 = inside]
    img_path str
    '''
    print("***/开始引线和标尺线的匹配/***")
    # 1.根据pinmap所在位置推测出大概十字线坐标
    pin_map_limation = get_np_array_in_txt(f'{YOLOX_DATA}/pin_map_limation.txt')
    w, h = get_img_info(img_path)
    a = np.array(([1, 1, 1, 1]))
    if (pin_map_limation != a).all():
        heng = (pin_map_limation[0][3] + pin_map_limation[0][1]) * 0.5
        shu = (pin_map_limation[0][2] + pin_map_limation[0][0]) * 0.5
    if img_path == f'{DATA}/side.jpg' or img_path == f'{DATA}/top.jpg':
        # w, h = get_img_info(img_path)
        heng = h / 2
        shu = w / 2

    ver_lines_heng, ver_lines_shu = find_all_lines(img_path, test_mode)

    pairs_length = np.zeros((0, 13))  # 存储pairs以及所表示的距离
    pairs_length_middle = np.zeros(13)

    # 横向直线的坐标排列np.二维数组[x1,y1,x2,y2]改为x1<x2 y1<y2
    new_ver_lines = np.zeros((0, 4))
    for i in range(len(ver_lines_heng)):
        if ver_lines_heng[i][0] > ver_lines_heng[i][2]:
            c = ver_lines_heng[i][0]
            ver_lines_heng[i][0] = ver_lines_heng[i][2]
            ver_lines_heng[i][2] = c
        if ver_lines_heng[i][1] > ver_lines_heng[i][3]:
            c = ver_lines_heng[i][1]
            ver_lines_heng[i][1] = ver_lines_heng[i][3]
            ver_lines_heng[i][3] = c
    # 滤除较短的直线
    min_length = 10  # 最短直线长
    for i in range(len(ver_lines_heng)):
        if max(abs(ver_lines_heng[i][2] - ver_lines_heng[i][0]),
               abs(ver_lines_heng[i][3] - ver_lines_heng[i][1])) > min_length:
            new_ver_lines = np.r_[new_ver_lines, [ver_lines_heng[i]]]
    ver_lines_heng = new_ver_lines
    min_length = 20  # 最短直线长
    new_ver_lines = np.zeros((0, 4))
    ver_lines_shu = np.array(ver_lines_shu)
    # ver_lines_shu = ver_lines_shu.reshape(ver_lines_shu.shape[0], -1)
    for i in range(len(ver_lines_shu)):
        if max(abs(ver_lines_shu[i][2] - ver_lines_shu[i][0]),
               abs(ver_lines_shu[i][3] - ver_lines_shu[i][1])) > min_length:
            new_ver_lines = np.r_[new_ver_lines, [ver_lines_shu[i]]]
    ver_lines_shu = new_ver_lines
    # print("len(ver_lines_heng)", len(ver_lines_heng))
    ratio = 0.4
    ra = 2
    print("开始视图**")
    for i in range(len(pairs)):
        print("一组pairs开始*")
        if pairs[i][4] == 0:  # 外向标尺线
            print("外向")
            ratio = 0.15
            if (pairs[i][2] - pairs[i][0]) > (pairs[i][3] - pairs[i][1]):  # 横向标尺线
                print("横向")
                left_straight = np.zeros((0, 5))  # 存储可能匹配的左侧直线，第五位是与标尺线左端点的距离
                right_straight = np.zeros((0, 5))  # 存储可能匹配的右侧直线，第五位是与标尺线右端点的距离
                middle = np.zeros((5))
                for j in range(len(ver_lines_shu)):
                    # 1.找左端附近的直线，要求横坐标在端点附近，纵坐标穿过或者在标尺线附近
                    if pairs[i][0] - ratio * (pairs[i][2] - pairs[i][0]) < ver_lines_shu[j][0] < pairs[i][0] + ratio * (
                            pairs[i][2] - pairs[i][0]):  # 横坐标在端点附近
                        if (not (pairs[i][1] > ver_lines_shu[j][3] or pairs[i][3] < ver_lines_shu[j][1])) or min(
                                abs(pairs[i][1] - ver_lines_shu[j][3]), abs(pairs[i][3] - ver_lines_shu[j][1])) < ra * (
                                pairs[i][3] - pairs[i][1]):  # 两矩形在y坐标上的高有重叠或者距离近
                            if (pairs[i][1] < heng and ver_lines_shu[j][3] > pairs[i][3]) or (
                                    pairs[i][1] > heng and ver_lines_shu[j][1] < pairs[i][
                                1]):  # 要求横pairs在图片上方时，匹配的直线在横pairs下方
                                middle[0:4] = ver_lines_shu[j]
                                middle[4] = abs(pairs[i][0] - ver_lines_shu[j][0])
                                left_straight = np.r_[left_straight, [middle]]
                                print("外向横pairs找到左竖线")

                    # 2.找右端附近的直线，要求横坐标在端点附近，纵坐标穿过或者在标尺线附近
                    if pairs[i][2] - ratio * (pairs[i][2] - pairs[i][0]) < ver_lines_shu[j][0] < pairs[i][2] + ratio * (
                            pairs[i][2] - pairs[i][0]):  # 横坐标在端点附近
                        if (not (pairs[i][1] > ver_lines_shu[j][3] or pairs[i][3] < ver_lines_shu[j][1])) or min(
                                abs(pairs[i][1] - ver_lines_shu[j][3]), abs(pairs[i][3] - ver_lines_shu[j][1])) < ra * (
                                pairs[i][3] - pairs[i][1]):  # 两矩形在y坐标上的高有重叠或者距离近
                            if (pairs[i][1] < heng and ver_lines_shu[j][3] > pairs[i][3]) or (
                                    pairs[i][1] > heng and ver_lines_shu[j][1] < pairs[i][
                                1]):  # 要求横pairs在图片上方时，匹配的直线在横pairs下方
                                middle[0:4] = ver_lines_shu[j]
                                middle[4] = abs(pairs[i][0] - ver_lines_shu[j][0])
                                right_straight = np.r_[right_straight, [middle]]
                                print("外向横pairs找到右竖线")
                left_straight = left_straight[np.argsort(left_straight[:, 4])]  # 按距离从小到大排序
                right_straight = right_straight[np.argsort(right_straight[:, 4])]  # 按距离从小到大排序
                if len(left_straight) > 0 and len(right_straight) > 0:
                    pairs_length_middle[0:4] = pairs[i, 0:4]
                    pairs_length_middle[4:8] = left_straight[0, 0:4]
                    pairs_length_middle[8:12] = right_straight[0, 0:4]
                    pairs_length_middle[12] = abs(left_straight[0, 0] - right_straight[0, 0])
                    pairs_length = np.r_[pairs_length, [pairs_length_middle]]
            if (pairs[i][2] - pairs[i][0]) < (pairs[i][3] - pairs[i][1]):  # 竖向标尺线
                print("竖向")
                up_straight = np.zeros((0, 5))  # 存储可能匹配的上侧直线，第五位是与标尺线上端点的距离
                down_straight = np.zeros((0, 5))  # 存储可能匹配的下侧直线，第五位是与标尺线下端点的距离
                middle = np.zeros((5))
                for j in range(len(ver_lines_heng)):
                    # 1.找上端附近的直线，要求纵坐标在端点附近，横坐标穿过或者在标尺线附近
                    if pairs[i][1] - ratio * (pairs[i][3] - pairs[i][1]) < ver_lines_heng[j][1] < pairs[i][
                        1] + ratio * (
                            pairs[i][3] - pairs[i][1]):  # 纵坐标在端点附近
                        if (not (pairs[i][0] > ver_lines_heng[j][2] or pairs[i][2] < ver_lines_heng[j][0])) or min(
                                abs(pairs[i][0] - ver_lines_heng[j][2]),
                                abs(pairs[i][2] - ver_lines_heng[j][0])) < ra * (
                                pairs[i][2] - pairs[i][0]):  # 两矩形在x坐标上的高有重叠或者距离近
                            if (pairs[i][0] > shu and ver_lines_heng[j][0] < pairs[i][0]) or (
                                    pairs[i][0] < shu and ver_lines_heng[j][2] > pairs[i][
                                2]):  # 要求竖pairs在图片左方时，匹配的直线在竖pairs右方
                                middle[0:4] = ver_lines_heng[j]
                                middle[4] = abs(pairs[i][1] - ver_lines_heng[j][1])
                                up_straight = np.r_[up_straight, [middle]]
                                print("外向竖pairs找到上横线")

                    # 2.找下端附近的直线，要求纵坐标在端点附近，横坐标穿过或者在标尺线附近
                    if pairs[i][3] - ratio * (pairs[i][3] - pairs[i][1]) < ver_lines_heng[j][1] < pairs[i][
                        3] + ratio * (
                            pairs[i][3] - pairs[i][1]):  # 横坐标在端点附近
                        if (not (pairs[i][0] > ver_lines_heng[j][2] or pairs[i][2] < ver_lines_heng[j][0])) or min(
                                abs(pairs[i][0] - ver_lines_heng[j][2]),
                                abs(pairs[i][2] - ver_lines_heng[j][0])) < ra * (
                                pairs[i][2] - pairs[i][0]):  # 两矩形在y坐标上的高有重叠或者距离近
                            if (pairs[i][0] > shu and ver_lines_heng[j][0] < pairs[i][0]) or (
                                    pairs[i][0] < shu and ver_lines_heng[j][2] > pairs[i][
                                2]):  # 要求竖pairs在图片左方时，匹配的直线在竖pairs右方
                                middle[0:4] = ver_lines_heng[j]
                                middle[4] = abs(pairs[i][0] - ver_lines_heng[j][0])
                                down_straight = np.r_[down_straight, [middle]]
                                print("外向竖pairs找到下横线")
                up_straight = up_straight[np.argsort(up_straight[:, 4])]  # 按距离从小到大排序
                down_straight = down_straight[np.argsort(down_straight[:, 4])]  # 按距离从小到大排序
                if len(up_straight) > 0 and len(down_straight) > 0:
                    pairs_length_middle[0:4] = pairs[i, 0:4]
                    pairs_length_middle[4:8] = up_straight[0, 0:4]
                    pairs_length_middle[8:12] = down_straight[0, 0:4]
                    pairs_length_middle[12] = abs(up_straight[0, 0] - down_straight[0, 0])
                    pairs_length = np.r_[pairs_length, [pairs_length_middle]]
        if pairs[i][4] == 1:  # 内向标尺线
            # 内向标尺线的引线一定离yolox检测出的两端点有一定距离
            print("内向")
            ratio = 0.5
            ratio_inside = 0.15
            if (pairs[i][2] - pairs[i][0]) > (pairs[i][3] - pairs[i][1]):  # 横向标尺线
                print("横向")
                left_straight = np.zeros((0, 5))  # 存储可能匹配的左侧直线，第五位是与标尺线左端点的距离
                right_straight = np.zeros((0, 5))  # 存储可能匹配的右侧直线，第五位是与标尺线右端点的距离
                middle = np.zeros((5))
                for j in range(len(ver_lines_shu)):
                    # 1.找左端附近的直线，要求横坐标在端点附近，纵坐标穿过或者在标尺线附近
                    if pairs[i][0] + ratio_inside * (pairs[i][2] - pairs[i][0]) < ver_lines_shu[j][0] < pairs[i][
                        0] + ratio * (
                            pairs[i][2] - pairs[i][0]):  # 直线横坐标在左端点右侧
                        if (not (pairs[i][1] > ver_lines_shu[j][3] or pairs[i][3] < ver_lines_shu[j][1])) or min(
                                abs(pairs[i][1] - ver_lines_shu[j][3]), abs(pairs[i][3] - ver_lines_shu[j][1])) < ra * (
                                pairs[i][3] - pairs[i][1]):  # 两矩形在y坐标上的高有重叠或者距离近
                            if (pairs[i][1] < heng and ver_lines_shu[j][3] > pairs[i][3]) or (
                                    pairs[i][1] > heng and ver_lines_shu[j][1] < pairs[i][
                                1]):  # 要求横pairs在图片上方时，匹配的直线在横pairs下方
                                middle[0:4] = ver_lines_shu[j]
                                middle[4] = abs(pairs[i][0] - ver_lines_shu[j][0])
                                left_straight = np.r_[left_straight, [middle]]
                                print("内向横pairs找到左竖线")

                    # 2.找右端附近的直线，要求横坐标在端点附近，纵坐标穿过或者在标尺线附近
                    if pairs[i][2] - ratio * (pairs[i][2] - pairs[i][0]) < ver_lines_shu[j][0] < pairs[i][
                        2] - ratio_inside * (pairs[i][2] - pairs[i][0]):  # 横坐标在右端点左侧
                        if (not (pairs[i][1] > ver_lines_shu[j][3] or pairs[i][3] < ver_lines_shu[j][1])) or min(
                                abs(pairs[i][1] - ver_lines_shu[j][3]), abs(pairs[i][3] - ver_lines_shu[j][1])) < ra * (
                                pairs[i][3] - pairs[i][1]):  # 两矩形在y坐标上的高有重叠或者距离近
                            if (pairs[i][1] < heng and ver_lines_shu[j][3] > pairs[i][3]) or (
                                    pairs[i][1] > heng and ver_lines_shu[j][1] < pairs[i][
                                1]):  # 要求横pairs在图片上方时，匹配的直线在横pairs下方
                                middle[0:4] = ver_lines_shu[j]
                                middle[4] = abs(pairs[i][0] - ver_lines_shu[j][0])
                                right_straight = np.r_[right_straight, [middle]]
                                print("内向横pairs找到右竖线")
                left_straight = left_straight[np.argsort(left_straight[:, 4])]  # 按距离从小到大排序
                right_straight = right_straight[np.argsort(right_straight[:, 4])]  # 按距离从小到大排序
                if len(left_straight) > 0 and len(right_straight) > 0:
                    pairs_length_middle[0:4] = pairs[i, 0:4]
                    pairs_length_middle[4:8] = left_straight[0, 0:4]
                    pairs_length_middle[8:12] = right_straight[0, 0:4]
                    pairs_length_middle[12] = abs(left_straight[0, 0] - right_straight[0, 0])
                    pairs_length = np.r_[pairs_length, [pairs_length_middle]]
            if (pairs[i][2] - pairs[i][0]) < (pairs[i][3] - pairs[i][1]):  # 竖向标尺线
                print("竖向")
                up_straight = np.zeros((0, 5))  # 存储可能匹配的上侧直线，第五位是与标尺线上端点的距离
                down_straight = np.zeros((0, 5))  # 存储可能匹配的下侧直线，第五位是与标尺线下端点的距离
                middle = np.zeros((5))
                for j in range(len(ver_lines_heng)):
                    # 1.找上端附近的直线，要求纵坐标在端点附近，横坐标穿过或者在标尺线附近
                    if pairs[i][1] + ratio_inside * (pairs[i][3] - pairs[i][1]) < ver_lines_heng[j][1] < pairs[i][
                        1] + ratio * (
                            pairs[i][3] - pairs[i][1]):  # 纵坐标在上端点下侧
                        if (not (pairs[i][0] > ver_lines_heng[j][2] or pairs[i][2] < ver_lines_heng[j][0])) or min(
                                abs(pairs[i][0] - ver_lines_heng[j][2]),
                                abs(pairs[i][2] - ver_lines_heng[j][0])) < ra * (
                                pairs[i][2] - pairs[i][0]):  # 两矩形在x坐标上的高有重叠或者距离近
                            if (pairs[i][0] > shu and ver_lines_heng[j][0] < pairs[i][0]) or (
                                    pairs[i][0] < shu and ver_lines_heng[j][2] > pairs[i][
                                2]):  # 要求竖pairs在图片左方时，匹配的直线在竖pairs右方
                                middle[0:4] = ver_lines_heng[j]
                                middle[4] = abs(pairs[i][1] - ver_lines_heng[j][1])
                                up_straight = np.r_[up_straight, [middle]]
                                print("内向竖pairs找到上横线")

                    # 2.找下端附近的直线，要求纵坐标在端点附近，横坐标穿过或者在标尺线附近
                    if pairs[i][3] - ratio * (pairs[i][3] - pairs[i][1]) < ver_lines_heng[j][1] < pairs[i][
                        3] - ratio_inside * (
                            pairs[i][3] - pairs[i][1]):  # 纵坐标在下端点上侧
                        if (not (pairs[i][0] > ver_lines_heng[j][2] or pairs[i][2] < ver_lines_heng[j][0])) or min(
                                abs(pairs[i][0] - ver_lines_heng[j][2]),
                                abs(pairs[i][2] - ver_lines_heng[j][0])) < ra * (
                                pairs[i][2] - pairs[i][0]):  # 两矩形在y坐标上的高有重叠或者距离近
                            if (pairs[i][0] > shu and ver_lines_heng[j][0] < pairs[i][0]) or (
                                    pairs[i][0] < shu and ver_lines_heng[j][2] > pairs[i][
                                2]):  # 要求竖pairs在图片左方时，匹配的直线在竖pairs右方
                                middle[0:4] = ver_lines_heng[j]
                                middle[4] = abs(pairs[i][0] - ver_lines_heng[j][0])
                                down_straight = np.r_[down_straight, [middle]]
                                print("内向竖pairs找到下横线")
                up_straight = up_straight[np.argsort(up_straight[:, 4])]  # 按距离从小到大排序
                down_straight = down_straight[np.argsort(down_straight[:, 4])]  # 按距离从小到大排序
                if len(up_straight) > 0 and len(down_straight) > 0:
                    pairs_length_middle[0:4] = pairs[i, 0:4]
                    pairs_length_middle[4:8] = up_straight[0, 0:4]
                    pairs_length_middle[8:12] = down_straight[0, 0:4]
                    pairs_length_middle[12] = abs(up_straight[0, 0] - down_straight[0, 0])
                    pairs_length = np.r_[pairs_length, [pairs_length_middle]]
        print("一组pairs结束*")
    if test_mode == 1:
        drawn_img = cv2.imread(img_path)
        for i in range(len(pairs_length)):
            x1 = int(pairs_length[i][4])
            x2 = int(pairs_length[i][6])
            y1 = int(pairs_length[i][5])
            y2 = int(pairs_length[i][7])
            drawn_img = cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            x1 = int(pairs_length[i][8])
            x2 = int(pairs_length[i][10])
            y1 = int(pairs_length[i][9])
            y2 = int(pairs_length[i][11])
            drawn_img = cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for i in range(len(pairs_length)):
            # 绘制一个红色矩形
            ptLeftTop = (int(pairs_length[i][0]), int(pairs_length[i][1]))
            ptRightBottom = (int(pairs_length[i][2]), int(pairs_length[i][3]))
            point_color = (0, 0, 255)  # BGR
            thickness = 2
            lineType = 8
            cv2.rectangle(drawn_img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        # Show image
        cv2.namedWindow("LSD", 0)
        cv2.imshow("LSD", drawn_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("视图结束**")
    try:
        drawn_img = cv2.imread(img_path)
        for i in range(len(pairs_length)):
            x1 = int(pairs_length[i][4])
            x2 = int(pairs_length[i][6])
            y1 = int(pairs_length[i][5])
            y2 = int(pairs_length[i][7])
            drawn_img = cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            x1 = int(pairs_length[i][8])
            x2 = int(pairs_length[i][10])
            y1 = int(pairs_length[i][9])
            y2 = int(pairs_length[i][11])
            drawn_img = cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for i in range(len(pairs_length)):
            # 绘制一个红色矩形
            ptLeftTop = (int(pairs_length[i][0]), int(pairs_length[i][1]))
            ptRightBottom = (int(pairs_length[i][2]), int(pairs_length[i][3]))
            point_color = (0, 0, 255)  # BGR
            thickness = 2
            lineType = 8
            cv2.rectangle(drawn_img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        # 保存图片
        path = OPENCV_OUTPUT_LINE + img_path
        cv2.imwrite(path, drawn_img)
        print("保存引线+标尺线组合成功:", path)
    except:
        print("保存引线+标尺线组合失败")
    print("***/结束引线和标尺线的匹配/***")
    return pairs_length  # np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]


def delete_other(other, guangJiedu, data):
    """剔除检测结果中 OTHER 类别的干扰项。"""
    other:np.(,4)[x1,y1,x2,y2]
    guangJiedu:np.(,4)[x1,y1,x2,y2]
    data:np.(,4)[x1,y1,x2,y2]
    '''
    # 将other和光洁度框线缩小防止误删
    ratio = 0.5
    ratio = ratio * 0.5
    for i in range(len(other)):
        other[i][0] = other[i][0] + ratio * abs(other[i][0] - other[i][2])
        other[i][1] = other[i][1] + ratio * abs(other[i][1] - other[i][3])
        other[i][2] = other[i][2] - ratio * abs(other[i][0] - other[i][2])
        other[i][3] = other[i][3] - ratio * abs(other[i][1] - other[i][3])
    for i in range(len(guangJiedu)):
        guangJiedu[i][0] = guangJiedu[i][0] + ratio * abs(guangJiedu[i][0] - guangJiedu[i][2])
        guangJiedu[i][1] = guangJiedu[i][1] + ratio * abs(guangJiedu[i][1] - guangJiedu[i][3])
        guangJiedu[i][2] = guangJiedu[i][2] - ratio * abs(guangJiedu[i][0] - guangJiedu[i][2])
        guangJiedu[i][3] = guangJiedu[i][3] - ratio * abs(guangJiedu[i][1] - guangJiedu[i][3])

    # 当外框重叠且重叠部分在两方中至少一方占面积比较大，即可筛除
    data_count = np.zeros((len(data)))  # 1=将要被筛出
    ratio = 0.5
    for i in range(len(data)):
        for j in range(len(other)):
            if not (data[i][0] > other[j][2] or data[i][2] < other[j][0]):  # 两矩形在x坐标上的长有重叠
                if not (data[i][1] > other[j][3] or data[i][3] < other[j][1]):  # 两矩形在y坐标上的高有重叠
                    l = data[i][2] - data[i][0] + other[j][2] - other[j][0] - max(abs(data[i][0] - other[j][2]),
                                                                                  abs(data[i][2] - other[j][0]))
                    w = data[i][3] - data[i][1] + other[j][3] - other[j][1] - max(abs(data[i][1] - other[j][3]),
                                                                                  abs(data[i][3] - other[j][1]))
                    if l * w / (data[i][2] - data[i][0]) * (data[i][3] - data[i][1]) > ratio or l * w / (
                            other[j][2] - other[j][0]) * (other[j][3] - other[j][1]) > ratio:
                        data_count[i] = 1
        for k in range(len(guangJiedu)):
            if not (data[i][0] > guangJiedu[k][2] or data[i][2] < guangJiedu[k][0]):  # 两矩形在x坐标上的长有重叠
                if not (data[i][1] > guangJiedu[k][3] or data[i][3] < guangJiedu[k][1]):  # 两矩形在y坐标上的高有重叠
                    l = data[i][2] - data[i][0] + guangJiedu[k][2] - guangJiedu[k][0] - max(
                        abs(data[i][0] - guangJiedu[k][2]),
                        abs(data[i][2] - guangJiedu[k][0]))
                    w = data[i][3] - data[i][1] + guangJiedu[k][3] - guangJiedu[k][1] - max(
                        abs(data[i][1] - guangJiedu[k][3]),
                        abs(data[i][3] - guangJiedu[k][1]))
                    if l * w / (data[i][2] - data[i][0]) * (data[i][3] - data[i][1]) > ratio or l * w / (
                            guangJiedu[k][2] - guangJiedu[k][0]) * (guangJiedu[k][3] - guangJiedu[k][1]) > ratio:
                        data_count[i] = 1
    new_data = np.zeros((0, 4))
    for i in range(len(data_count)):
        if data_count[i] == 0:
            new_data = np.r_[new_data, [data[i]]]
    return new_data


def find_serial_number_letter(serial_numbers, serial_letters, bottom_dbnet_data):
    """结合序号与字母框定位底部的引脚索引。"""
    serial_numbers:np(,4)[x1,y1,x2,y2]
    serial_letters:np(,4)[x1,y1,x2,y2]
    bottom_dbnet_data:np(,4)[x1,y1,x2,y2]
    '''
    # 将serial提取出唯一值
    if len(serial_numbers) >= 1:
        maxlength = 0
        max_no = -1
        for i in range(len(serial_numbers)):
            if maxlength < max(serial_numbers[i][2] - serial_numbers[i][0],
                               serial_numbers[i][3] - serial_numbers[i][1]):
                maxlength = max(serial_numbers[i][2] - serial_numbers[i][0],
                                serial_numbers[i][3] - serial_numbers[i][1])
                max_no = i
        only_serial_numbers = serial_numbers[max_no]
        # print("only_serial_numbers", only_serial_numbers)
    if len(serial_letters) >= 1:
        maxlength = 0
        max_no = -1
        for i in range(len(serial_letters)):
            if maxlength < max(serial_letters[i][2] - serial_letters[i][0],
                               serial_letters[i][3] - serial_letters[i][1]):
                maxlength = max(serial_letters[i][2] - serial_letters[i][0],
                                serial_letters[i][3] - serial_letters[i][1])
                max_no = i
        only_serial_letters = serial_letters[max_no]
        # print("only_serial_letters", only_serial_letters)
    # 提取serial中的数字和字母
    ratio = 0.2
    serial_numbers_data = np.zeros((0, 4))
    serial_letters_data = np.zeros((0, 4))
    bottom_dbnet_data_account = np.zeros((len(bottom_dbnet_data)))  # 1 = 是serial的文本，需要剔除
    new_bottom_dbnet_data = np.zeros((0, 4))
    if len(serial_numbers) != 0:
        for i in range(len(bottom_dbnet_data)):
            if not (bottom_dbnet_data[i][0] > only_serial_numbers[2] or bottom_dbnet_data[i][2] < only_serial_numbers[
                0]):  # 两矩形在x坐标上的长有重叠
                if not (bottom_dbnet_data[i][1] > only_serial_numbers[3] or bottom_dbnet_data[i][3] <
                        only_serial_numbers[
                            1]):  # 两矩形在y坐标上的高有重叠
                    # print('********')
                    l = abs(bottom_dbnet_data[i][2] - bottom_dbnet_data[i][0]) + abs(only_serial_numbers[2] - \
                                                                                     only_serial_numbers[0]) - (
                                max(only_serial_numbers[2], bottom_dbnet_data[i][2]) - min(only_serial_numbers[0],
                                                                                           bottom_dbnet_data[i][0]))

                    w = bottom_dbnet_data[i][3] - bottom_dbnet_data[i][1] + only_serial_numbers[3] - \
                        only_serial_numbers[1] - (
                                max(only_serial_numbers[3], bottom_dbnet_data[i][3]) - min(only_serial_numbers[1],
                                                                                           bottom_dbnet_data[i][1]))
                    if l * w / (bottom_dbnet_data[i][2] - bottom_dbnet_data[i][0]) * (
                            bottom_dbnet_data[i][3] - bottom_dbnet_data[i][1]) > ratio or l * w / (
                            only_serial_numbers[2] - only_serial_numbers[0]) * (
                            only_serial_numbers[3] - only_serial_numbers[1]) > ratio:
                        serial_numbers_data = np.r_[serial_numbers_data, [bottom_dbnet_data[i]]]
                        bottom_dbnet_data_account[i] = 1
    if len(serial_letters) != 0:
        for i in range(len(bottom_dbnet_data)):
            if not (bottom_dbnet_data[i][0] > only_serial_letters[2] or bottom_dbnet_data[i][2] < only_serial_letters[
                0]):  # 两矩形在x坐标上的长有重叠
                if not (bottom_dbnet_data[i][1] > only_serial_letters[3] or bottom_dbnet_data[i][3] <
                        only_serial_letters[
                            1]):  # 两矩形在y坐标上的高有重叠
                    l = abs(bottom_dbnet_data[i][2] - bottom_dbnet_data[i][0]) + abs(only_serial_letters[2] - \
                                                                                     only_serial_letters[0]) - (
                                max(only_serial_letters[2], bottom_dbnet_data[i][2]) - min(only_serial_letters[0],
                                                                                           bottom_dbnet_data[i][0]))

                    w = bottom_dbnet_data[i][3] - bottom_dbnet_data[i][1] + only_serial_letters[3] - \
                        only_serial_letters[1] - (
                                max(only_serial_letters[3], bottom_dbnet_data[i][3]) - min(only_serial_letters[1],
                                                                                           bottom_dbnet_data[i][1]))
                    if l * w / (bottom_dbnet_data[i][2] - bottom_dbnet_data[i][0]) * (
                            bottom_dbnet_data[i][3] - bottom_dbnet_data[i][1]) > ratio or l * w / (
                            only_serial_letters[2] - only_serial_letters[0]) * (
                            only_serial_letters[3] - only_serial_letters[1]) > ratio:
                        serial_letters_data = np.r_[serial_letters_data, [bottom_dbnet_data[i]]]
                        bottom_dbnet_data_account[i] = 1
    for i in range(len(bottom_dbnet_data_account)):
        if bottom_dbnet_data_account[i] == 0:
            new_bottom_dbnet_data = np.r_[new_bottom_dbnet_data, [bottom_dbnet_data[i]]]
    return serial_numbers_data, serial_letters_data, new_bottom_dbnet_data


def filter_bottom_ocr_data(bottom_ocr_data, bottom_dbnet_data_serial, serial_numbers, serial_letters,
    """清理底部 OCR 文本，保留有效的序号与尺寸。"""
                           bottom_dbnet_data):
    '''
    输出serial_numbers_data:np.(,4)['x1','y1','x2','y2','str']
    serial_numbers:np(,4)[x1,y1,x2,y2]
    '''
    serial_numbers_data = np.zeros((0, 5))
    serial_letters_data = np.zeros((0, 5))
    bottom_ocr_new_data = []
    for i in range(len(bottom_dbnet_data_serial)):
        for j in range(len(serial_numbers)):
            if (bottom_dbnet_data_serial[i] == serial_numbers[j]).all():
                mid_str = np.empty(5, dtype=np.dtype('U10'))
                mid_str[0: 4] = serial_numbers[j].astype(str)
                mid_str[4] = bottom_ocr_data[i]
                serial_numbers_data = np.r_[serial_numbers_data, [mid_str]]
                break
        for k in range(len(serial_letters)):
            if (bottom_dbnet_data_serial[i] == serial_letters[k]).all():
                mid_str = np.empty(5, dtype=np.dtype('U10'))
                mid_str[0: 4] = serial_letters[k].astype(str)
                mid_str[4] = bottom_ocr_data[i]
                serial_letters_data = np.r_[serial_letters_data, [mid_str]]
                break
        for l in range(len(bottom_dbnet_data)):
            if (bottom_dbnet_data_serial[i] == bottom_dbnet_data[l]).all():
                bottom_ocr_new_data.append(bottom_ocr_data[i])
                break
    return serial_numbers_data, serial_letters_data, bottom_ocr_new_data


def shilter_table_data(ocr_data):
    """对表格结构的 OCR 结果进行筛查与纠错。"""
    ocr_data = {'location': dbnet_data[i], 'ocr_strings': ocr_data[i], 'key_info': [],
               'matched_pairs_location': [], 'matched_pairs_outside_or_inside': [],
               'matched_pairs_yinXian': [], 'Absolutely': [], 'max_medium_min': []}
    '''
    new_dbnet_data = []
    for i in range(len(ocr_data)):
        strings = ocr_data[i]['ocr_strings']
        if len(strings) < 5:
            new_dbnet_data.append(ocr_data[i])
    return new_dbnet_data


def convert_Dic(dbnet_data, ocr_data):
    """把 DBNet/OCR 结果整合成字典形式。"""
    将ocr识别出来的字符串与位置信息结合为字典类型
    dbnet_data:np(, 4)[x1,y1,x2,y2]
    ocr_data:list['string','str']
    '''
    new_ocr_data = []
    for i in range(len(dbnet_data)):
        dic = {'location': dbnet_data[i], 'ocr_strings': ocr_data[i], 'key_info': [],
               'matched_pairs_location': [], 'matched_pairs_outside_or_inside': [],
               'matched_pairs_yinXian': [], 'Absolutely': [], 'max_medium_min': []}
        new_ocr_data.append(dic)
    return new_ocr_data


def filter_ocr_data_0(ocr_data):
    """应用规则 0 对 OCR 数据进行过滤。"""
    筛除‘’
    '''
    new_ocr_data = []
    for i in range(len(ocr_data)):
        if not (ocr_data[i]['ocr_strings'] == ''):
            new_ocr_data.append(ocr_data[i])
    return new_ocr_data


def filter_ocr_data__1(ocr_data):
    """应用规则 -1 对 OCR 数据进行过滤。"""
    new_ocr_data = []
    for i in range(len(ocr_data)):
        if not (ocr_data[i]['key_info'] == []):
            new_ocr_data.append(ocr_data[i])
    return new_ocr_data


def filter_ocr_data__2(ocr_data):
    """应用规则 -2 对 OCR 数据进行过滤。"""
    清理key_info中不含数字的数据
    '''
    new_ocr_data = []
    for i in range(len(ocr_data)):
        right_key = 0
        for j in range(len(ocr_data[i]['key_info'])):
            if right_key == 1:
                break
            for k in range(len(ocr_data[i]['key_info'][j])):
                try:
                    a = float(ocr_data[i]['key_info'][j][k])
                    right_key = 1
                    break
                except:
                    pass
        if right_key == 1:
            new_ocr_data.append(ocr_data[i])
    ocr_data = new_ocr_data

    return ocr_data


def filter_ocr_data_1(list):
    """应用规则 1 对 OCR 数据进行过滤。"""
    字符串中删除‘数字‘ + 'X’ (字符串中不存在'='时),'Absolutely'为'mb_pin_diameter'
    '''
    for i in range(len(list)):
        str_data1 = re.findall("=", list[i]['ocr_strings'])
        if len(str_data1) == 0:
            str_data2 = re.findall("(\d+\.?\d*)[Xx]", list[i]['ocr_strings'])
            str_data = re.sub("(\d+\.?\d*)[Xx]", '', list[i]['ocr_strings'])
            if len(str_data2) != 0:
                list[i]['Absolutely'] = 'mb_pin_diameter'
            list[i]['ocr_strings'] = str_data
    return list


def filter_ocr_data_2(list):
    """应用规则 2 对 OCR 数据进行过滤。"""
    字符串中删除'PIN1'和’PIN‘
    '''
    for i in range(len(list)):
        str_data = re.sub("[Pp][Ii][Nn]1*", '', list[i]['ocr_strings'])
        list[i]['ocr_strings'] = str_data
    return list


def filter_ocr_data_3(list):
    """应用规则 3 对 OCR 数据进行过滤。"""
    字符串中删除'A1'
    '''
    for i in range(len(list)):
        str_data = re.sub("[Aa]1", '', list[i]['ocr_strings'])
        list[i]['ocr_strings'] = str_data
    return list


def filter_ocr_data_11(list):
    """应用规则 11 对 OCR 数据进行过滤。"""
    删除'note' + '整数数字'
    '''
    for i in range(len(list)):
        str_data = re.sub("[Nn][Oo][Tt][Ee][23456789]", '', list[i]['ocr_strings'])
        list[i]['ocr_strings'] = str_data
    return list


def filter_ocr_data_4(list):
    """应用规则 4 对 OCR 数据进行过滤。"""
    ’,‘改为’.‘
    '''
    for i in range(len(list)):
        str_data = re.sub("[,，]", '.', list[i]['ocr_strings'])
        list[i]['ocr_strings'] = str_data
    return list


def filter_ocr_data_5(list):
    """应用规则 5 对 OCR 数据进行过滤。"""
    单个字符时，字符串中删除'A''B''C''D'
    '''
    for i in range(len(list)):
        if len(list[i]['ocr_strings']) == 1:
            str_data = re.sub("[AaBbCcDd]", '', list[i]['ocr_strings'])
            list[i]['ocr_strings'] = str_data
    return list


def filter_ocr_data_6(list):
    """应用规则 6 对 OCR 数据进行过滤。"""
    提取’数字‘’+‘’-‘’=‘’Φ‘’±‘’max‘’nom‘’min‘'x'
    如果检测到"±",仅保留"±"以及符号的前一位数字和后一位数字
    '''
    for i in range(len(list)):
        str_data = re.findall("\d+(?:\.\d+)?|=|\+|-|Φ|±|[Mm][Aa][Xx]|[Nn][Oo][Mm]|[Mm][Ii][Nn]|[Xx*]",
                              list[i]['ocr_strings'])
        str_data = [x.strip() for x in str_data if x.strip() != '']  # 将字符串列表中空项删除
        list[i]['key_info'] = str_data

        for j in range(len(str_data)):
            if j > 0:
                if str_data[j] == '±':
                    try:
                        a = float(str_data[j - 1])
                        b = float(str_data[j + 1])
                        new_str_data = []
                        new_str_data.append(str_data[j - 1])
                        new_str_data.append(str_data[j])
                        new_str_data.append(str_data[j + 1])
                        for k in range(len(str_data)):
                            if str_data[k] == 'Φ':
                                new_str_data.append(str_data[k])
                        list[i]['key_info'] = new_str_data
                    except:
                        pass

    return list


def filter_ocr_data_7(list):
    """应用规则 7 对 OCR 数据进行过滤。"""
    删除key_info中的'.'
    当key_info中只有'x'时删除
    '''
    for i in range(len(list)):
        new_key_info = []
        for k in range(len(list[i]['key_info'])):
            if list[i]['key_info'][k] != '.':
                new_key_info.append(list[i]['key_info'][k])
        list[i]['key_info'] = new_key_info

        if len(list[i]['key_info']) == 1:
            if list[i]['key_info'][0] == 'X' or list[i]['key_info'][0] == 'x':
                list[i]['key_info'] = []
    return list


def filter_ocr_data_8(list):
    """应用规则 8 对 OCR 数据进行过滤。"""
    找‘Φ’，并删除'Φ'然后标识'absolute'
    '''
    for i in range(len(list)):
        str_data = re.findall("Φ", list[i]['ocr_strings'])
        str_data = [x.strip() for x in str_data if x.strip() != '']  # 将字符串列表中空项删除
        if len(str_data) > 0:
            str_data1 = re.sub("Φ", '', list[i]['ocr_strings'])
            list[i]['ocr_strings'] = str_data1
            list[i]['Absolutely'] = 'pin_diameter'
    return list


def filter_ocr_data_9(ocr_data):
    """应用规则 9 对 OCR 数据进行过滤。"""
    key_info中的数字如果以'0'开头而第二个字符却没有小数点，则添加小数点
    '''
    for i in range(len(ocr_data)):
        for j in range(len(ocr_data[i]['key_info'])):
            for k in range(len(ocr_data[i]['key_info'][j])):
                try:
                    a = float(ocr_data[i]['key_info'][j][k])
                    if ocr_data[i]['key_info'][j][k][0] == '0':
                        if ocr_data[i]['key_info'][j][k][1] != '.':
                            b = '.'
                            str_list = list(ocr_data[i]['key_info'][j][k])
                            str_list.insert(1, b)
                            a_b = ''.join(str_list)
                            ocr_data[i]['key_info'][j][k] = a_b
                except:
                    pass
    return ocr_data


def filter_ocr_data_10(ocr_data):
    """应用规则 10 对 OCR 数据进行过滤。"""
    # 删除key_info中的'0','0.','00'
    '''

    for i in range(len(ocr_data)):
        new_key_info = []
        for j in range(len(ocr_data[i]['key_info'])):
            string = ocr_data[i]['key_info'][j]
            if not (string == '0' or string == '0.' or string == '00'):
                new_key_info.append(string)
        ocr_data[i]['key_info'] = new_key_info
    return ocr_data


def filter_ocr_data_12(ocr_data):
    """应用规则 12 对 OCR 数据进行过滤。"""
    删除公差特别大的标注
    '''
    new_ocr_data = []
    for i in range(len(ocr_data)):
        if abs(ocr_data[i]['max_medium_min'][0] - ocr_data[i]['max_medium_min'][1]) < 1 and abs(
                ocr_data[i]['max_medium_min'][1] - ocr_data[i]['max_medium_min'][2]) < 1:
            new_ocr_data.append(ocr_data[i])
        else:
            print('删除公差特别大的标注:', ocr_data[i]['max_medium_min'])
    return new_ocr_data

def time_save_find_pinmap(bottom_border):
    """异步执行 pinmap 识别并缓存结果。"""
    result_queue = queue.Queue()
    thread = threading.Thread(target=long_running_task, args=(result_queue, bottom_border))
    thread.start()

    thread.join(timeout=10)  # 设置超时时间为5秒
    if thread.is_alive():
        print("读取pinmap进程花费时间过长，跳过")
        pin_map = np.ones((10, 10))
        color = np.full_like(pin_map, fill_value=2)
        # 记录pin的行列数
        pin_num_x_y = np.array([0, 0])
        pin_num_x_y = pin_num_x_y.astype(int)
        path = f'{YOLOX_DATA}\pin_num.txt'
        np.savetxt(path, pin_num_x_y)
    else:
        try:
            output_list = result_queue.get_nowait()  # 尝试获取结果
            half_index = output_list.shape[0] // 2  # 取行数的一半作为分割点

            pin_map = output_list[:half_index, :]  # 上半部分
            color = output_list[half_index:, :]  # 下半部分

            # print("Result:", pin_map)
            # print("Result:", color)
            # print("Result:", pin_map)
        except queue.Empty:
            print("Queue is empty, no result available.")
            pin_map = np.ones((10, 10))
            color = np.full_like(pin_map, fill_value=2)
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

    return pin_map, color

def long_running_task(result_queue, bottom_border):
    """在子线程中执行耗时的 pinmap 识别任务。"""
    print()
    print("***/开始检测pin/***")
    result = find_pin(bottom_border)
    # try:
    #     result = find_pin()
    # except:
    #     print("pinmap没有正常读取，请记录pdf并反馈")
    #     result = np.ones((10, 10))
    result_queue.put(result)
    print("***/结束检测pin/***")
    print()
def yolox_dbnet_ocr_match(test_mode, letter_or_number):

    """串联 YOLO、DBNet 与 OCR，完成匹配流程。"""
    package_classes = 'BGA'
    # import time
    start = time.time()
    key = test_mode  # 是否展示并调试过程
    package_path = DATA
    # empty_data = np.empty((0, 4))
    img_path = f'{package_path}/top.jpg'

    top_dbnet_data = dbnet_get_text_box(img_path)
    top_yolox_pairs, top_yolox_num, top_yolox_serial_num, top_pin, top_other, top_pad, top_border, top_angle_pairs, top_BGA_serial_num, top_BGA_serial_letter = yolo_classify(
        img_path, package_classes)
    img_path = f'{package_path}/bottom.jpg'
    bottom_dbnet_data = dbnet_get_text_box(img_path)
    bottom_yolox_pairs, bottom_yolox_num, bottom_yolox_serial_num, bottom_pin, bottom_other, bottom_pad, bottom_border, bottom_angle_pairs, bottom_BGA_serial_num, bottom_BGA_serial_letter = yolo_classify(
        img_path, package_classes)
    img_path = f'{package_path}/side.jpg'

    side_dbnet_data = dbnet_get_text_box(img_path)
    side_yolox_pairs, side_yolox_num, side_yolox_serial_num, side_pin, side_other, side_pad, side_border, side_angle_pairs, side_BGA_serial_num, side_BGA_serial_letter = yolo_classify(
        img_path, package_classes)



    # # 1.在各个视图中用yolox识别标尺线、多值标注，dbnet识别标注坐标位置
    # path = r"data/top.jpg"
    # top_yolox_pairs, top_yolox_num, top_other, dbnet_time1 = get_pairs_data(path)
    # top_yolox_pairs = np.around(top_yolox_pairs, decimals=2)
    # top_yolox_num = np.around(top_yolox_num, decimals=2)
    # # top_dbnet_data = np.around(top_dbnet_data, decimals=2)
    # # 参数格式:top_yolox_pairs  np.二维数组[x1,y1,x2,y2,0 = outside 1 = inside]
    # # 参数格式:top_yolox_num np.二维数组[x1,y1,x2,y2]
    # # 参数格式:top_dbnet_data np.二维数组[x1,y1,x2,y2]
    #
    # path = r"data/bottom.jpg"
    # bottom_yolox_pairs, bottom_yolox_num, bottom_other, dbnet_time2 = get_pairs_data(path)
    # bottom_yolox_pairs = np.around(bottom_yolox_pairs, decimals=2)
    # bottom_yolox_num = np.around(bottom_yolox_num, decimals=2)
    # # bottom_dbnet_data = np.around(bottom_dbnet_data, decimals=2)
    # # 参数格式:top_yolox_pairs  np.二维数组[x1,y1,x2,y2,0 = outside 1 = inside]
    # # 参数格式:top_yolox_num np.二维数组[x1,y1,x2,y2]
    # # 参数格式:top_dbnet_data np.二维数组[x1,y1,x2,y2]
    #
    # path = r"data/side.jpg"
    # side_yolox_pairs, side_yolox_num, side_other, dbnet_time3 = get_pairs_data(path)
    # side_yolox_pairs = np.around(side_yolox_pairs, decimals=2)
    # side_yolox_num = np.around(side_yolox_num, decimals=2)
    # # side_dbnet_data = np.around(side_dbnet_data, decimals=2)
    # # 参数格式:top_yolox_pairs  np.二维数组[x1,y1,x2,y2,0 = outside 1 = inside]
    # # 参数格式:top_yolox_num np.二维数组[x1,y1,x2,y2]
    # # 参数格式:top_dbnet_data np.二维数组[x1,y1,x2,y2]
    end1 = time.time()
    # 2.yolox检测无用的数字标注（other）并删除yolox和dbnet检测的标注中的无用标注
    # from output_other_location import begain_output_other_location

    # results = result_queue.get()
    # BGA独有流程：获取pinmap，找到缺pin位置信息
    pin_map, color = time_save_find_pinmap(bottom_border)
    # print("输出pinmap", pin_map)
    # print("输出color", color)

    guangJiedu = np.zeros((0, 4))
    top_yolox_num = delete_other(top_other, guangJiedu, top_yolox_num)
    top_dbnet_data = delete_other(top_other, guangJiedu, top_dbnet_data)

    bottom_yolox_num = delete_other(bottom_other, guangJiedu, bottom_yolox_num)
    bottom_dbnet_data = delete_other(bottom_other, guangJiedu, bottom_dbnet_data)

    side_yolox_num = delete_other(side_other, guangJiedu, side_yolox_num)
    side_dbnet_data = delete_other(side_other, guangJiedu, side_dbnet_data)
    # 3.yolox检测bottom中标记行列序号的数字和字母，提取并分离出yolox和dbnet检测的标注中的
    img_path = 'data/bottom.jpg'
    # serial_numbers, serial_letters = begain_output_serial_number_letter_location(img_path)
    serial_letters = bottom_BGA_serial_letter
    serial_numbers = bottom_BGA_serial_num
    bottom_dbnet_data_serial = bottom_dbnet_data.copy()
    serial_numbers_data, serial_letters_data, bottom_dbnet_data = find_serial_number_letter(serial_numbers,
                                                                                            serial_letters,
                                                                                            bottom_dbnet_data)
    # print("serial_numbers_data, serial_letters_data\n", serial_numbers_data, serial_letters_data)
    # # 补充，bottom视图中的dbnet检测出多余项需要筛出。大多多余项集中在pin图中，来自output_pin_num4.py的get_pinmap（）中采集了pinmap在bottom上的位置
    # bottom_dbnet_data = filter_bottom_dbnet(bottom_dbnet_data)
    # if key == 1:
    #     print("展示dbnet检测的文本（滤除pin_map处的各种空文本）bottom_dbnet_data")
    #     img_path = r"data/bottom.jpg"
    #     show_data(img_path, bottom_dbnet_data)  # 展示dbnet框选的尺寸数字

    # 4.根据筛选去other，光洁度，和标记着行列数的数字和英文字母之后的bottom的ocr结果判断是表格型BGA还是数字型
    # img_path = 'data/bottom.jpg'
    # letter_or_number, bottom_dbnet_data_table = ocr_onnx_table_or_number(img_path, bottom_dbnet_data)
    if letter_or_number == 'number':
        # 每个尺寸线在两端附近找垂直于标尺线的两个平行直线并计算距离
        img_path = f'{DATA}/top.jpg'
        top_yolox_pairs_length = find_pairs_length(img_path, top_yolox_pairs, test_mode)
        # top_yolox_pairs_length np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
        img_path = f'{DATA}/bottom.jpg'
        bottom_yolox_pairs_length = find_pairs_length(img_path, bottom_yolox_pairs, test_mode)
        img_path = f'{DATA}/side.jpg'
        side_yolox_pairs_length = find_pairs_length(img_path, side_yolox_pairs, test_mode)
        # 去除标尺线数据中标记着标尺线内外向的数据
        top_yolox_pairs_copy = top_yolox_pairs.copy()
        bottom_yolox_pairs_copy = bottom_yolox_pairs.copy()
        side_yolox_pairs_copy = side_yolox_pairs.copy()
        top_yolox_pairs = top_yolox_pairs[:, 0:4]
        # top_yolox_pairs  np.二维数组[x1,y1,x2,y2]
        bottom_yolox_pairs = bottom_yolox_pairs[:, 0:4]
        # bottom_yolox_pairs  np.二维数组[x1,y1,x2,y2]
        side_yolox_pairs = side_yolox_pairs[:, 0:4]
        # side_yolox_pairs  np.二维数组[x1,y1,x2,y2]
        if key == 1:
            print("展示dbnet检测的文本（删除other）top_dbnet_data")
            img_path = f'{DATA}/top.jpg'
            show_data(img_path, top_dbnet_data)  # 展示dbnet框选的尺寸数字
            print("展示dbnet检测的文本（删除other，标记行列序号的数字和字母标注）bottom_dbnet_data")
            img_path = f'{DATA}/bottom.jpg'
            show_data(img_path, bottom_dbnet_data)  # 展示dbnet框选的尺寸数字
            print("展示dbnet检测的文本（删除other）side_dbnet_data")
            img_path = f'{DATA}/side.jpg'
            show_data(img_path, side_dbnet_data)  # 展示dbnet框选的尺寸数字

        # 6.1.2ocr识别，剔除文字项
        print("---开始各个视图的SVTR识别---")
        start1 = time.time()
        path = f'{DATA}/top.jpg'
        top_ocr_data = ocr_data(path, top_dbnet_data)
        path = f'{DATA}/bottom.jpg'
        bottom_ocr_data = ocr_data(path, bottom_dbnet_data_serial)
        serial_numbers_data, serial_letters_data, bottom_ocr_data = filter_bottom_ocr_data(bottom_ocr_data,
                                                                                           bottom_dbnet_data_serial,
                                                                                           serial_numbers_data,
                                                                                           serial_letters_data,
                                                                                           bottom_dbnet_data)

        path = f'{DATA}/side.jpg'
        side_ocr_data = ocr_data(path, side_dbnet_data)
        end = time.time()
        print("---结束各个视图的SVTR识别---")
        print("yolox+dbnet时间：", end1 - start)
        # print("dbnet时间：", dbnet_time1 + dbnet_time2 + dbnet_time3)
        print("ocr时间：", end - start1)
        print("yolox+dbnet+ocr时间：", end - start)
        # 6.1ocr后处理，输出各个标注的max_medium_min
        # 编辑为字典类型
        top_ocr_data = convert_Dic(top_dbnet_data, top_ocr_data)
        # [{'location': array([1245., 88., 1302., 135.]), 'ocr_strings': ''},{'location': array([635., 110., 725., 156.]), 'ocr_strings': '15.0'}]
        bottom_ocr_data = convert_Dic(bottom_dbnet_data, bottom_ocr_data)
        side_ocr_data = convert_Dic(side_dbnet_data, side_ocr_data)
        # 筛除非参数信息
        # 删除SVTR识别为空
        top_ocr_data = filter_ocr_data_0(top_ocr_data)
        bottom_ocr_data = filter_ocr_data_0(bottom_ocr_data)
        side_ocr_data = filter_ocr_data_0(side_ocr_data)
        # 逗号改为.
        top_ocr_data = filter_ocr_data_4(top_ocr_data)
        bottom_ocr_data = filter_ocr_data_4(bottom_ocr_data)
        side_ocr_data = filter_ocr_data_4(side_ocr_data)
        # 删除’数字‘+’X‘（没有’=‘时）并标记'Absolutely'为'mb_pin_diameter'
        top_ocr_data = filter_ocr_data_1(top_ocr_data)
        bottom_ocr_data = filter_ocr_data_1(bottom_ocr_data)
        side_ocr_data = filter_ocr_data_1(side_ocr_data)
        # 删除’PIN1‘
        top_ocr_data = filter_ocr_data_2(top_ocr_data)
        bottom_ocr_data = filter_ocr_data_2(bottom_ocr_data)
        side_ocr_data = filter_ocr_data_2(side_ocr_data)
        # 删除’A1‘
        top_ocr_data = filter_ocr_data_3(top_ocr_data)
        bottom_ocr_data = filter_ocr_data_3(bottom_ocr_data)
        side_ocr_data = filter_ocr_data_3(side_ocr_data)
        # 删除'note' + '数字'
        top_ocr_data = filter_ocr_data_11(top_ocr_data)
        bottom_ocr_data = filter_ocr_data_11(bottom_ocr_data)
        side_ocr_data = filter_ocr_data_11(side_ocr_data)
        # 删除'A''B''C''D'
        top_ocr_data = filter_ocr_data_5(top_ocr_data)
        bottom_ocr_data = filter_ocr_data_5(bottom_ocr_data)
        side_ocr_data = filter_ocr_data_5(side_ocr_data)
        print("经过预处理得到的top视图的SVTR结果:\n", *top_ocr_data, sep='\n')
        print("经过预处理得到的bottom视图的SVTR结果:\n", *bottom_ocr_data, sep='\n')
        print("经过预处理得到的side视图的SVTR结果:\n", *side_ocr_data, sep='\n')
        # (1).提取’数字‘’+‘’-‘’=‘’Φ‘’±‘’max‘’nom‘’min‘’x'(2).当出现'±'时仅保留'±''数字''Φ'
        top_ocr_data = filter_ocr_data_6(top_ocr_data)
        bottom_ocr_data = filter_ocr_data_6(bottom_ocr_data)
        side_ocr_data = filter_ocr_data_6(side_ocr_data)
        # key_info中的数字如果以'0'开头而第二个字符却没有小数点，则添加小数点
        top_ocr_data = filter_ocr_data_9(top_ocr_data)
        bottom_ocr_data = filter_ocr_data_9(bottom_ocr_data)
        side_ocr_data = filter_ocr_data_9(side_ocr_data)
        # 删除key_info中的'0''0.','00'
        top_ocr_data = filter_ocr_data_10(top_ocr_data)
        bottom_ocr_data = filter_ocr_data_10(bottom_ocr_data)
        side_ocr_data = filter_ocr_data_10(side_ocr_data)
        # (1)删除key_info中的'.'(2)当key_info中只有'x'时删除
        top_ocr_data = filter_ocr_data_7(top_ocr_data)
        bottom_ocr_data = filter_ocr_data_7(bottom_ocr_data)
        side_ocr_data = filter_ocr_data_7(side_ocr_data)
        # 删除标注关键信息检测识别为空
        top_ocr_data = filter_ocr_data__1(top_ocr_data)
        bottom_ocr_data = filter_ocr_data__1(bottom_ocr_data)
        side_ocr_data = filter_ocr_data__1(side_ocr_data)
        print("经过第一步后处理得到的top视图的SVTR结果:\n", *top_ocr_data, sep='\n')
        print("经过第一步后处理得到的bottom视图的SVTR结果:\n", *bottom_ocr_data, sep='\n')
        print("经过第一步后处理得到的side视图的SVTR结果:\n", *side_ocr_data, sep='\n')
        # 6.1.1借助yolox将dbnet标注的框线坐标以及ocr内容合并
        top_ocr_data = bind_data(top_yolox_num, top_ocr_data)
        bottom_ocr_data = bind_data(bottom_yolox_num, bottom_ocr_data)
        side_ocr_data = bind_data(side_yolox_num, side_ocr_data)
        # 清理key_info中不含数字的数据
        top_ocr_data = filter_ocr_data__2(top_ocr_data)
        bottom_ocr_data = filter_ocr_data__2(bottom_ocr_data)
        side_ocr_data = filter_ocr_data__2(side_ocr_data)
        print("经过bind得到的top视图的SVTR结果:\n", *top_ocr_data, sep='\n')
        print("经过bind得到的bottom视图的SVTR结果:\n", *bottom_ocr_data, sep='\n')
        print("经过bind得到的side视图的SVTR结果:\n", *side_ocr_data, sep='\n')
        # 计算出各个标注的max_medium_min以及是否从符号上就可以判定该标注的语义
        top_ocr_data = cal_max_medium_min_top(top_ocr_data)
        bottom_ocr_data = cal_max_medium_min_bottom(bottom_ocr_data)
        side_ocr_data = cal_max_medium_min_side(side_ocr_data)
        print("经过第二步后处理（yolox）得到的top视图的SVTR结果:\n", *top_ocr_data, sep='\n')
        print("经过第二步后处理（yolox）得到的bottom视图的SVTR结果:\n", *bottom_ocr_data, sep='\n')
        print("经过第二步后处理（yolox）得到的side视图的SVTR结果:\n", *side_ocr_data, sep='\n')
        # 删除公差特别大的标注
        top_ocr_data = filter_ocr_data_12(top_ocr_data)
        bottom_ocr_data = filter_ocr_data_12(bottom_ocr_data)
        side_ocr_data = filter_ocr_data_12(side_ocr_data)
        # 补充test模式下可展示并修改ocr结果
        if key == 1:
            img_path = r"data/top.jpg"
            print("top的ocr结果")
            top_ocr_data = show_ocr_result(img_path, top_ocr_data)  # 展示dbnet框选的尺寸数字
            print("top_ocr_data\n", top_ocr_data)

            img_path = r"data/bottom.jpg"
            print("bottom的ocr结果")
            bottom_ocr_data = show_ocr_result(img_path, bottom_ocr_data)  # 展示dbnet框选的尺寸数字
            print("bottom_ocr_data\n", bottom_ocr_data)

            img_path = r"data/side.jpg"
            print("side的ocr结果")
            side_ocr_data = show_ocr_result(img_path, side_ocr_data)  # 展示dbnet框选的尺寸数字
            print("side_ocr_data\n", side_ocr_data)
        # 6.1.2.2针对各个视图，剔除对参数输出有影响的尺寸数字
        side_ocr_data = BGA_side_filter(side_ocr_data)
        # 7匹配pairs和data
        img_path = f"{DATA}/top.jpg"
        top_ocr_data = match_pairs_data(img_path, top_yolox_pairs, top_ocr_data)
        print("top_ocr_data\n", top_ocr_data)
        if key == 1:
            print("展示top匹配的尺寸线和标注")
            show_matched_pairs_data(img_path, top_ocr_data)
        img_path = f"{DATA}/bottom.jpg"
        bottom_ocr_data = match_pairs_data(img_path, bottom_yolox_pairs, bottom_ocr_data)
        print("bottom_ocr_data\n", bottom_ocr_data)
        if key == 1:
            print("展示bottom匹配的尺寸线和标注")
            show_matched_pairs_data(img_path, bottom_ocr_data)
        img_path = f"{DATA}/side.jpg"
        side_ocr_data = match_pairs_data(img_path, side_yolox_pairs, side_ocr_data)
        print("side_pairs_data\n", side_ocr_data)
        if key == 1:
            print("展示side匹配的尺寸线和标注")
            show_matched_pairs_data(img_path, side_ocr_data)
        # 引线信息和标尺线的种类写入字典
        top_ocr_data = get_yinxian_info(top_ocr_data, top_yolox_pairs_length)
        bottom_ocr_data = get_yinxian_info(bottom_ocr_data, bottom_yolox_pairs_length)
        side_ocr_data = get_yinxian_info(side_ocr_data, side_yolox_pairs_length)
        top_ocr_data = get_pairs_info(top_ocr_data, top_yolox_pairs_copy)
        bottom_ocr_data = get_pairs_info(bottom_ocr_data, bottom_yolox_pairs_copy)
        side_ocr_data = get_pairs_info(side_ocr_data, side_yolox_pairs_copy)
        print("经过第三步后处理（opencv）得到的top视图的SVTR结果:\n", *top_ocr_data, sep='\n')
        print("经过第三步后处理（opencv）得到的bottom视图的SVTR结果:\n", *bottom_ocr_data, sep='\n')
        print("经过第三步后处理（opencv）得到的side视图的SVTR结果:\n", *side_ocr_data, sep='\n')
        # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]
        # top_yolox_pairs_length,np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
        yolox_pairs_top = io_1(top_ocr_data)
        yolox_pairs_bottom = io_1(bottom_ocr_data)
        yolox_pairs_side = io_1(side_ocr_data)
        # print("yolox_pairs_top\n", *yolox_pairs_top, sep='\n')
        # print("yolox_pairs_bottom\n", *yolox_pairs_bottom, sep='\n')
        # print("yolox_pairs_side\n", *yolox_pairs_side, sep='\n')
        # print("serial_numbers_data, serial_letters_data\n", serial_numbers_data, serial_letters_data)
        # # serial_numbers_data: np.(, 5)['x1', 'y1', 'x2', 'y2', 'str']
        return yolox_pairs_top, yolox_pairs_bottom, yolox_pairs_side, \
            top_yolox_pairs_length, bottom_yolox_pairs_length, side_yolox_pairs_length, \
            serial_numbers_data, serial_letters_data, serial_numbers, serial_letters, \
            letter_or_number, top_ocr_data, bottom_ocr_data, side_ocr_data, pin_map, top_border, bottom_border, bottom_pin, color

    if letter_or_number == 'table':
        # 每个尺寸线在两端附近找垂直于标尺线的两个平行直线并计算距离
        # img_path = 'data/top.jpg'
        # top_yolox_pairs_length = find_pairs_length(img_path, top_yolox_pairs, test_mode)
        # # top_yolox_pairs_length np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
        # img_path = 'data/bottom.jpg'
        # bottom_yolox_pairs_length = find_pairs_length(img_path, bottom_yolox_pairs, test_mode)
        # img_path = 'data/side.jpg'
        # side_yolox_pairs_length = find_pairs_length(img_path, side_yolox_pairs, test_mode)
        # 去除标尺线数据中标记着标尺线内外向的数据
        # top_yolox_pairs_copy = top_yolox_pairs.copy()
        # bottom_yolox_pairs_copy = bottom_yolox_pairs.copy()
        # side_yolox_pairs_copy = side_yolox_pairs.copy()
        # top_yolox_pairs = top_yolox_pairs[:, 0:4]
        # # top_yolox_pairs  np.二维数组[x1,y1,x2,y2]
        # bottom_yolox_pairs = bottom_yolox_pairs[:, 0:4]
        # # bottom_yolox_pairs  np.二维数组[x1,y1,x2,y2]
        # side_yolox_pairs = side_yolox_pairs[:, 0:4]
        # # side_yolox_pairs  np.二维数组[x1,y1,x2,y2]
        # if key == 1:
        #     print("展示dbnet检测的文本（删除other）top_dbnet_data")
        #     img_path = r"data/top.jpg"
        #     show_data(img_path, top_dbnet_data)  # 展示dbnet框选的尺寸数字
        #     print("展示dbnet检测的文本（删除other，标记行列序号的数字和字母标注）bottom_dbnet_data")
        #     img_path = r"data/bottom.jpg"
        #     show_data(img_path, bottom_dbnet_data)  # 展示dbnet框选的尺寸数字
        #     print("展示dbnet检测的文本（删除other）side_dbnet_data")
        #     img_path = r"data/side.jpg"
        #     show_data(img_path, side_dbnet_data)  # 展示dbnet框选的尺寸数字
        # 6.表格型BGA处理
        # 6.1ocr识别，
        # path = r"data/top.jpg"
        # top_ocr_data = ocr_data(path, top_dbnet_data)
        path = r"data/bottom.jpg"
        bottom_ocr_data = ocr_data(path, bottom_dbnet_data_serial)
        serial_numbers_data, serial_letters_data, bottom_ocr_data = filter_bottom_ocr_data(bottom_ocr_data,
                                                                                           bottom_dbnet_data_serial,
                                                                                           serial_numbers_data,
                                                                                           serial_letters_data,
                                                                                           bottom_dbnet_data)

        # path = r"data/side.jpg"
        # side_ocr_data = ocr_data(path, side_dbnet_data)
        # 6.2ocr后处理，输出各个标注'字母'+'数字'
        # 编辑为字典类型
        # top_ocr_data = convert_Dic(top_dbnet_data, top_ocr_data)
        # # [{'location': array([1245., 88., 1302., 135.]), 'ocr_strings': ''},{'location': array([635., 110., 725., 156.]), 'ocr_strings': '15.0'}]
        # bottom_ocr_data = convert_Dic(bottom_dbnet_data, bottom_ocr_data)
        # print("side_dbnet_data, side_ocr_data", side_dbnet_data, side_ocr_data)
        # side_ocr_data = convert_Dic(side_dbnet_data, side_ocr_data)
        # # 筛除非参数信息
        # # 删除过长字符
        # top_ocr_data = shilter_table_data(top_ocr_data)
        # bottom_ocr_data = shilter_table_data(bottom_ocr_data)
        # side_ocr_data = shilter_table_data(side_ocr_data)
        # # 删除SVTR识别为空
        # top_ocr_data = filter_ocr_data_0(top_ocr_data)
        # bottom_ocr_data = filter_ocr_data_0(bottom_ocr_data)
        # side_ocr_data = filter_ocr_data_0(side_ocr_data)
        # # 删除’数字‘+’X‘（没有’=‘时）
        # top_ocr_data = filter_ocr_data_1(top_ocr_data)
        # bottom_ocr_data = filter_ocr_data_1(bottom_ocr_data)
        # side_ocr_data = filter_ocr_data_1(side_ocr_data)
        # # 删除’PIN1‘
        # top_ocr_data = filter_ocr_data_2(top_ocr_data)
        # bottom_ocr_data = filter_ocr_data_2(bottom_ocr_data)
        # side_ocr_data = filter_ocr_data_2(side_ocr_data)
        # # 找‘Φ’，并删除'Φ'然后标识'absolute'
        # top_ocr_data = filter_ocr_data_8(top_ocr_data)
        # bottom_ocr_data = filter_ocr_data_8(bottom_ocr_data)
        # side_ocr_data = filter_ocr_data_8(side_ocr_data)
        # # 删除SVTR识别为空
        # top_ocr_data = filter_ocr_data_0(top_ocr_data)
        # bottom_ocr_data = filter_ocr_data_0(bottom_ocr_data)
        # side_ocr_data = filter_ocr_data_0(side_ocr_data)
        # # 补充test模式下可展示并修改ocr结果
        # if key == 1:
        #     img_path = r"data/top.jpg"
        #     print("top的ocr结果")
        #     top_dbnet_data = show_ocr_result_table(img_path, top_dbnet_data)  # 展示dbnet框选的尺寸数字
        #     # top_dbnet_data = delete_ocr_zeros(top_dbnet_data)
        #     print("top_dbnet_data\n", top_dbnet_data)
        #
        #     img_path = r"data/bottom.jpg"
        #     print("bottom的ocr结果")
        #     bottom_dbnet_data = show_ocr_result_table(img_path, bottom_dbnet_data)  # 展示dbnet框选的尺寸数字
        #     # bottom_dbnet_data = delete_ocr_zeros(bottom_dbnet_data)
        #     print("bottom_dbnet_data\n", bottom_dbnet_data)
        #
        #     img_path = r"data/side.jpg"
        #     print("side的ocr结果")
        #     side_dbnet_data = show_ocr_result_table(img_path, side_dbnet_data)  # 展示dbnet框选的尺寸数字
        #     # side_dbnet_data = delete_ocr_zeros(side_dbnet_data)
        #     print("side_dbnet_data\n", side_dbnet_data)
        # 转换ocr_data为[['0','1','2','3','A1'],['0','1','2','3','B1']]格式
        # top_dbnet_data = io_2(top_ocr_data)
        # bottom_dbnet_data = io_2(bottom_ocr_data)
        # side_dbnet_data = io_2(side_ocr_data)
        # 7匹配pairs和data
        # img_path = r"data/top.jpg"
        # top_pairs_data = match_pairs_data_table(top_yolox_pairs, top_dbnet_data)
        # print("top_pairs_data\n", top_pairs_data)
        # if key == 1:
        #     print("展示top匹配的尺寸线和尺寸数字")
        #     show_matched_pairs_data_table(img_path, top_pairs_data)
        #
        # img_path = r"data/bottom.jpg"
        # bottom_pairs_data = match_pairs_data_table(bottom_yolox_pairs, bottom_dbnet_data)
        # print("bottom_pairs_data\n", bottom_pairs_data)
        # if key == 1:
        #     print("展示bottom匹配的尺寸线和尺寸数字")
        #     show_matched_pairs_data_table(img_path, bottom_pairs_data)
        #
        # img_path = r"data/side.jpg"
        # side_pairs_data = match_pairs_data_table(side_yolox_pairs, side_dbnet_data)
        # print("side_pairs_data\n", side_pairs_data)
        # if key == 1:
        #     print("展示side匹配的尺寸线和尺寸数字")
        #     show_matched_pairs_data_table(img_path, side_pairs_data)
        top_pairs_data = 0
        bottom_pairs_data = 0
        side_pairs_data = 0
        top_yolox_pairs_length = 0
        bottom_yolox_pairs_length = 0
        side_yolox_pairs_length = 0
        top_ocr_data = 0
        side_ocr_data = 0

        return top_pairs_data, bottom_pairs_data, side_pairs_data, top_yolox_pairs_length, bottom_yolox_pairs_length, side_yolox_pairs_length, serial_numbers_data, serial_letters_data, serial_numbers, serial_letters, letter_or_number, top_ocr_data, bottom_ocr_data, side_ocr_data
        # yolox_pairs_top, yolox_pairs_bottom, yolox_pairs_side, top_yolox_pairs_length, bottom_yolox_pairs_length, side_yolox_pairs_length, serial_numbers_data, serial_letters_data, serial_numbers, serial_letters, letter_or_number, top_ocr_data, bottom_ocr_data, side_ocr_data


def find_pin_diameter(pin_diameter, high, top_data_np, bottom_data_np, side_data_np, pitch_x, pitch_y):
    """根据候选数据挑选最合适的 pin 直径。"""
    pin_diameter = np.zeros((0, 3))
    pin_diameter_1 = np.zeros((0, 3))  # 存储可能的pin直径值，
    for i in range(len(bottom_data_np)):
        if (bottom_data_np[i][1] < pitch_x).any() and (bottom_data_np[i][1] < pitch_y).any() and bottom_data_np[i,
                                                                                                 1:4] not in high and bottom_data_np[
                                                                                                                      i,
                                                                                                                      1:4] not in pin_diameter_1:  # 如果data最大值比pitch值小
            pin_diameter_1 = np.r_[pin_diameter_1, [bottom_data_np[i, 1:4]]]
    for i in range(len(side_data_np)):
        if (side_data_np[i][1] < pitch_x).any() and (side_data_np[i][1] < pitch_y).any() and side_data_np[i,
                                                                                             1:4] not in high and side_data_np[
                                                                                                                  i,
                                                                                                                  1:4] not in pin_diameter_1:  # 如果data最大值比pitch值小
            pin_diameter_1 = np.r_[pin_diameter_1, [side_data_np[i, 1:4]]]
    for i in range(len(top_data_np)):
        if (top_data_np[i][1] < pitch_x).any() and (top_data_np[i][1] < pitch_y).any() and top_data_np[i,
                                                                                           1:4] not in high and top_data_np[
                                                                                                                i,
                                                                                                                1:4] not in pin_diameter_1:  # 如果data最大值比pitch值小
            pin_diameter_1 = np.r_[pin_diameter_1, [top_data_np[i, 1:4]]]
    pin_diameter_1 = pin_diameter_1[np.argsort(-pin_diameter_1[:, 0])]
    if len(pin_diameter_1) != 0:
        pin_diameter = np.r_[pin_diameter, [pin_diameter_1[0]]]
    # 洗去重复项
    if len(pin_diameter) > 1:
        pin_diameter_2 = np.zeros((0, 3))
        for i in range(len(pin_diameter)):
            if pin_diameter[i] not in pin_diameter_2:
                pin_diameter_2 = np.r_[pin_diameter_2, [pin_diameter[i]]]
        pin_diameter = pin_diameter_2
    # 进一步筛选
    pin_diameter_3 = np.zeros((0, 3))
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if ((pin_diameter[i] / pitch_x[0]) > 0.85).any() or ((pin_diameter[i] / pitch_x[0]) < 0.45).any():
                print("filter")
            else:
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        pin_diameter = pin_diameter_3
    pin_diameter_3 = np.zeros((0, 3))
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if ((pin_diameter[i] / pitch_x[0]) > 0.80).any() or ((pin_diameter[i] / pitch_x[0]) < 0.45).any():
                print("filter")
            else:
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        if len(pin_diameter_3) != 0 and len(pin_diameter_3) != len(pin_diameter):
            pin_diameter = pin_diameter_3
    pin_diameter_3 = np.zeros((0, 3))
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if ((pin_diameter[i] / pitch_x[0]) > 0.75).any() or ((pin_diameter[i] / pitch_x[0]) < 0.45).any():
                print("filter")
            else:
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        if len(pin_diameter_3) != 0 and len(pin_diameter_3) != len(pin_diameter):
            pin_diameter = pin_diameter_3
    # 进一步筛选如果多个尺寸数字中仅有一个含误差的，确定含误差为直径
    pin_diameter_3 = np.zeros((0, 3))
    j = 0
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if pin_diameter[i][0] == pin_diameter[i][1] == pin_diameter[i][2]:
                print("filter")
            else:
                j += 1
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        if j == 1:
            pin_diameter = pin_diameter_3
    return pin_diameter


def empty_folder(folder_path):
    """清空指定文件夹内容。"""
    try:
        shutil.rmtree(folder_path)
    except FileNotFoundError:
        print(f"文件夹 {folder_path} 不存在！")


# def find_body(top_pairs_data, bottom_pairs_data):
#     # 1.yolox检测top和bottom中的实体外框
#     from output_top_body_location import begain_output_top_body_location
#     from output_bottom_body_location import begain_output_bottom_body_location
#     top_body = begain_output_top_body_location()
#     bottom_body = begain_output_bottom_body_location()
#     # 补充：将匹配得到的标注添加到引线中
#     new_top_yolox_pairs_length = np.zeros((len(top_yolox_pairs_length), 16))
#     middle = np.zeros((16))
#     for i in range(len(top_yolox_pairs_length)):
#         for j in range(len(yolox_pairs_top)):
#             if (yolox_pairs_top[j, 0:3] == top_yolox_pairs_length[i, 0:3]).all():
#                 middle[0:13] = top_yolox_pairs_length[i]
#                 middle[13:16] = yolox_pairs_top[j, -3:]
#                 new_top_yolox_pairs_length = np.r_[new_top_yolox_pairs_length, [middle]]
#     new_bottom_yolox_pairs_length = np.zeros((len(bottom_yolox_pairs_length), 16))
#     middle = np.zeros((16))
#     for i in range(len(bottom_yolox_pairs_length)):
#         for j in range(len(yolox_pairs_bottom)):
#             if (yolox_pairs_bottom[j, 0:3] == bottom_yolox_pairs_length[i, 0:3]).all():
#                 middle[0:13] = bottom_yolox_pairs_length[i]
#                 middle[13:16] = yolox_pairs_bottom[j, -3:]
#                 new_bottom_yolox_pairs_length = np.r_[new_bottom_yolox_pairs_length, [middle]]
#     # 2.引线下引top找body
#     gao = np.zeros((3))
#     kuan = np.zeros((3))
#     ratio = 0.2
#     for i in range(len(new_top_yolox_pairs_length)):
#         ruler_1 = 0
#         ruler_2 = 0
#         if abs(new_top_yolox_pairs_length[i][0] - new_top_yolox_pairs_length[i][2]) > abs(
#                 new_top_yolox_pairs_length[i][1] - new_top_yolox_pairs_length[i][3]):
#             ruler_1 = new_top_yolox_pairs_length[i][4]
#             ruler_2 = new_top_yolox_pairs_length[i][8]
#
#             ruler_3 = top_body[j][0]
#             ruler_4 = top_body[j][2]
#             if (abs(ruler_1 - ruler_3) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
#                 new_top_yolox_pairs_length[i][12] < ratio) or (
#                     abs(ruler_1 - ruler_4) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
#                     new_top_yolox_pairs_length[i][12] < ratio):
#                 print("引线下引到实体top_body，找到body_x参数")
#                 kuan = new_top_yolox_pairs_length[i][-3:]
#         else:
#             ruler_1 = new_top_yolox_pairs_length[i][5]
#             ruler_2 = new_top_yolox_pairs_length[i][9]
#             ruler_3 = top_body[j][1]
#             ruler_4 = top_body[j][3]
#             if (abs(ruler_1 - ruler_3) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
#                 new_top_yolox_pairs_length[i][12] < ratio) or (
#                     abs(ruler_1 - ruler_4) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
#                     new_top_yolox_pairs_length[i][12] < ratio):
#                 print("引线下引到实体top_body，找到body_y参数")
#                 gao = new_top_yolox_pairs_length[i][-3:]
#     # 3.引线下引bottom找body
#     bottom_gao = np.zeros((3))
#     bottom_kaun = np.zeros((3))
#     ratio = 0.2
#     for i in range(len(new_bottom_yolox_pairs_length)):
#         ruler_1 = 0
#         ruler_2 = 0
#         if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) > abs(
#                 new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):
#             ruler_1 = new_bottom_yolox_pairs_length[i][4]
#             ruler_2 = new_bottom_yolox_pairs_length[i][8]
#
#             ruler_3 = bottom_body[j][0]
#             ruler_4 = bottom_body[j][2]
#             if (abs(ruler_1 - ruler_3) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
#                 new_bottom_yolox_pairs_length[i][12] < ratio) or (
#                     abs(ruler_1 - ruler_4) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
#                     new_bottom_yolox_pairs_length[i][12] < ratio):
#                 print("引线下引到实体bottom_body，找到body_x参数")
#                 bottom_kuan = new_bottom_yolox_pairs_length[i][-3:]
#         else:
#             ruler_1 = new_bottom_yolox_pairs_length[i][5]
#             ruler_2 = new_bottom_yolox_pairs_length[i][9]
#             ruler_3 = bottom_body[j][1]
#             ruler_4 = bottom_body[j][3]
#             if (abs(ruler_1 - ruler_3) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
#                 new_bottom_yolox_pairs_length[i][12] < ratio) or (
#                     abs(ruler_1 - ruler_4) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
#                     new_bottom_yolox_pairs_length[i][12] < ratio):
#                 print("引线下引到实体bottom_body，找到body_y参数")
#                 bottom_gao = new_bottom_yolox_pairs_length[i][-3:]
#     # 3.引线下引方法找不到时，用引线距离是否与外框近似
#     zer = np.zeros((3))
#     top_body_heng = top_body[0][2] - top_body[0][0]
#     top_body_shu = top_body[0][3] - top_body[0][1]
#     bottom_body_heng = bottom_body[0][2] - bottom_body[0][0]
#     bottom_body_shu = bottom_body[0][3] - bottom_body[0][1]
#     ratio = 0.16
#     if (kuan == zer).all():
#         for i in range(len(top_pairs_data)):
#             if abs((top_pairs_data[i][2] - top_pairs_data[i][0]) - top_body_heng) / top_body_heng < ratio:
#                 kuan = top_pairs_data[i, -3:]
#     if (gao == zer).all():
#         for i in range(len(top_pairs_data)):
#             if abs((top_pairs_data[i][3] - top_pairs_data[i][1]) - top_body_shu) / top_body_shu < ratio:
#                 gao = top_pairs_data[i, -3:]
#     if (bottom_kuan == zer).all():
#         for i in range(len(bottom_pairs_data)):
#             if abs((bottom_pairs_data[i][2] - bottom_pairs_data[i][0]) - bottom_body_heng) / bottom_body_heng < ratio:
#                 bottom_kuan = bottom_pairs_data[i, -3:]
#     if (bottom_gao == zer).all():
#         for i in range(len(bottom_pairs_data)):
#             if abs((bottom_pairs_data[i][3] - bottom_pairs_data[i][1]) - bottom_body_shu) / bottom_body_shu < ratio:
#                 bottom_gao = bottom_pairs_data[i, -3:]
#
#     # top和bottom互补
#     if (kuan == zer).all() and (bottom_kuan != zer).any():
#         kuan = bottom_kuan
#     if (bottom_kuan == zer).all() and (kuan != zer).any():
#         bottom_kuan = kuan
#     if (gao == zer).all() and (bottom_gao != zer).any():
#         gao = bottom_gao
#     if (bottom_gao == zer).all() and (gao != zer).any():
#         bottom_gao = gao
#     # top的x和y互补
#     if (kuan == zer).all() and (gao != zer).any():
#         kuan = gao
#     if (gao == zer).all() and (kuan != zer).any():
#         gao = kuan
#     # bottom的x和y互补
#     if (bottom_kuan == zer).all() and (bottom_gao != zer).any():
#         bottom_kuan = bottom_gao
#     if (bottom_gao == zer).all() and (bottom_kuan != zer).any():
#         bottom_gao = bottom_kuan
#     # bottom和top中找最合适的
#     body_x = kuan
#     body_y = gao
#     return body_x, body_y


def yinXinan_find_pitch(yolox_pairs_bottom, bottom_yolox_pairs_length, pin):
    """在引线识别结果中寻找行列间距。"""
    # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]
    # top_yolox_pairs_length,np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
    '''
    print("---开始用引线方法寻找pitch---")
    pitch_max = 2.5  # 限定最大pitch值
    # 补充：将匹配得到的标注添加到引线中
    new_bottom_yolox_pairs_length = np.zeros((len(bottom_yolox_pairs_length), 16))
    middle = np.zeros((16))
    for i in range(len(bottom_yolox_pairs_length)):
        for j in range(len(yolox_pairs_bottom)):
            if (yolox_pairs_bottom[j, 0:3] == bottom_yolox_pairs_length[i, 0:3]).all() and yolox_pairs_bottom[j][8] == \
                    yolox_pairs_bottom[j][9] == yolox_pairs_bottom[j][10]:
                middle[0:13] = bottom_yolox_pairs_length[i]
                middle[13:16] = yolox_pairs_bottom[j, -3:]
                new_bottom_yolox_pairs_length = np.r_[new_bottom_yolox_pairs_length, [middle]]
    # 只筛选出max=medium=min的值

    # #按引线间距离从小到大排序
    # bottom_yolox_pairs_length_new = new_bottom_yolox_pairs_length[np.argsort(new_bottom_yolox_pairs_length[:, 12])]
    # #筛出较长的标尺线
    # bottom_yolox_pairs_length_new_new = np.zeros((0, 16))
    # for i in range(len(bottom_yolox_pairs_length_new)):
    #     if bottom_yolox_pairs_length_new[i][12] < bottom_yolox_pairs_length_new[0][12] * 2:
    #         bottom_yolox_pairs_length_new_new = np.r_[bottom_yolox_pairs_length_new_new, [bottom_yolox_pairs_length_new[i]]]
    # new_bottom_yolox_pairs_length = bottom_yolox_pairs_length_new_new
    #
    # from output_pin_yinXian_find_pitch import begain_output_pin_location
    # pin = begain_output_pin_location()
    # 计算pin的平均宽度
    px = 0
    py = 0
    for i in range(len(pin)):
        px = px + abs(pin[i][2] - pin[i][0])
        py = py + abs(pin[i][3] - pin[i][1])
    px = px / len(pin)
    py = py / len(pin)
    # 计算平均pitch
    for_num = 6  # 随机抽取次数
    # import random
    # from random import randint
    x_pitch_1 = 9999  # 图片中的pin之间的距离
    y_pitch_1 = 9999
    for i in range(for_num):
        pin_no = randint(0, len(pin) - 1)
        x_middle = (pin[pin_no][2] + pin[pin_no][0]) / 2
        y_middle = (pin[pin_no][3] + pin[pin_no][1]) / 2

        for j in range(len(pin)):
            if j != pin_no:
                x_j_middle = (pin[j][2] + pin[j][0]) / 2
                y_j_middle = (pin[j][3] + pin[j][1]) / 2
                x_pitch_2 = abs(x_j_middle - x_middle)
                y_pitch_2 = abs(y_j_middle - y_middle)
                # print((x_pitch_2))
                if px < x_pitch_2 and x_pitch_2 < x_pitch_1:
                    x_pitch_1 = x_pitch_2
                if py < y_pitch_2 and y_pitch_2 < y_pitch_1:
                    y_pitch_1 = y_pitch_2
    #
    pitch_x = np.zeros(3)
    pitch_y = np.zeros(3)
    mb_pitch_x = np.zeros((0, 3))  # 存储可能的pitch
    acc_mb_x = []  # 存储可能的pitch匹配到的次数
    mb_pitch_y = np.zeros((0, 3))
    acc_mb_y = []
    for i in range(len(new_bottom_yolox_pairs_length)):
        ruler_1 = 0
        ruler_2 = 0
        ratio = 0.2
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) > abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):  # 横向标尺线
            ruler_1 = new_bottom_yolox_pairs_length[i][4]
            ruler_2 = new_bottom_yolox_pairs_length[i][8]
            for j in range(len(pin)):
                if abs(ruler_1 - (pin[j][0] + pin[j][2]) * 0.5) / abs(pin[j][0] - pin[j][2]) < ratio:
                    for k in range(len(pin)):
                        if k != j:
                            if abs(ruler_2 - (pin[k][0] + pin[k][2]) * 0.5) / abs(pin[k][0] - pin[k][2]) < ratio:
                                if abs(abs((pin[k][0] + pin[k][2]) * 0.5 - (
                                        pin[j][0] + pin[j][2]) * 0.5) - x_pitch_1) / x_pitch_1 < ratio and (
                                        new_bottom_yolox_pairs_length[i, -3:] < pitch_max).all():
                                    # pitch_x = new_bottom_yolox_pairs_length[i, -3:]
                                    if new_bottom_yolox_pairs_length[i, -3:] not in mb_pitch_x:
                                        mb_pitch_x = np.r_[mb_pitch_x, [new_bottom_yolox_pairs_length[i, -3:]]]
                                        acc_mb_x.append(1)
                                    else:
                                        for p in range(len(mb_pitch_x)):
                                            if np.array_equal(mb_pitch_x[p], new_bottom_yolox_pairs_length[i, -3:]):
                                                acc_mb_x[p] += 1
                                    # print("引线下引找到pitch_x", pitch_x)
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) < abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):  # 竖向标尺线
            ruler_1 = new_bottom_yolox_pairs_length[i][5]
            ruler_2 = new_bottom_yolox_pairs_length[i][9]
            for j in range(len(pin)):
                if abs(ruler_1 - (pin[j][1] + pin[j][3]) * 0.5) / abs(pin[j][1] - pin[j][3]) < ratio:
                    for k in range(len(pin)):
                        if k != j:
                            if abs(ruler_2 - (pin[k][1] + pin[k][3]) * 0.5) / abs(pin[k][1] - pin[k][3]) < ratio:
                                if abs(abs((pin[k][1] + pin[k][3]) * 0.5 - (
                                        pin[j][1] + pin[j][3]) * 0.5) - y_pitch_1) / y_pitch_1 < ratio and (
                                        new_bottom_yolox_pairs_length[i, -3:] < pitch_max).all():
                                    if new_bottom_yolox_pairs_length[i, -3:] not in mb_pitch_y:
                                        mb_pitch_y = np.r_[mb_pitch_y, [new_bottom_yolox_pairs_length[i, -3:]]]
                                        acc_mb_y.append(1)
                                    else:
                                        for p in range(len(mb_pitch_y)):
                                            if np.array_equal(mb_pitch_y[p], new_bottom_yolox_pairs_length[i, -3:]):
                                                acc_mb_y[p] += 1
                                    # pitch_y = new_bottom_yolox_pairs_length[i, -3:]
                                    # print("引线下引找到pitch_y", pitch_y)
    try:
        max_value = max(acc_mb_x)  # 求列表最大值
        max_idx = acc_mb_x.index(max_value)  # 求最大值对应索引
        pitch_x = mb_pitch_x[max_idx]
        print("引线下引找到pitch_x:", pitch_x)
    except:
        pass
    try:
        max_value = max(acc_mb_y)  # 求列表最大值
        max_idx = acc_mb_x.index(max_value)  # 求最大值对应索引
        pitch_y = mb_pitch_y[max_idx]
        print("引线下引找到pitch_y:", pitch_y)
    except:
        pass

    # 如果引线位置匹配不到数值，那么根据引线长度匹配
    ze = np.zeros(3)
    ratio = 0.18
    if (pitch_x == ze).all():
        for i in range(len(new_bottom_yolox_pairs_length)):
            if abs(new_bottom_yolox_pairs_length[i][12] - x_pitch_1) / x_pitch_1 < ratio and (
                    new_bottom_yolox_pairs_length[i, -3:] < pitch_max).all():
                pitch_x = new_bottom_yolox_pairs_length[i][-3:]
                print("引线之间距离近似pitch_x，找到pitch_x参数:", pitch_x)
    if (pitch_y == ze).all():
        for i in range(len(new_bottom_yolox_pairs_length)):
            if abs(new_bottom_yolox_pairs_length[i][12] - y_pitch_1) / y_pitch_1 < ratio and (
                    new_bottom_yolox_pairs_length[i, -3:] < pitch_max).all():
                pitch_y = new_bottom_yolox_pairs_length[i][-3:]
                print("引线之间距离近似pitch_y，找到pitch_y参数:", pitch_y)
    # pitch_x和pitch_y互补
    zer = np.zeros(3)
    if (pitch_x == zer).all() and (pitch_y != zer).any():
        pitch_x = pitch_y
    if (pitch_y == zer).all() and (pitch_x != zer).any():
        pitch_y = pitch_x
    # 将pitch改为一维数组中的一个数
    if (pitch_x != zer).any():
        pitch_x = np.array([pitch_x[0]])
    else:
        pitch_x = np.array([0])
    if (pitch_y != zer).any():
        pitch_y = np.array([pitch_y[0]])
    else:
        pitch_y = np.array([0])
    print("---结束用引线方法寻找pitch---")
    return pitch_x, pitch_y


def yinXinan_find_pitch_table(yolox_pairs_bottom, bottom_yolox_pairs_length):
    """在表格模式下计算行列间距。"""
    # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]
    # top_yolox_pairs_length,np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
    '''
    print("---开始用引线方法寻找pitch---")
    pitch_max = 2.5
    pitch_str = np.empty(len(bottom_yolox_pairs_length), dtype=np.dtype('U10'))
    # pitch_str记录每个找到引线的标尺线匹配的标注字符串
    new_bottom_yolox_pairs_length = bottom_yolox_pairs_length
    new_bottom_yolox_pairs_length_str = bottom_yolox_pairs_length.astype(str)

    for i in range(len(bottom_yolox_pairs_length)):
        key = 0
        for j in range(len(yolox_pairs_bottom)):
            if (new_bottom_yolox_pairs_length_str[i, 0: 4] == yolox_pairs_bottom[j, 0: 4]).all():
                key = 1
                pitch_str[i] = yolox_pairs_bottom[j, 8]
        if key == 0:
            pitch_str[i] = ''

    pin = begain_output_pin_location()
    # 计算pin的平均宽度
    px = 0
    py = 0
    for i in range(len(pin)):
        px = px + abs(pin[i][2] - pin[i][0])
        py = py + abs(pin[i][3] - pin[i][1])
    px = px / len(pin)
    py = py / len(pin)
    # 计算平均pitch
    for_num = 6  # 随机抽取次数
    # import random
    # from random import randint
    x_pitch_1 = 9999  # 图片中的pin之间的距离
    y_pitch_1 = 9999
    for i in range(for_num):
        pin_no = randint(0, len(pin) - 1)
        x_middle = (pin[pin_no][2] + pin[pin_no][0]) / 2
        y_middle = (pin[pin_no][3] + pin[pin_no][1]) / 2

        for j in range(len(pin)):
            if j != pin_no:
                x_j_middle = (pin[j][2] + pin[j][0]) / 2
                y_j_middle = (pin[j][3] + pin[j][1]) / 2
                x_pitch_2 = abs(x_j_middle - x_middle)
                y_pitch_2 = abs(y_j_middle - y_middle)
                # print((x_pitch_2))
                if px < x_pitch_2 and x_pitch_2 < x_pitch_1:
                    x_pitch_1 = x_pitch_2
                if py < y_pitch_2 and y_pitch_2 < y_pitch_1:
                    y_pitch_1 = y_pitch_2
    #
    pitch_x = ''
    pitch_y = ''
    for i in range(len(new_bottom_yolox_pairs_length)):
        ruler_1 = 0
        ruler_2 = 0
        ratio = 0.2
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) > abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):  # 横向标尺线
            ruler_1 = new_bottom_yolox_pairs_length[i][4]
            ruler_2 = new_bottom_yolox_pairs_length[i][8]
            for j in range(len(pin)):
                if abs(ruler_1 - (pin[j][0] + pin[j][2]) * 0.5) / abs(pin[j][0] - pin[j][2]) < ratio:
                    for k in range(len(pin)):
                        if k != j:
                            if abs(ruler_2 - (pin[k][0] + pin[k][2]) * 0.5) / abs(pin[k][0] - pin[k][2]) < ratio:
                                if abs(abs((pin[k][0] + pin[k][2]) * 0.5 - (
                                        pin[j][0] + pin[j][2]) * 0.5) - x_pitch_1) / x_pitch_1 < ratio:
                                    pitch_x = pitch_str[i]
                                    print("引线下引找到pitch_x", pitch_x)
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) < abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):  # 竖向标尺线
            ruler_1 = new_bottom_yolox_pairs_length[i][5]
            ruler_2 = new_bottom_yolox_pairs_length[i][9]
            for j in range(len(pin)):
                if abs(ruler_1 - (pin[j][1] + pin[j][3]) * 0.5) / abs(pin[j][1] - pin[j][3]) < ratio:
                    for k in range(len(pin)):
                        if k != j:
                            if abs(ruler_2 - (pin[k][1] + pin[k][3]) * 0.5) / abs(pin[k][1] - pin[k][3]) < ratio:
                                if abs(abs((pin[k][1] + pin[k][3]) * 0.5 - (
                                        pin[j][1] + pin[j][3]) * 0.5) - y_pitch_1) / y_pitch_1 < ratio:
                                    pitch_y = pitch_str[i]
                                    print("引线下引找到pitch_y", pitch_y)
    # 如果引线位置匹配不到数值，那么根据引线长度匹配
    ze = ''
    ratio = 0.18
    if pitch_x == ze:
        for i in range(len(new_bottom_yolox_pairs_length)):
            if abs(new_bottom_yolox_pairs_length[i][12] - x_pitch_1) / x_pitch_1 < ratio:
                print("引线之间距离近似pitch_x，找到pitch_x参数")
                pitch_x = pitch_str[i]
    if pitch_y == ze:
        for i in range(len(new_bottom_yolox_pairs_length)):
            if abs(new_bottom_yolox_pairs_length[i][12] - y_pitch_1) / y_pitch_1 < ratio:
                print("引线之间距离近似pitch_y，找到pitch_y参数")
                pitch_y = pitch_str[i]
    # pitch_x和pitch_y互补
    zer = ''
    if pitch_x == zer and pitch_y != zer:
        pitch_x = pitch_y
    if pitch_y == zer and pitch_x != zer:
        pitch_y = pitch_x
    # 将pitch改为一维数组中的一个数
    print("---结束用引线方法寻找pitch---")
    return pitch_x, pitch_y


def yinXinan_find_pin_diameter(yolox_pairs_bottom, bottom_yolox_pairs_length, pin):
    """根据引线信息估算 pin 直径。"""
        # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]
        # top_yolox_pairs_length,np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
    '''
    print("---开始用引线方法寻找pin_diameter---")
    # 补充：将匹配得到的标注添加到引线中
    new_bottom_yolox_pairs_length = np.zeros((len(bottom_yolox_pairs_length), 16))
    middle = np.zeros((16))
    for i in range(len(bottom_yolox_pairs_length)):
        for j in range(len(yolox_pairs_bottom)):
            if (yolox_pairs_bottom[j, 0:3] == bottom_yolox_pairs_length[i, 0:3]).all():
                middle[0:13] = bottom_yolox_pairs_length[i]
                middle[13:16] = yolox_pairs_bottom[j, -3:]
                new_bottom_yolox_pairs_length = np.r_[new_bottom_yolox_pairs_length, [middle]]

    # #按引线间距离从小到大排序
    # bottom_yolox_pairs_length_new = new_bottom_yolox_pairs_length[np.argsort(new_bottom_yolox_pairs_length[:, 12])]
    # #筛出较长的标尺线
    # bottom_yolox_pairs_length_new_new = np.zeros((0, 16))
    # for i in range(len(bottom_yolox_pairs_length_new)):
    #     if bottom_yolox_pairs_length_new[i][12] < bottom_yolox_pairs_length_new[0][12] * 2:
    #         bottom_yolox_pairs_length_new_new = np.r_[bottom_yolox_pairs_length_new_new, [bottom_yolox_pairs_length_new[i]]]
    # new_bottom_yolox_pairs_length = bottom_yolox_pairs_length_new_new
    #
    # from output_pin_yinXian_find_pitch import begain_output_pin_location
    # pin = begain_output_pin_location()
    # 计算pin的平均宽度
    px = 0
    py = 0
    for i in range(len(pin)):
        px = px + abs(pin[i][2] - pin[i][0])
        py = py + abs(pin[i][3] - pin[i][1])
    px = px / len(pin)
    py = py / len(pin)

    pin_diameter = np.zeros(3)
    for i in range(len(new_bottom_yolox_pairs_length)):
        ruler_1 = 0
        ruler_2 = 0
        ratio = 0.2
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) > abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):  # 横向标尺线
            ruler_1 = new_bottom_yolox_pairs_length[i][4]
            ruler_2 = new_bottom_yolox_pairs_length[i][8]
            for j in range(len(pin)):
                if (abs(ruler_1 - pin[j][0]) / abs(pin[j][0] - pin[j][2]) < ratio and abs(ruler_2 - pin[j][2]) / abs(
                        pin[j][0] - pin[j][2]) < ratio) or (
                        abs(ruler_1 - pin[j][2]) / abs(pin[j][0] - pin[j][2]) < ratio and abs(
                    ruler_2 - pin[j][0]) / abs(pin[j][0] - pin[j][2]) < ratio):
                    pin_diameter = new_bottom_yolox_pairs_length[i][-3:]
                    print("引线下引找到pin_diameter", pin_diameter)
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) < abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):  # 竖向标尺线
            ruler_1 = new_bottom_yolox_pairs_length[i][5]
            ruler_2 = new_bottom_yolox_pairs_length[i][9]
            for j in range(len(pin)):
                if (abs(ruler_1 - pin[j][1]) / abs(pin[j][0] - pin[j][2]) < ratio and abs(ruler_2 - pin[j][3]) / abs(
                        pin[j][0] - pin[j][2]) < ratio) or (
                        abs(ruler_1 - pin[j][3]) / abs(pin[j][0] - pin[j][2]) < ratio and abs(
                    ruler_2 - pin[j][1]) / abs(pin[j][0] - pin[j][2]) < ratio):
                    pin_diameter = new_bottom_yolox_pairs_length[i][-3:]
                    print("引线下引找到pin_diameter", pin_diameter)
    # 如果引线位置匹配不到数值，那么根据引线长度匹配
    ze = np.zeros((3))
    ratio = 0.18
    if (pin_diameter == ze).all():
        for i in range(len(new_bottom_yolox_pairs_length)):
            if abs(new_bottom_yolox_pairs_length[i][12] - px) / px < ratio or abs(
                    new_bottom_yolox_pairs_length[i][12] - py) / py < ratio:
                pin_diameter = new_bottom_yolox_pairs_length[i][-3:]
                print("引线之间距离近似pin_diameter，找到pin_diameter参数:", pin_diameter)
    print("---结束用引线方法寻找pin_diameter---")
    return pin_diameter


def yinXinan_find_pin_diameter_table(yolox_pairs_bottom, bottom_yolox_pairs_length):
    """在表格模式下估算 pin 直径。"""
        # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]
        # top_yolox_pairs_length,np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
    '''
    print("---开始用引线方法寻找pin_diameter---")
    bottom_str = np.empty(len(bottom_yolox_pairs_length), dtype=np.dtype('U10'))
    # top_str， bottom_str记录每个找到引线的标尺线匹配的标注字符串
    new_bottom_yolox_pairs_length = bottom_yolox_pairs_length
    new_bottom_yolox_pairs_length_str = bottom_yolox_pairs_length.astype(str)
    for i in range(len(bottom_yolox_pairs_length)):
        key = 0
        for j in range(len(yolox_pairs_bottom)):
            if (new_bottom_yolox_pairs_length_str[i, 0: 4] == yolox_pairs_bottom[j, 0: 4]).all():
                key = 1
                bottom_str[i] = yolox_pairs_bottom[j, 8]
        if key == 0:
            bottom_str[i] = ''

    # from output_pin_yinXian_find_pitch import begain_output_pin_location
    pin = begain_output_pin_location()
    # 计算pin的平均宽度
    px = 0
    py = 0
    for i in range(len(pin)):
        px = px + abs(pin[i][2] - pin[i][0])
        py = py + abs(pin[i][3] - pin[i][1])
    px = px / len(pin)
    py = py / len(pin)

    pin_diameter = ''
    for i in range(len(new_bottom_yolox_pairs_length)):
        ruler_1 = 0
        ruler_2 = 0
        ratio = 0.2
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) > abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):  # 横向标尺线
            ruler_1 = new_bottom_yolox_pairs_length[i][4]
            ruler_2 = new_bottom_yolox_pairs_length[i][8]
            for j in range(len(pin)):
                if (abs(ruler_1 - pin[j][0]) / abs(pin[j][0] - pin[j][2]) < ratio and abs(ruler_2 - pin[j][2]) / abs(
                        pin[j][0] - pin[j][2]) < ratio) or (
                        abs(ruler_1 - pin[j][2]) / abs(pin[j][0] - pin[j][2]) < ratio and abs(
                    ruler_2 - pin[j][0]) / abs(pin[j][0] - pin[j][2]) < ratio):
                    pin_diameter = bottom_str[i]
                    print("引线下引找到pin_diameter", pin_diameter)
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) < abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):  # 竖向标尺线
            ruler_1 = new_bottom_yolox_pairs_length[i][5]
            ruler_2 = new_bottom_yolox_pairs_length[i][9]
            for j in range(len(pin)):
                if (abs(ruler_1 - pin[j][1]) / abs(pin[j][0] - pin[j][2]) < ratio and abs(ruler_2 - pin[j][3]) / abs(
                        pin[j][0] - pin[j][2]) < ratio) or (
                        abs(ruler_1 - pin[j][3]) / abs(pin[j][0] - pin[j][2]) < ratio and abs(
                    ruler_2 - pin[j][1]) / abs(pin[j][0] - pin[j][2]) < ratio):
                    pin_diameter = bottom_str[i]
                    print("引线下引找到pin_diameter", pin_diameter)
    # 如果引线位置匹配不到数值，那么根据引线长度匹配
    ze = ''
    ratio = 0.18
    if pin_diameter == ze:
        for i in range(len(new_bottom_yolox_pairs_length)):
            if abs(new_bottom_yolox_pairs_length[i][12] - px) / px < ratio or abs(
                    new_bottom_yolox_pairs_length[i][12] - py) / py < ratio:
                pin_diameter = bottom_str[i]
                print("引线之间距离近似pin_diameter，找到pin_diameter参数:", pin_diameter)
    print("---结束用引线方法寻找pin_diameter---")
    return pin_diameter


def yinXinan_find_body(yolox_pairs_top, top_yolox_pairs_length, yolox_pairs_bottom, bottom_yolox_pairs_length, top_border, bottom_border):
    """从上下视图测得封装长宽。"""
    # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]
    # top_yolox_pairs_length,np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
    '''

    print("---开始用引线方法寻找body---")
    # print("top_yolox_pairs_length, bottom_yolox_pairs_length\n", top_yolox_pairs_length, bottom_yolox_pairs_length)
    # 补充：将匹配得到的标注添加到引线中
    min_body = 1.28  # 设置最小的body长
    new_top_yolox_pairs_length = np.zeros((len(top_yolox_pairs_length), 16))
    middle = np.zeros(16)
    for i in range(len(top_yolox_pairs_length)):
        for j in range(len(yolox_pairs_top)):
            if (yolox_pairs_top[j, 0:3] == top_yolox_pairs_length[i, 0:3]).all():
                middle[0:13] = top_yolox_pairs_length[i]
                middle[13:16] = yolox_pairs_top[j, -3:]
                new_top_yolox_pairs_length = np.r_[new_top_yolox_pairs_length, [middle]]
    new_bottom_yolox_pairs_length = np.zeros((len(bottom_yolox_pairs_length), 16))
    middle = np.zeros(16)
    for i in range(len(bottom_yolox_pairs_length)):
        for j in range(len(yolox_pairs_bottom)):
            if (yolox_pairs_bottom[j, 0:3] == bottom_yolox_pairs_length[i, 0:3]).all():
                middle[0:13] = bottom_yolox_pairs_length[i]
                middle[13:16] = yolox_pairs_bottom[j, -3:]
                new_bottom_yolox_pairs_length = np.r_[new_bottom_yolox_pairs_length, [middle]]

    img_path = f'{DATA}/top.jpg'
    top_body = output_body(img_path, name='top')

    if len(top_body) == 0:
        # top_body = begain_output_top_body_location()  # top_body:np(1,4)[x1,y1,x2,y2]
        top_body = top_border
        if len(top_body) == 0:
            top_body = np.zeros((1, 4))
    img_path = f'{DATA}/bottom.jpg'

    bottom_body = output_body(img_path, name='bottom')

    if len(bottom_body) == 0:
        # bottom_body = begain_output_bottom_body_location()  # bottom_body:np(1,4)[x1,y1,x2,y2]
        bottom_body = bottom_border
        if len(bottom_body) == 0:
            bottom_body = np.zeros((1, 4))
    # 引线下引找长宽
    zer = np.zeros(3)
    gao = np.zeros(3)
    kuan = np.zeros(3)
    bottom_gao = np.zeros(3)
    bottom_kuan = np.zeros(3)
    ratio = 0.03
    for i in range(len(new_top_yolox_pairs_length)):

        ruler_1 = 0
        ruler_2 = 0
        if abs(new_top_yolox_pairs_length[i][0] - new_top_yolox_pairs_length[i][2]) > abs(
                new_top_yolox_pairs_length[i][1] - new_top_yolox_pairs_length[i][3]):
            ruler_1 = new_top_yolox_pairs_length[i][4]
            ruler_2 = new_top_yolox_pairs_length[i][8]
            ruler_3 = top_body[0][0]
            ruler_4 = top_body[0][2]
            if new_top_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_top_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
                        new_top_yolox_pairs_length[i][12] < ratio):
                    if (new_top_yolox_pairs_length[i][-3:] > min_body).all():
                        kuan = new_top_yolox_pairs_length[i][-3:]
                        print("引线下引到实体top_body宽，找到top_body_x参数", kuan)
        if abs(new_top_yolox_pairs_length[i][0] - new_top_yolox_pairs_length[i][2]) < abs(
                new_top_yolox_pairs_length[i][1] - new_top_yolox_pairs_length[i][3]):
            ruler_1 = new_top_yolox_pairs_length[i][5]
            ruler_2 = new_top_yolox_pairs_length[i][9]
            ruler_3 = top_body[0][1]
            ruler_4 = top_body[0][3]
            if new_top_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_top_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
                        new_top_yolox_pairs_length[i][12] < ratio):
                    if (new_top_yolox_pairs_length[i][-3:] > min_body).all():
                        gao = new_top_yolox_pairs_length[i][-3:]
                        print("引线下引到实体top_body高，找到top_body_y参数", gao)
    ratio = 0.03
    for i in range(len(new_bottom_yolox_pairs_length)):
        ruler_1 = 0
        ruler_2 = 0
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) > abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):
            ruler_1 = new_bottom_yolox_pairs_length[i][4]
            ruler_2 = new_bottom_yolox_pairs_length[i][8]
            ruler_3 = bottom_body[0][0]
            ruler_4 = bottom_body[0][2]
            if new_bottom_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_bottom_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
                        new_bottom_yolox_pairs_length[i][12] < ratio):
                    if (new_bottom_yolox_pairs_length[i][-3:] > min_body).all():
                        bottom_kuan = new_bottom_yolox_pairs_length[i][-3:]
                        print("引线下引到实体bottom_body宽，找到bottom_body_x参数", bottom_kuan)
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) < abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):
            ruler_1 = new_bottom_yolox_pairs_length[i][5]
            ruler_2 = new_bottom_yolox_pairs_length[i][9]
            ruler_3 = bottom_body[0][1]
            ruler_4 = bottom_body[0][3]
            if new_bottom_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_bottom_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
                        new_bottom_yolox_pairs_length[i][12] < ratio):
                    if (new_bottom_yolox_pairs_length[i][-3:] > min_body).all():
                        bottom_gao = new_bottom_yolox_pairs_length[i][-3:]
                        print("引线下引到实体bottom_body高，找到bottom_body_y参数", bottom_gao)

    # 2.1.计算top中body的长宽
    print("top_body是", top_body)
    if len(top_body) > 0:
        top_body_heng = top_body[0][2] - top_body[0][0]
        top_body_shu = top_body[0][3] - top_body[0][1]

        # 2.在误差允许范围内，pairs对应长宽则认为是长宽
        ratio = 0.1
        for i in range(len(new_top_yolox_pairs_length)):
            if abs(new_top_yolox_pairs_length[i][12] - top_body_heng) / top_body_heng < ratio and (kuan == zer).all():
                if (new_top_yolox_pairs_length[i][-3:] > min_body).all():
                    kuan = new_top_yolox_pairs_length[i, -3:]
                    print("引线之间距离近似kuan，找到kuan参数", kuan)
            if abs(new_top_yolox_pairs_length[i][12] - top_body_shu) / top_body_shu < ratio and (gao == zer).all():
                if (new_top_yolox_pairs_length[i][-3:] > min_body).all():
                    gao = new_top_yolox_pairs_length[i, -3:]
                    print("引线之间距离近似gao，找到gao参数", gao)
    # 3.计算bottom中body的长和宽

    if len(bottom_body) > 0:
        bottom_body_heng = bottom_body[0][2] - bottom_body[0][0]
        bottom_body_shu = bottom_body[0][3] - bottom_body[0][1]

        # 4.在误差允许范围内，pairs对应长宽则认为是长宽
        ratio = 0.1
        for i in range(len(new_bottom_yolox_pairs_length)):
            if abs(new_bottom_yolox_pairs_length[i][12] - bottom_body_heng) / bottom_body_heng < ratio and (
                    bottom_kuan == zer).all():
                if (new_bottom_yolox_pairs_length[i][-3:] > min_body).all():
                    bottom_kuan = new_bottom_yolox_pairs_length[i, -3:]
                    print("引线之间距离近似bottom_kuan，找到bottom_kuan参数", bottom_kuan)
            if abs(new_bottom_yolox_pairs_length[i][12] - bottom_body_shu) / bottom_body_shu < ratio and (
                    bottom_gao == zer).all():
                if (new_bottom_yolox_pairs_length[i][-3:] > min_body).all():
                    bottom_gao = new_bottom_yolox_pairs_length[i, -3:]
                    print("引线之间距离近似bottom_gao，找到bottom_gao参数", bottom_gao)
    # top和bottom互补
    if (kuan == zer).all() and (bottom_kuan != zer).any():
        kuan = bottom_kuan
    if (bottom_kuan == zer).all() and (kuan != zer).any():
        bottom_kuan = kuan
    if (gao == zer).all() and (bottom_gao != zer).any():
        gao = bottom_gao
    if (bottom_gao == zer).all() and (gao != zer).any():
        bottom_gao = gao
    # top的x和y互补
    if (kuan == zer).all() and (gao != zer).any():
        kuan = gao
    if (gao == zer).all() and (kuan != zer).any():
        gao = kuan
    # bottom的x和y互补
    if (bottom_kuan == zer).all() and (bottom_gao != zer).any():
        bottom_kuan = bottom_gao
    if (bottom_gao == zer).all() and (bottom_kuan != zer).any():
        bottom_gao = bottom_kuan
    # bottom和top中找最合适的
    body_x = np.zeros(3)
    body_y = np.zeros(3)
    if (kuan != zer).any() and (bottom_kuan != zer).any():
        if (kuan > bottom_kuan).all():
            body_x = kuan
        if (kuan < bottom_kuan).all():
            body_x = bottom_kuan
    if (gao != zer).any() and (bottom_gao != zer).any():
        if (gao > bottom_gao).all():
            body_x = gao
        if (gao < bottom_gao).all():
            body_x = bottom_gao
    if (body_x == zer).all():
        body_x = kuan
    if (body_y == zer).all():
        body_y = gao
    print("---结束用引线方法寻找body---")
    return body_x, body_y


def yinXinan_find_body_table(yolox_pairs_top, top_yolox_pairs_length, yolox_pairs_bottom, bottom_yolox_pairs_length):
    """在表格模式下估算封装长宽。"""
    # yolox_pairs_top,np.二维数组（，9）[pairs_'x1'_'y1'_'x2'_'y2',标注'x1'_'y1'_'x2'_'y2'，'A1']
    # top_yolox_pairs_length,np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
    '''
    print("---开始用引线方法寻找body---")
    top_str = np.empty(len(top_yolox_pairs_length), dtype=np.dtype('U10'))
    bottom_str = np.empty(len(bottom_yolox_pairs_length), dtype=np.dtype('U10'))
    # top_str， bottom_str记录每个找到引线的标尺线匹配的标注字符串
    new_top_yolox_pairs_length = top_yolox_pairs_length
    new_top_yolox_pairs_length_str = top_yolox_pairs_length.astype(str)
    new_bottom_yolox_pairs_length = bottom_yolox_pairs_length
    new_bottom_yolox_pairs_length_str = bottom_yolox_pairs_length.astype(str)
    for i in range(len(top_yolox_pairs_length)):
        key = 0
        for j in range(len(yolox_pairs_top)):
            if (new_top_yolox_pairs_length_str[i, 0: 4] == yolox_pairs_top[j, 0: 4]).all():
                key = 1
                top_str[i] = yolox_pairs_top[j, 8]
        if key == 0:
            top_str[i] = ''
    for i in range(len(bottom_yolox_pairs_length)):
        key = 0
        for j in range(len(yolox_pairs_bottom)):
            if (new_bottom_yolox_pairs_length_str[i, 0: 4] == yolox_pairs_bottom[j, 0: 4]).all():
                key = 1
                bottom_str[i] = yolox_pairs_bottom[j, 8]
        if key == 0:
            bottom_str[i] = ''

    # from output_top_body_location import begain_output_top_body_location
    # from output_bottom_body_location import begain_output_bottom_body_location
    # from output_body import output_body
    # 1.yolox检测top和bottom的外框

    # top_body = begain_output_top_body_location()#top_body:np(1,4)[x1,y1,x2,y2]
    # while len(top_body) != 1:
    #     top_body = begain_output_top_body_location()  # top_body:np(1,4)[x1,y1,x2,y2]
    #
    # bottom_body = begain_output_bottom_body_location()#bottom_body:np(1,4)[x1,y1,x2,y2]
    # while len(bottom_body) != 1:
    #     bottom_body = begain_output_bottom_body_location()  # bottom_body:np(1,4)[x1,y1,x2,y2]
    img_path = 'data/top.jpg'
    top_body = output_body(img_path, name='top')
    while len(top_body) != 1:
        top_body = begain_output_top_body_location()  # top_body:np(1,4)[x1,y1,x2,y2]
    img_path = 'data/bottom.jpg'
    bottom_body = output_body(img_path, name='bottom')
    while len(bottom_body) != 1:
        bottom_body = begain_output_bottom_body_location()  # bottom_body:np(1,4)[x1,y1,x2,y2]
    # 引线下引找长宽
    zer = ''
    gao = ''
    kuan = ''
    bottom_gao = ''
    bottom_kuan = ''
    ratio = 0.1
    for i in range(len(new_top_yolox_pairs_length)):
        ruler_1 = 0
        ruler_2 = 0
        if abs(new_top_yolox_pairs_length[i][0] - new_top_yolox_pairs_length[i][2]) > abs(
                new_top_yolox_pairs_length[i][1] - new_top_yolox_pairs_length[i][3]):
            ruler_1 = new_top_yolox_pairs_length[i][4]
            ruler_2 = new_top_yolox_pairs_length[i][8]
            ruler_3 = top_body[0][0]
            ruler_4 = top_body[0][2]
            if new_top_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_top_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
                        new_top_yolox_pairs_length[i][12] < ratio):
                    print("引线下引到实体top_body宽，找到top_body_x参数")
                    # for j in range(len(yolox_pairs_top)):
                    #     if (new_top_yolox_pairs_length_str[i, 0: 4] == yolox_pairs_top[j, 0: 4]).all():
                    #         kuan = yolox_pairs_top[j, 8]
                    kuan = top_str[i]
        if abs(new_top_yolox_pairs_length[i][0] - new_top_yolox_pairs_length[i][2]) < abs(
                new_top_yolox_pairs_length[i][1] - new_top_yolox_pairs_length[i][3]):
            ruler_1 = new_top_yolox_pairs_length[i][5]
            ruler_2 = new_top_yolox_pairs_length[i][9]
            ruler_3 = top_body[0][1]
            ruler_4 = top_body[0][3]
            if new_top_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_top_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
                        new_top_yolox_pairs_length[i][12] < ratio):
                    print("引线下引到实体top_body高，找到top_body_y参数")
                    # for j in range(len(yolox_pairs_top)):
                    #     if (new_top_yolox_pairs_length_str[i, 0: 4] == yolox_pairs_top[j, 0: 4]).all():
                    #         gao = yolox_pairs_top[j, 8]
                    gao = top_str[i]
    ratio = 0.1
    for i in range(len(new_bottom_yolox_pairs_length)):
        ruler_1 = 0
        ruler_2 = 0
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) > abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):
            ruler_1 = new_bottom_yolox_pairs_length[i][4]
            ruler_2 = new_bottom_yolox_pairs_length[i][8]
            ruler_3 = bottom_body[0][0]
            ruler_4 = bottom_body[0][2]
            if new_bottom_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_bottom_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
                        new_bottom_yolox_pairs_length[i][12] < ratio):
                    print("引线下引到实体bottom_body宽，找到bottom_body_x参数")
                    # for j in range(len(yolox_pairs_bottom)):
                    #     if (new_bottom_yolox_pairs_length_str[i, 0: 4] == yolox_pairs_bottom[j, 0: 4]).all():
                    #         bottom_kuan = yolox_pairs_bottom[j, 8]
                    bottom_kuan = bottom_str[i]
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) < abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):
            ruler_1 = new_bottom_yolox_pairs_length[i][5]
            ruler_2 = new_bottom_yolox_pairs_length[i][9]
            ruler_3 = bottom_body[0][1]
            ruler_4 = bottom_body[0][3]
            if new_bottom_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_bottom_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
                        new_bottom_yolox_pairs_length[i][12] < ratio):
                    print("引线下引到实体bottom_body高，找到bottom_body_y参数")
                    # for j in range(len(yolox_pairs_bottom)):
                    #     if (new_bottom_yolox_pairs_length_str[i, 0: 4] == yolox_pairs_bottom[j, 0: 4]).all():
                    #         bottom_gao = yolox_pairs_bottom[j, 8]
                    bottom_gao = bottom_str[i]

    # 2.1.计算top中body的长宽

    if len(top_body) > 0:
        top_body_heng = top_body[0][2] - top_body[0][0]
        top_body_shu = top_body[0][3] - top_body[0][1]

        # 2.在误差允许范围内，pairs对应长宽则认为是长宽
        ratio = 0.16
        for i in range(len(new_top_yolox_pairs_length)):
            if abs(new_top_yolox_pairs_length[i][12] - top_body_heng) / top_body_heng < ratio and kuan == zer:
                print("引线之间距离近似kuan，找到kuan参数")
                # for j in range(len(yolox_pairs_top)):
                #     if (new_top_yolox_pairs_length_str[i, 0: 4] == yolox_pairs_top[j, 0: 4]).all():
                #         kuan = yolox_pairs_top[j, 8]
                kuan = top_str[i]
            if abs(new_top_yolox_pairs_length[i][12] - top_body_shu) / top_body_shu < ratio and gao == zer:
                print("引线之间距离近似gao，找到gao参数")
                # for j in range(len(yolox_pairs_top)):
                #     if (new_top_yolox_pairs_length_str[i, 0: 4] == yolox_pairs_top[j, 0: 4]).all():
                #         gao = yolox_pairs_top[j, 8]
                gao = top_str[i]
    # 3.计算bottom中body的长和宽

    if len(bottom_body) > 0:
        bottom_body_heng = bottom_body[0][2] - bottom_body[0][0]
        bottom_body_shu = bottom_body[0][3] - bottom_body[0][1]

        # 4.在误差允许范围内，pairs对应长宽则认为是长宽
        ratio = 0.16
        for i in range(len(new_bottom_yolox_pairs_length)):
            if abs(new_bottom_yolox_pairs_length[i][
                       12] - bottom_body_heng) / bottom_body_heng < ratio and bottom_kuan == zer:
                print("引线之间距离近似bottom_kuan，找到bottom_kuan参数")
                # for j in range(len(yolox_pairs_bottom)):
                #     if (new_bottom_yolox_pairs_length_str[i, 0: 4] == yolox_pairs_bottom[j, 0: 4]).all():
                #         bottom_kuan = yolox_pairs_bottom[j, 8]
                bottom_kuan = bottom_str[i]
            if abs(new_bottom_yolox_pairs_length[i][
                       12] - bottom_body_shu) / bottom_body_shu < ratio and bottom_gao == zer:
                print("引线之间距离近似bottom_gao，找到bottom_gao参数")
                # for j in range(len(yolox_pairs_bottom)):
                #     if (new_bottom_yolox_pairs_length_str[i, 0: 4] == yolox_pairs_bottom[j, 0: 4]).all():
                #         bottom_gao = yolox_pairs_bottom[j, 8]
                bottom_gao = bottom_str[i]
    # top和bottom互补
    if kuan == zer and bottom_kuan != zer:
        kuan = bottom_kuan
    if bottom_kuan == zer and kuan != zer:
        bottom_kuan = kuan
    if gao == zer and bottom_gao != zer:
        gao = bottom_gao
    if bottom_gao == zer and gao != zer:
        bottom_gao = gao
    # top的x和y互补
    if kuan == zer and gao != zer:
        kuan = gao
    if gao == zer and kuan != zer:
        gao = kuan
    # bottom的x和y互补
    if bottom_kuan == zer and bottom_gao != zer:
        bottom_kuan = bottom_gao
    if bottom_gao == zer and bottom_kuan != zer:
        bottom_gao = bottom_kuan
    # bottom和top中找最合适的
    body_x = ''
    body_y = ''
    if kuan != zer and bottom_kuan != zer:
        if len(kuan) > len(bottom_kuan):
            body_x = bottom_kuan
        if len(kuan) < len(bottom_kuan):
            body_x = kuan
    if gao != zer and bottom_gao != zer:
        if len(gao) > len(bottom_gao):
            body_x = bottom_gao
        if len(gao) < len(bottom_gao):
            body_x = gao
    if body_x == zer:
        body_x = kuan
    if body_y == zer:
        body_y = gao
    print("---结束用引线方法寻找body---")
    return body_x, body_y


def yinXian_find_side_high_standoff(yolox_pairs_side, side_yolox_pairs_length):
    """根据侧视标尺寻找高度与 standoff。"""
    # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]
    # top_yolox_pairs_length,np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
    '''
    print("---开始用引线方法寻找high和standoff---")
    high = np.zeros((3))
    standoff = np.zeros((3))
    # 1.yolox检测side的实体外框和支撑的外框，如果检测不止一个，按照外框面积大小从大到小排序

    # from output_side_body_standoff_location import begain_output_side_body_standoff_location
    side_body, standoff_body = begain_output_side_body_standoff_location(img_path=f'{DATA}/side.jpg')
    if len(side_body) > 1:
        a = np.zeros((len(side_body)))
        side_body = np.c_[side_body, a]
        for i in range(len(side_body)):
            side_body[i][4] = abs((side_body[i][0]) - (side_body[i][2])) * abs((side_body[i][1]) - (side_body[i][3]))
        side_body = side_body[np.argsort(side_body[:, 4])]  # 按距离从大到小排序
        side_body = side_body[::-1]
        side_body = side_body[:, 0:4]
    if len(standoff_body) > 1:
        a = np.zeros((len(standoff_body)))
        standoff_body = np.c_[standoff_body, a]
        for i in range(len(standoff_body)):
            standoff_body[i][4] = abs((standoff_body[i][0]) - (standoff_body[i][2])) * abs(
                (standoff_body[i][1]) - (standoff_body[i][3]))
        standoff_body = standoff_body[np.argsort(standoff_body[:, 4])]  # 按距离从大到小排序
        standoff_body = standoff_body[::-1]
        standoff_body = standoff_body[:, 0:4]
    # 2.根据引线的位置判断标尺线是否标记的是side的高
    # 补充：将匹配得到的标注添加到引线中
    new_side_yolox_pairs_length = np.zeros((len(side_yolox_pairs_length), 16))
    middle = np.zeros((16))
    for i in range(len(side_yolox_pairs_length)):
        for j in range(len(yolox_pairs_side)):
            if (yolox_pairs_side[j, 0:3] == side_yolox_pairs_length[i, 0:3]).all():
                middle[0:13] = side_yolox_pairs_length[i]
                middle[13:16] = yolox_pairs_side[j, -3:]
                new_side_yolox_pairs_length = np.r_[new_side_yolox_pairs_length, [middle]]
    ratio = 0.2
    for i in range(len(new_side_yolox_pairs_length)):
        ruler_1 = 0
        ruler_2 = 0
        if abs(new_side_yolox_pairs_length[i][0] - new_side_yolox_pairs_length[i][2]) > abs(
                new_side_yolox_pairs_length[i][1] - new_side_yolox_pairs_length[i][3]):
            ruler_1 = new_side_yolox_pairs_length[i][4]
            ruler_2 = new_side_yolox_pairs_length[i][8]
        else:
            ruler_1 = new_side_yolox_pairs_length[i][5]
            ruler_2 = new_side_yolox_pairs_length[i][9]
        for j in range(len(side_body)):
            ruler_3 = 0
            ruler_4 = 0
            if abs(side_body[j][0] - side_body[j][2]) < abs(side_body[j][1] - side_body[j][3]):
                ruler_3 = side_body[j][0]
                ruler_4 = side_body[j][2]
            else:
                ruler_3 = side_body[j][1]
                ruler_4 = side_body[j][3]
            if new_side_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_side_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_side_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_side_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
                        new_side_yolox_pairs_length[i][12] < ratio):
                    print("引线下引到实体side_body，找到high参数")
                    high = new_side_yolox_pairs_length[i][-3:]
    ratio = 0.2
    for i in range(len(new_side_yolox_pairs_length)):
        ruler_1 = 0
        ruler_2 = 0
        if abs(new_side_yolox_pairs_length[i][0] - new_side_yolox_pairs_length[i][2]) > abs(
                new_side_yolox_pairs_length[i][1] - new_side_yolox_pairs_length[i][3]):
            ruler_1 = new_side_yolox_pairs_length[i][4]
            ruler_2 = new_side_yolox_pairs_length[i][8]
        else:
            ruler_1 = new_side_yolox_pairs_length[i][5]
            ruler_2 = new_side_yolox_pairs_length[i][9]
        for j in range(len(standoff_body)):
            ruler_3 = 0
            ruler_4 = 0
            if abs(standoff_body[j][0] - standoff_body[j][2]) < abs(standoff_body[j][1] - standoff_body[j][3]):
                ruler_3 = standoff_body[j][0]
                ruler_4 = standoff_body[j][2]
            else:
                ruler_3 = standoff_body[j][1]
                ruler_4 = standoff_body[j][3]
            if new_side_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_side_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_side_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_side_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
                        new_side_yolox_pairs_length[i][12] < ratio):
                    if (new_side_yolox_pairs_length[i][-3:] != high).all():
                        print("引线下引到实体standoff，找到standoff参数")
                        standoff = new_side_yolox_pairs_length[i][-3:]
    # 如果引线位置匹配不到数值，那么根据引线长度匹配
    ze = np.zeros(3)
    ratio = 0.18
    if (high == ze).all() and len(side_body) > 0:
        for i in range(len(new_side_yolox_pairs_length)):
            high_strength = min(abs(side_body[0][0] - side_body[0][2]), abs(side_body[0][1] - side_body[0][3]))
            if abs(new_side_yolox_pairs_length[i][12] - high_strength) / high_strength < ratio:
                print("引线之间距离近似high，找到high参数")
                high = new_side_yolox_pairs_length[i][-3:]
    if (standoff == ze).all() and len(standoff_body) > 0:
        for i in range(len(new_side_yolox_pairs_length)):
            standoff_strength = min(abs(standoff_body[0][0] - standoff_body[0][2]),
                                    abs(standoff_body[0][1] - standoff_body[0][3]))
            if abs(new_side_yolox_pairs_length[i][12] - standoff_strength) / standoff_strength < ratio:
                print("引线之间距离近似standoff，找到standoff参数")
                standoff = new_side_yolox_pairs_length[i][-3:]
    print("---结束用引线方法寻找high和standoff---")
    return high, standoff


def yinXian_find_side_high_standoff_table(yolox_pairs_side, side_yolox_pairs_length):
    """在表格模式下寻找高度与 standoff。"""
    # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]
    # top_yolox_pairs_length,np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
    '''
    print("---开始用引线方法寻找high和standoff---")
    high = ''
    standoff = ''
    # 1.yolox检测side的实体外框和支撑的外框，如果检测不止一个，按照外框面积大小从大到小排序
    # from output_side_body_standoff_location import begain_output_side_body_standoff_location
    side_body, standoff_body = begain_output_side_body_standoff_location(img_path='data/side.jpg')
    if len(side_body) > 1:
        a = np.zeros((len(side_body)))
        side_body = np.c_[side_body, a]
        for i in range(len(side_body)):
            side_body[i][4] = abs((side_body[i][0]) - (side_body[i][2])) * abs((side_body[i][1]) - (side_body[i][3]))
        side_body = side_body[np.argsort(side_body[:, 4])]  # 按距离从大到小排序
        side_body = side_body[::-1]
        side_body = side_body[:, 0:4]
    if len(standoff_body) > 1:
        a = np.zeros((len(standoff_body)))
        standoff_body = np.c_[standoff_body, a]
        for i in range(len(standoff_body)):
            standoff_body[i][4] = abs((standoff_body[i][0]) - (standoff_body[i][2])) * abs(
                (standoff_body[i][1]) - (standoff_body[i][3]))
        standoff_body = standoff_body[np.argsort(standoff_body[:, 4])]  # 按距离从大到小排序
        standoff_body = standoff_body[::-1]
        standoff_body = standoff_body[:, 0:4]
    # 2.根据引线的位置判断标尺线是否标记的是side的高

    side_str = np.empty(len(side_yolox_pairs_length), dtype=np.dtype('U10'))
    # top_str， bottom_str记录每个找到引线的标尺线匹配的标注字符串
    new_side_yolox_pairs_length = side_yolox_pairs_length
    new_side_yolox_pairs_length_str = side_yolox_pairs_length.astype(str)

    for i in range(len(side_yolox_pairs_length)):
        key = 0
        for j in range(len(yolox_pairs_side)):
            if (new_side_yolox_pairs_length_str[i, 0: 4] == yolox_pairs_side[j, 0: 4]).all():
                key = 1
                side_str[i] = yolox_pairs_side[j, 8]
        if key == 0:
            side_str[i] = ''
    ratio = 0.2
    for i in range(len(new_side_yolox_pairs_length)):
        ruler_1 = 0
        ruler_2 = 0
        if abs(new_side_yolox_pairs_length[i][0] - new_side_yolox_pairs_length[i][2]) > abs(
                new_side_yolox_pairs_length[i][1] - new_side_yolox_pairs_length[i][3]):
            ruler_1 = new_side_yolox_pairs_length[i][4]
            ruler_2 = new_side_yolox_pairs_length[i][8]
        else:
            ruler_1 = new_side_yolox_pairs_length[i][5]
            ruler_2 = new_side_yolox_pairs_length[i][9]
        for j in range(len(side_body)):
            ruler_3 = 0
            ruler_4 = 0
            if abs(side_body[j][0] - side_body[j][2]) < abs(side_body[j][1] - side_body[j][3]):
                ruler_3 = side_body[j][0]
                ruler_4 = side_body[j][2]
            else:
                ruler_3 = side_body[j][1]
                ruler_4 = side_body[j][3]
            if new_side_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_side_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_side_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_side_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
                        new_side_yolox_pairs_length[i][12] < ratio):
                    print("引线下引到实体side_body，找到high参数")
                    high = side_str[i]
    ratio = 0.2
    for i in range(len(new_side_yolox_pairs_length)):
        ruler_1 = 0
        ruler_2 = 0
        if abs(new_side_yolox_pairs_length[i][0] - new_side_yolox_pairs_length[i][2]) > abs(
                new_side_yolox_pairs_length[i][1] - new_side_yolox_pairs_length[i][3]):
            ruler_1 = new_side_yolox_pairs_length[i][4]
            ruler_2 = new_side_yolox_pairs_length[i][8]
        else:
            ruler_1 = new_side_yolox_pairs_length[i][5]
            ruler_2 = new_side_yolox_pairs_length[i][9]
        for j in range(len(standoff_body)):
            ruler_3 = 0
            ruler_4 = 0
            if abs(standoff_body[j][0] - standoff_body[j][2]) < abs(standoff_body[j][1] - standoff_body[j][3]):
                ruler_3 = standoff_body[j][0]
                ruler_4 = standoff_body[j][2]
            else:
                ruler_3 = standoff_body[j][1]
                ruler_4 = standoff_body[j][3]
            if new_side_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_side_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_side_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_side_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_3) /
                        new_side_yolox_pairs_length[i][12] < ratio):
                    print("引线下引到实体standoff，找到standoff参数")
                    standoff = side_str[i]
    # 如果引线位置匹配不到数值，那么根据引线长度匹配
    ze = ''
    ratio = 0.18
    if high == ze:
        for i in range(len(new_side_yolox_pairs_length)):
            high_strength = min(abs(side_body[0][0] - side_body[0][2]), abs(side_body[0][1] - side_body[0][3]))
            if abs(new_side_yolox_pairs_length[i][12] - high_strength) / high_strength < ratio:
                print("引线之间距离近似high，找到high参数")
                high = side_str[i]
    if standoff == ze:
        for i in range(len(new_side_yolox_pairs_length)):
            standoff_strength = min(abs(standoff_body[0][0] - standoff_body[0][2]),
                                    abs(standoff_body[0][1] - standoff_body[0][3]))
            if abs(new_side_yolox_pairs_length[i][12] - standoff_strength) / standoff_strength < ratio:
                print("引线之间距离近似standoff，找到standoff参数")
                standoff = side_str[i]
    print("---结束用引线方法寻找high和standoff---")
    return high, standoff


def ocr_en_cn(img_path, location):
    """调用 OCR 模型识别中英文混合文本。"""
    location:np.(,4)[x1,y1,x2,y2]
    '''
    show_img_key = 0  # 是否显示过程中ocr待检测图片 0 = 不显示，1 = 显示
    data = np.array([[0, 0, 0, 0, '0']])
    # 加载ocr模型
    ocr = PaddleOCR(use_angle_cls=True,
                    lang="en",
                    # det_model_dir="ppocr_model/det/en/en_PP-OCRv3_det_infer",
                    det_model_dir="ppocr_model/det/en/en_PP-OCRv3_sever_det_infer",

                    # rec_model_dir='ppocr_model/rec/en/en_PP-OCRv3_rec_infer',
                    rec_model_dir='ppocr_model/rec/en/en_PP-OCRv3_sever_rec_infer',
                    cls_model_dir='ppocr_model/cls/ch_ppocr_mobile_v2.0_cls_infer',
                    use_gpu=False)  # 导入模型， 禁用gpu
    # 打开图片
    with open(img_path, 'rb') as f:
        np_arr = np.frombuffer(f.read(), dtype=np.uint8)
        # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    #以彩图读取
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取
    no = 1  # ocr识别图片的序号
    # 针对location中的location在图片中的坐标，裁剪出图片区域用来给ocr检测
    for i in range(len(location)):
        # 裁剪识别区域的时候需要扩展一圈，以防yolox极限检测框导致某些数据边缘没有被检测
        # 只可能横着的裁剪成竖着的
        # 方案：横着的一定图片方向正确，扩展一圈识别，如果没有识别到，等比例扩大再识别；竖着的图片看宽长比是否小于0.7，小于则顺时针旋转90，如果识别不到，先认为是文本方向错误，逆时针90转回来识别。如果还识别不到，则等比例放大重复上述
        box = np.array([[location[i][0], location[i][1]], [location[i][2], location[i][1]],
                        [location[i][2], location[i][3]], [location[i][0], location[i][3]]],
                       np.float32)
        box_img = get_rotate_crop_image(img, box)  # yolox检测的原始location区域
        height, width = box_img.shape[0], box_img.shape[1]
        # ****************
        box_img = cv2.resize(box_img, (320, int(height * 320 / width)))  # 等比例放缩成320，适合ocr识别
        # ****************
        if show_img_key == 1:
            cv2.imshow('no_correct_img', box_img)  # 显示当前ocr的识别区域
            cv2.waitKey(0)
        if location[i][2] - location[i][0] > location[i][3] - location[i][1]:
            KuoZhan_ratio = 0.25  # 扩展的比例
            KuoZhan_x = KuoZhan_ratio * abs(location[i][2] - location[i][0]) * (0.5)
            KuoZhan_y = KuoZhan_ratio * abs(location[i][3] - location[i][1]) * (0.5)
            box = np.array([[location[i][0] - KuoZhan_x, location[i][1] - KuoZhan_y],
                            [location[i][2] + KuoZhan_x, location[i][1] - KuoZhan_y],
                            [location[i][2] + KuoZhan_x, location[i][3] + KuoZhan_y],
                            [location[i][0] - KuoZhan_x, location[i][3] + KuoZhan_y]], np.float32)

            box_img = get_rotate_crop_image(img, box)

        if location[i][2] - location[i][0] < location[i][3] - location[i][
            1]:  # 竖着的data需要旋转90,如果识别不到那就是本来文本是横着结果截取范围是竖着的
            KuoZhan_ratio = 0.25
            KuoZhan_x = KuoZhan_ratio * abs(location[i][2] - location[i][0]) * (0.5)
            KuoZhan_y = KuoZhan_ratio * abs(location[i][3] - location[i][1]) * (0.5)
            box = np.array([[location[i][0] - KuoZhan_x, location[i][1] - KuoZhan_y],
                            [location[i][2] + KuoZhan_x, location[i][1] - KuoZhan_y],
                            [location[i][2] + KuoZhan_x, location[i][3] + KuoZhan_y],
                            [location[i][0] - KuoZhan_x, location[i][3] + KuoZhan_y]], np.float32)
            box_img = get_rotate_crop_image(img, box)
            # print("(location[i][10] - location[i][8])/(location[i][11]-location[i][9])",
            #       (location[i][10] - location[i][8]) / (location[i][11] - location[i][9]))
            rotate_key = 0
            length_to_weight = 0.7  # 长宽比 小于1
            if (location[i][2] - location[i][0]) / (
                    location[i][3] - location[i][1]) < length_to_weight:
                rotate_key = 1
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90

        height, width = box_img.shape[0], box_img.shape[1]
        # ****************
        box_img = cv2.resize(box_img, (320, int(height * 320 / width)))  # 等比例放缩成320，适合ocr识别

        print("正在OCR识别第", no, "个pairs匹配的data")
        no += 1
        if show_img_key == 1:
            cv2.imshow('origin_img', box_img)  # 显示当前ocr的识别区域
            cv2.waitKey(0)

        result = ocr.ocr(box_img, cls=True, det=True, rec=True)  # 问题：有可能yolox检测不到data，需要在yolox检测方面解决

        if (result == [None] or result == [[]]) and location[i][2] - location[i][0] < location[i][3] - location[i][
            1] and rotate_key == 0:
            box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
            rotate_key = 1
            if show_img_key == 1:
                cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                cv2.waitKey(0)
            result = ocr.ocr(box_img, cls=True, det=True, rec=True)

        if (result == [None] or result == [[]]) and location[i][2] - location[i][0] < location[i][3] - location[i][
            1] and rotate_key == 1:
            box_img = cv2.rotate(box_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90
            if show_img_key == 1:
                cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                cv2.waitKey(0)
            result = ocr.ocr(box_img, cls=True, det=True, rec=True)

        if result == [None] or result == [[]]:  # 如果识别不到，那么把图片”等比例“放大，不等比例会导致OCR识别失败
            if location[i][2] - location[i][0] > location[i][3] - location[i][1]:
                KuoZhan_ratio = 0.25
                KuoZhan_x = KuoZhan_ratio * abs(location[i][2] - location[i][0]) * (0.5)
                KuoZhan_y = KuoZhan_ratio * abs(location[i][3] - location[i][1]) * (0.5)
                box = np.array([[location[i][0] - KuoZhan_x, location[i][1] - KuoZhan_y],
                                [location[i][2] + KuoZhan_x, location[i][1] - KuoZhan_y],
                                [location[i][2] + KuoZhan_x, location[i][3] + KuoZhan_y],
                                [location[i][0] - KuoZhan_x, location[i][3] + KuoZhan_y]], np.float32)

                box_img = get_rotate_crop_image(img, box)

            if location[i][2] - location[i][0] < location[i][3] - location[i][
                1]:  # 竖着的data需要旋转90,如果识别不到那就是本来文本是横着结果截取范围是竖着的
                KuoZhan_ratio = 0.25
                KuoZhan_x = KuoZhan_ratio * abs(location[i][2] - location[i][0]) * (0.5)
                KuoZhan_y = KuoZhan_ratio * abs(location[i][3] - location[i][1]) * (0.5)
                box = np.array([[location[i][0] - KuoZhan_x, location[i][1] - KuoZhan_y],
                                [location[i][2] + KuoZhan_x, location[i][1] - KuoZhan_y],
                                [location[i][2] + KuoZhan_x, location[i][3] + KuoZhan_y],
                                [location[i][0] - KuoZhan_x, location[i][3] + KuoZhan_y]], np.float32)
                box_img = get_rotate_crop_image(img, box)
                # print("(location[i][10] - location[i][8])/(location[i][11]-location[i][9])",
                #       (location[i][10] - location[i][8]) / (location[i][11] - location[i][9]))
                rotate_key = 0
                if (location[i][2] - location[i][0]) / (location[i][3] - location[i][1]) < 0.75:
                    rotate_key = 1
                    box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
            # box_img = cv2.resize(box_img, (160, 80),cv2.INTER_AREA)
            box_img = img_resize(box_img)  # 等比例放大
            box_img = hist(box_img, show_img_key)  # 图像增强

            if show_img_key == 1:
                cv2.imshow('enhance_origin_img', box_img)  # 显示当前ocr的识别区域
                cv2.waitKey(0)

            result = ocr.ocr(box_img, cls=True, det=True, rec=True)  # 问题：有可能yolox检测不到data，需要在yolox检测方面解决

            if (result == [None] or result == [[]]) and location[i][2] - location[i][0] < location[i][3] - \
                    location[i][1] and rotate_key == 0:
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
                rotate_key = 1
                if show_img_key == 1:
                    cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                    cv2.waitKey(0)
                result = ocr.ocr(box_img, cls=True, det=True, rec=True)

            if (result == [None] or result == [[]]) and location[i][2] - location[i][0] < location[i][3] - \
                    location[i][1] and rotate_key == 1:
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90
                if show_img_key == 1:
                    cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                    cv2.waitKey(0)
                result = ocr.ocr(box_img, cls=True, det=True, rec=True)

        print("OCR识别data的result\n", result)

        data_mid = [location[i][0], location[i][1], location[i][2], location[i][3], ]
        for j in range(len(result[0])):
            if str(result[0][j][1][0]) != '-':
                data_mid = np.append(data_mid, str(result[0][j][1][0]))
            print(result[0][j][1][0])
        if len(data_mid) > 4 and data_mid[4] != '-':
            data = np.r_[data, [data_mid[0:5]]]
    data = data[1:, :]
    return data


def ocr_en_cn_onnx(img_path, location):
    """使用 ONNX OCR 模型识别中英文文本。"""
    location:np.(,4)[x1,y1,x2,y2]
    '''

    # from ocr_onnx.onnx_use import Run_onnx
    show_img_key = 0  # 是否显示过程中ocr待检测图片 0 = 不显示，1 = 显示

    data = np.array([[0, 0, 0, 0, '0']])

    # 打开图片
    with open(img_path, 'rb') as f:
        np_arr = np.frombuffer(f.read(), dtype=np.uint8)
        # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    #以彩图读取
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取
    no = 1  # ocr识别图片的序号
    # 针对location中的location在图片中的坐标，裁剪出图片区域用来给ocr检测
    for i in range(len(location)):
        # 裁剪识别区域的时候需要扩展一圈，以防yolox极限检测框导致某些数据边缘没有被检测
        # 只可能横着的裁剪成竖着的
        # 方案：横着的一定图片方向正确，扩展一圈识别，如果没有识别到，等比例扩大再识别；竖着的图片看宽长比是否小于0.7，小于则顺时针旋转90，如果识别不到，先认为是文本方向错误，逆时针90转回来识别。如果还识别不到，则等比例放大重复上述
        box = np.array([[location[i][0], location[i][1]], [location[i][2], location[i][1]],
                        [location[i][2], location[i][3]], [location[i][0], location[i][3]]],
                       np.float32)
        box_img = get_rotate_crop_image(img, box)  # yolox检测的原始location区域
        height, width = box_img.shape[0], box_img.shape[1]
        # ****************
        box_img = cv2.resize(box_img, (320, int(height * 320 / width)))  # 等比例放缩成320，适合ocr识别
        # ****************
        if show_img_key == 1:
            cv2.imshow('no_correct_img', box_img)  # 显示当前ocr的识别区域
            cv2.waitKey(0)
        if location[i][2] - location[i][0] > location[i][3] - location[i][1]:
            KuoZhan_ratio = 0  # 扩展的比例
            KuoZhan_x = KuoZhan_ratio * abs(location[i][2] - location[i][0]) * (0.5)
            KuoZhan_y = KuoZhan_ratio * abs(location[i][3] - location[i][1]) * (0.5)
            box = np.array([[location[i][0] - KuoZhan_x, location[i][1] - KuoZhan_y],
                            [location[i][2] + KuoZhan_x, location[i][1] - KuoZhan_y],
                            [location[i][2] + KuoZhan_x, location[i][3] + KuoZhan_y],
                            [location[i][0] - KuoZhan_x, location[i][3] + KuoZhan_y]], np.float32)

            box_img = get_rotate_crop_image(img, box)

        if location[i][2] - location[i][0] < location[i][3] - location[i][
            1]:  # 竖着的data需要旋转90,如果识别不到那就是本来文本是横着结果截取范围是竖着的
            KuoZhan_ratio = 0
            KuoZhan_x = KuoZhan_ratio * abs(location[i][2] - location[i][0]) * (0.5)
            KuoZhan_y = KuoZhan_ratio * abs(location[i][3] - location[i][1]) * (0.5)
            box = np.array([[location[i][0] - KuoZhan_x, location[i][1] - KuoZhan_y],
                            [location[i][2] + KuoZhan_x, location[i][1] - KuoZhan_y],
                            [location[i][2] + KuoZhan_x, location[i][3] + KuoZhan_y],
                            [location[i][0] - KuoZhan_x, location[i][3] + KuoZhan_y]], np.float32)
            box_img = get_rotate_crop_image(img, box)
            # print("(location[i][10] - location[i][8])/(location[i][11]-location[i][9])",
            #       (location[i][10] - location[i][8]) / (location[i][11] - location[i][9]))
            rotate_key = 0
            length_to_weight = 0.7  # 长宽比 小于1
            if (location[i][2] - location[i][0]) / (
                    location[i][3] - location[i][1]) < length_to_weight:
                rotate_key = 1
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90

        height, width = box_img.shape[0], box_img.shape[1]
        # ****************
        box_img = cv2.resize(box_img, (320, int(height * 320 / width)))  # 等比例放缩成320，适合ocr识别

        print("正在OCR识别第", no, "个pairs匹配的data")
        no += 1
        if show_img_key == 1:
            cv2.imshow('origin_img', box_img)  # 显示当前ocr的识别区域
            cv2.waitKey(0)

        img_p = "ocr_onnx/img/img.jpg"
        cv2.imwrite("ocr_onnx/img/img.jpg", box_img)
        result = Run_onnx(img_p)

        if (result == [None] or result == [[]]) and location[i][2] - location[i][0] < location[i][3] - location[i][
            1] and rotate_key == 0:
            box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
            rotate_key = 1
            if show_img_key == 1:
                cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                cv2.waitKey(0)
            img_p = "ocr_onnx/img/img.jpg"
            cv2.imwrite("ocr_onnx/img/img.jpg", box_img)
            result = Run_onnx(img_p)

        if (result == [None] or result == [[]]) and location[i][2] - location[i][0] < location[i][3] - location[i][
            1] and rotate_key == 1:
            box_img = cv2.rotate(box_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90
            if show_img_key == 1:
                cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                cv2.waitKey(0)
            img_p = "ocr_onnx/img/img.jpg"
            cv2.imwrite("ocr_onnx/img/img.jpg", box_img)
            result = Run_onnx(img_p)

        if result == [None] or result == [[]]:  # 如果识别不到，那么把图片”等比例“放大，不等比例会导致OCR识别失败
            if location[i][2] - location[i][0] > location[i][3] - location[i][1]:
                KuoZhan_ratio = 0
                KuoZhan_x = KuoZhan_ratio * abs(location[i][2] - location[i][0]) * (0.5)
                KuoZhan_y = KuoZhan_ratio * abs(location[i][3] - location[i][1]) * (0.5)
                box = np.array([[location[i][0] - KuoZhan_x, location[i][1] - KuoZhan_y],
                                [location[i][2] + KuoZhan_x, location[i][1] - KuoZhan_y],
                                [location[i][2] + KuoZhan_x, location[i][3] + KuoZhan_y],
                                [location[i][0] - KuoZhan_x, location[i][3] + KuoZhan_y]], np.float32)

                box_img = get_rotate_crop_image(img, box)

            if location[i][2] - location[i][0] < location[i][3] - location[i][
                1]:  # 竖着的data需要旋转90,如果识别不到那就是本来文本是横着结果截取范围是竖着的
                KuoZhan_ratio = 0
                KuoZhan_x = KuoZhan_ratio * abs(location[i][2] - location[i][0]) * (0.5)
                KuoZhan_y = KuoZhan_ratio * abs(location[i][3] - location[i][1]) * (0.5)
                box = np.array([[location[i][0] - KuoZhan_x, location[i][1] - KuoZhan_y],
                                [location[i][2] + KuoZhan_x, location[i][1] - KuoZhan_y],
                                [location[i][2] + KuoZhan_x, location[i][3] + KuoZhan_y],
                                [location[i][0] - KuoZhan_x, location[i][3] + KuoZhan_y]], np.float32)
                box_img = get_rotate_crop_image(img, box)
                # print("(location[i][10] - location[i][8])/(location[i][11]-location[i][9])",
                #       (location[i][10] - location[i][8]) / (location[i][11] - location[i][9]))
                rotate_key = 0
                if (location[i][2] - location[i][0]) / (location[i][3] - location[i][1]) < 0.75:
                    rotate_key = 1
                    box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
            # box_img = cv2.resize(box_img, (160, 80),cv2.INTER_AREA)
            box_img = img_resize(box_img)  # 等比例放大
            box_img = hist(box_img, show_img_key)  # 图像增强

            if show_img_key == 1:
                cv2.imshow('enhance_origin_img', box_img)  # 显示当前ocr的识别区域
                cv2.waitKey(0)

            img_p = "ocr_onnx/img/img.jpg"
            cv2.imwrite("ocr_onnx/img/img.jpg", box_img)
            result = Run_onnx(img_p)

            if (result == [None] or result == [[]]) and location[i][2] - location[i][0] < location[i][3] - \
                    location[i][1] and rotate_key == 0:
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90
                rotate_key = 1
                if show_img_key == 1:
                    cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                    cv2.waitKey(0)
                img_p = "ocr_onnx/img/img.jpg"
                cv2.imwrite("ocr_onnx/img/img.jpg", box_img)
                result = Run_onnx(img_p)

            if (result == [None] or result == [[]]) and location[i][2] - location[i][0] < location[i][3] - \
                    location[i][1] and rotate_key == 1:
                box_img = cv2.rotate(box_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90
                if show_img_key == 1:
                    cv2.imshow('clockwise 90 img', box_img)  # 显示当前ocr的识别区域
                    cv2.waitKey(0)
                img_p = "ocr_onnx/img/img.jpg"
                cv2.imwrite("ocr_onnx/img/img.jpg", box_img)
                result = Run_onnx(img_p)

        print("OCR识别data的result\n", result)

        data_mid = [location[i][0], location[i][1], location[i][2], location[i][3], ]
        for j in range(len(result)):
            if str(result[j]) != '-':
                data_mid = np.append(data_mid, str(result[j]))
            print(result[j])
        if len(data_mid) > 4 and data_mid[4] != '-':
            data = np.r_[data, [data_mid[0:5]]]
    data = data[1:, :]
    print(data)
    return data


def correct_serial_letters_data(serial_letters_data):
    """整理序号字母的识别结果。"""
    for i in range(len(serial_letters_data)):
        j = -1
        for every_letter in serial_letters_data[i][4]:
            j += 1
            if every_letter == '8':
                every_letter = 'B'
                strings = list(serial_letters_data[i][4])
                strings[j] = 'B'
                serial_letters_data[i][4] = ''.join(strings)
    return serial_letters_data


def find_pin_num_pin_1(serial_numbers_data, serial_letters_data, serial_numbers, serial_letters):
    """综合序号信息确定行列数及 Pin1 位置。"""
    serial_numbers_data:np.(,4)['x1','y1','x2','y2','str']
    serial_letters_data:np.(,4)['x1','y1','x2','y2','str']
    serial_numbers:np.(,4)[x1,y1,x2,y2)
    serial_letters:np.(,4)[x1,y1,x2,y2)
    '''
    # 默认输出
    pin_num_x_serial = 0
    pin_num_y_serial = 0
    pin_num_serial_number = 0
    pin_num_serial_letter = 0
    pin_1_location = np.array([-1, -1])
    # pin_1_location = [X, Y],X = 0:横向用数字标记序号，纵向用字母标记序号；X= 1，横向用字母标记序号，纵向用数字标记序号
    # pin_1_location = [X, Y],Y = 0 = 左上角,1= 右上角，2 = 右下角，3 = 左下角
    if len(serial_numbers_data) > 0 or len(serial_numbers_data) > 0:

        # ocr识别serial_number,serial_letter
        # img_path = 'data/bottom.jpg'
        # serial_numbers_data = ocr_en_cn_onnx(img_path, serial_numbers_data)
        # print('serial_numbers_data', serial_numbers_data)
        # serial_letters_data = ocr_en_cn_onnx(img_path, serial_letters_data)
        # print('serial_letters_data', serial_letters_data)
        # 根据经验修改ocr识别的错误
        serial_letters_data = correct_serial_letters_data(serial_letters_data)
        print('修正之后的serial_letters_data', serial_letters_data)
        # 根据serial_number最大值找行列数
        serial_number = np.zeros((0))
        new_serial_numbers_data = np.array([['0', '0', '0', '0', '0']])
        for i in range(len(serial_numbers_data)):
            try:
                serial_number = np.append(serial_number, int(serial_numbers_data[i][4]))
                new_serial_numbers_data = np.r_[new_serial_numbers_data, [serial_numbers_data[i]]]
            except:
                print("在用数字标识的pin行列序号中ocr识别到非数字信息，删除")

        serial_numbers_data = new_serial_numbers_data
        print('修正之后的serial_numbers_data', serial_numbers_data)
        serial_number = -(np.sort(-serial_number))  # 从大到小排列
        pin_num_serial_number = 0
        for i in range(len(serial_number)):
            if len(serial_number) > 1 and i + 1 < len(serial_number):
                if serial_number[i] - serial_number[i + 1] < 3:
                    pin_num_serial_number = serial_number[i]
                    break

        # 根据serial_letter最大值找行列数
        letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'U', 'V', 'W',
                       'Y']
        letter_list_a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 't', 'u', 'v', 'w',
                         'y']
        serial_letter = np.zeros((0))
        # 将字母序列转为数字序列
        for i in range(len(serial_letters_data)):
            letter_number = 0
            no = 0
            letters = serial_letters_data[i][4]
            letters = letters[::-1]  # 倒序
            for every_letter in letters:
                no += 1
                for j in range(len(letter_list)):
                    if letter_list[j] == every_letter or letter_list_a[j] == every_letter:
                        letter_number += 20 ** (no - 1) * (j + 1)
            serial_letter = np.append(serial_letter, letter_number)
            serial_letters_data[i][4] = str(letter_number)
        print("serial_letters_data", serial_letters_data)
        serial_letter = -(np.sort(-serial_letter))  # 从大到小排列
        pin_num_serial_letter = 0
        for i in range(len(serial_letter)):
            if len(serial_letter) > 1 and i + 1 < len(serial_letter):
                if serial_letter[i] - serial_letter[i + 1] < 3:
                    pin_num_serial_letter = serial_letter[i]
                    break
        print('pin_num_serial_number, pin_num_serial_letter', pin_num_serial_number, pin_num_serial_letter)
    if pin_num_serial_number != 0:
        if len(serial_numbers) > 0:
            if abs(serial_numbers[0][0] - serial_numbers[0][2]) > abs(serial_numbers[0][1] - serial_numbers[0][3]):
                pin_num_x_serial = pin_num_serial_number
            else:
                pin_num_y_serial = pin_num_serial_number
    if pin_num_serial_letter != 0:
        if len(serial_letters) > 0:
            if abs(serial_letters[0][0] - serial_letters[0][2]) > abs(serial_letters[0][1] - serial_letters[0][3]):
                pin_num_x_serial = pin_num_serial_letter
            else:
                pin_num_y_serial = pin_num_serial_letter
    print("pin_num_x_serial, pin_num_y_serial", pin_num_x_serial, pin_num_y_serial)
    # pin_1_location = [X, Y],X = 0:横向用数字标记序号，纵向用字母标记序号；X= 1，横向用字母标记序号，纵向用数字标记序号
    # pin_1_location = [X, Y],Y = 0 = 左上角,1= 右上角，2 = 右下角，3 = 左下角
    # pin1定位
    if len(serial_numbers_data) > 0 or len(serial_numbers_data) > 0:
        if len(serial_numbers) > 0:
            if abs(serial_numbers[0][0] - serial_numbers[0][2]) > abs(serial_numbers[0][1] - serial_numbers[0][3]):
                pin_1_location[0] = 0
            else:
                pin_1_location[0] = 1
        if len(serial_letters) > 0:
            if abs(serial_letters[0][0] - serial_letters[0][2]) > abs(serial_letters[0][1] - serial_letters[0][3]):
                pin_1_location[0] = 1
            else:
                pin_1_location[0] = 0

        heng_begain = -1
        shu_begain = -1
        serial_numbers_data = serial_numbers_data.astype(np.float32)
        serial_numbers_data = serial_numbers_data.astype(np.int32)
        # 删除0
        new_serial_numbers_data = np.zeros((0, 5))
        for i in range(len(serial_numbers_data)):
            if serial_numbers_data[i][4] != 0:
                new_serial_numbers_data = np.r_[new_serial_numbers_data, [serial_numbers_data[i]]]
        serial_numbers_data = new_serial_numbers_data

        if len(serial_numbers_data) > 0:
            if abs(serial_numbers[0][0] - serial_numbers[0][2]) > abs(serial_numbers[0][1] - serial_numbers[0][3]):
                if len(serial_numbers_data) >= 2:
                    if len(serial_numbers_data[0]) == 5:
                        serial_numbers_data = serial_numbers_data[np.argsort(serial_numbers_data[:, 4])]  # 按序号从小到大排序
                        if serial_numbers_data[0, 0] < serial_numbers_data[(len(serial_numbers_data) - 1), 0]:
                            heng_begain = 0
                        else:
                            heng_begain = 1
            if abs(serial_numbers[0][0] - serial_numbers[0][2]) < abs(serial_numbers[0][1] - serial_numbers[0][3]):
                if len(serial_numbers_data) >= 2:
                    if len(serial_numbers_data[0]) == 5:
                        serial_numbers_data = serial_numbers_data[np.argsort(serial_numbers_data[:, 4])]  # 按序号从小到大排序
                        if serial_numbers_data[0, 1] < serial_numbers_data[(len(serial_numbers_data) - 1), 1]:
                            shu_begain = 0
                        else:
                            shu_begain = 1
        if len(serial_letters_data) > 0:
            serial_letters_data = serial_letters_data.astype(np.float32)
            serial_letters_data = serial_letters_data.astype(np.int32)
            # 删除0
            new_serial_letters_data = np.zeros((0, 5))
            for i in range(len(serial_letters_data)):
                if serial_letters_data[i][4] != 0:
                    new_serial_letters_data = np.r_[new_serial_letters_data, [serial_letters_data[i]]]
            serial_letters_data = new_serial_letters_data
            if abs(serial_letters[0][0] - serial_letters[0][2]) > abs(serial_letters[0][1] - serial_letters[0][3]):
                if len(serial_letters_data) >= 2:
                    if len(serial_letters_data[0]) == 5:
                        serial_letters_data = serial_letters_data[np.argsort(serial_letters_data[:, 4])]  # 按序号从小到大排序
                        if serial_letters_data[0, 0] < serial_letters_data[(len(serial_letters_data) - 1), 0]:
                            heng_begain = 0
                        else:
                            heng_begain = 1
            if abs(serial_letters[0][0] - serial_letters[0][2]) < abs(serial_letters[0][1] - serial_letters[0][3]):
                if len(serial_letters_data) >= 2:
                    if len(serial_letters_data[0]) == 5:
                        serial_letters_data = serial_letters_data[np.argsort(serial_letters_data[:, 4])]  # 按序号从小到大排序
                        if serial_letters_data[0, 1] < serial_letters_data[(len(serial_letters_data) - 1), 1]:
                            shu_begain = 0
                        else:
                            shu_begain = 1
        if heng_begain == 0 and shu_begain == 0:
            pin_1_location[1] = 0
        if heng_begain == 1 and shu_begain == 0:
            pin_1_location[1] = 1
        if heng_begain == 1 and shu_begain == 1:
            pin_1_location[1] = 2
        if heng_begain == 0 and shu_begain == 1:
            pin_1_location[1] = 3
        if heng_begain == 0 and shu_begain == -1:
            pin_1_location[1] = 0
        if heng_begain == 1 and shu_begain == -1:
            pin_1_location[1] = 1
        if heng_begain == -1 and shu_begain == 0:
            pin_1_location[1] = 0
        if heng_begain == -1 and shu_begain == 1:
            pin_1_location[1] = 3

    return pin_num_x_serial, pin_num_y_serial, pin_1_location


def get_absolute_high(high, side_ocr_data):
    """从 OCR 文本中解析绝对高度。"""
    当只找到一个max则判断绝对是high
    当找到多个max哪个最大哪个就是high
    '''
    high_max = np.zeros((0, 3))
    for i in range(len(side_ocr_data)):
        if side_ocr_data[i]['Absolutely'] == 'high':
            try:
                high_max = np.r_[high_max, [side_ocr_data[i]['max_medium_min']]]
            except:
                pass
    if len(high_max) > 0:
        high_max = high_max[np.argsort(-high_max[:, 0])]
        high = high_max[0]
        print("找到绝对正确的high:", high)
    return high


def get_absolute_pin_num(pin_x_num, pin_y_num, bottom_ocr_data):
    """从 OCR 文本中解析绝对的引脚数量。"""

    '''
    for i in range(len(bottom_ocr_data)):
        if bottom_ocr_data[i]['Absolutely'] == 'pin_num_x':
            try:
                pin_x_num = bottom_ocr_data[i]['max_medium_min'][0]
                print("找到绝对正确的pin_num_x:", pin_x_num)
            except:
                pass
        if bottom_ocr_data[i]['Absolutely'] == 'pin_num_y':
            try:
                pin_y_num = bottom_ocr_data[i]['max_medium_min'][0]
                print("找到绝对正确的pin_num_y:", pin_y_num)
            except:
                pass
    return pin_x_num, pin_y_num


def get_absolute_pitch(pitch_x, pitch_y, bottom_ocr_data):
    """从 OCR 文本中解析绝对的间距。"""

    '''
    for i in range(len(bottom_ocr_data)):
        if bottom_ocr_data[i]['Absolutely'] == 'pitch_x':
            try:
                pitch_x = [bottom_ocr_data[i]['max_medium_min'][0]]
                print("找到绝对正确的pitch_x:", pitch_x)
            except:
                pass
        if bottom_ocr_data[i]['Absolutely'] == 'pitch_y':
            try:
                pitch_y = [bottom_ocr_data[i]['max_medium_min'][0]]
                print("找到绝对正确的pitch_y:", pitch_y)
            except:
                pass
    return pitch_x, pitch_y


def get_absolute_pin_diameter(pin_diameter, top_ocr_data, bottom_ocr_data, side_ocr_data):
    """从 OCR 文本中解析绝对的 pin 直径。"""
    mb_pin_diameter = np.zeros((0, 3))  # 记录所有可能的pin直径
    mb_ratio = []  # 记录每个pin直径的置信度
    zer = np.zeros((0, 3))

    for i in range(len(bottom_ocr_data)):
        if bottom_ocr_data[i]['Absolutely'] == 'pin_diameter':
            mb_pin_diameter = np.r_[mb_pin_diameter, [bottom_ocr_data[i]['max_medium_min']]]
            mb_ratio.append(1)
        if bottom_ocr_data[i]['Absolutely'] == 'pin_diameter+':
            mb_pin_diameter = np.r_[mb_pin_diameter, [bottom_ocr_data[i]['max_medium_min']]]
            mb_ratio.append(2)
    for i in range(len(side_ocr_data)):
        if side_ocr_data[i]['Absolutely'] == 'pin_diameter':
            mb_pin_diameter = np.r_[mb_pin_diameter, [side_ocr_data[i]['max_medium_min']]]
            mb_ratio.append(1)
        if side_ocr_data[i]['Absolutely'] == 'pin_diameter+':
            mb_pin_diameter = np.r_[mb_pin_diameter, [side_ocr_data[i]['max_medium_min']]]
            mb_ratio.append(2)
    for i in range(len(top_ocr_data)):
        if top_ocr_data[i]['Absolutely'] == 'pin_diameter':
            mb_pin_diameter = np.r_[mb_pin_diameter, [top_ocr_data[i]['max_medium_min']]]
            mb_ratio.append(1)
        if top_ocr_data[i]['Absolutely'] == 'pin_diameter+':
            mb_pin_diameter = np.r_[mb_pin_diameter, [top_ocr_data[i]['max_medium_min']]]
            mb_ratio.append(2)
    print("可能的pin直径:\n", mb_pin_diameter)
    '''
    将全为0的pin直径置信度为0，将三值不等的pin直径置信度加1
    '''
    for i in range(len(mb_ratio)):
        if mb_pin_diameter[i][0] == mb_pin_diameter[i][1] == mb_pin_diameter[i][2] == 0:
            mb_ratio[i] = 0
        if mb_pin_diameter[i][0] != mb_pin_diameter[i][1] or mb_pin_diameter[i][1] != mb_pin_diameter[i][2]:
            mb_ratio[i] += 1
    try:
        max_value = max(mb_ratio)  # 求列表最大值
        max_idx = mb_ratio.index(max_value)  # 求最大值对应索引
        if np.array_equal(zer, pin_diameter):
            pin_diameter = np.r_[pin_diameter, [mb_pin_diameter[max_idx]]]
            print("找到绝对正确的pin直径:\n", pin_diameter)

    except:
        pass

    return pin_diameter


def get_pin_diameter_table_absolute(top_ocr_data, bottom_ocr_data, side_ocr_data):
    """在表格信息中获取 pin 直径。"""
    pin_diameter = ''
    if pin_diameter == '':
        for i in range(len(bottom_ocr_data)):
            if bottom_ocr_data[i]['Absolutely'] == 'pin_diameter':
                pin_diameter = bottom_ocr_data[i]['ocr_strings']
                print('找到绝对正确的pin_diameter', pin_diameter)
    if pin_diameter == '':
        for i in range(len(side_ocr_data)):
            if side_ocr_data[i]['Absolutely'] == 'pin_diameter':
                pin_diameter = side_ocr_data[i]['ocr_strings']
                print('找到绝对正确的pin_diameter', pin_diameter)
    if pin_diameter == '':
        for i in range(len(top_ocr_data)):
            if top_ocr_data[i]['Absolutely'] == 'pin_diameter':
                pin_diameter = top_ocr_data[i]['ocr_strings']
                print('找到绝对正确的pin_diameter', pin_diameter)
    return pin_diameter


def output_table_BGA(body_x_yinXian, body_y_yinXian, pitch_x_yinXian, pitch_y_yinXian, high_yinXian,
    """输出 BGA 参数表格结果。"""
                     pin_diameter_yinXian, standoff_yinXian, pin_num_x_serial, pin_num_y_serial, pin_1_location,
                     table_dic):
    body_x = np.zeros(3)
    body_y = np.zeros(3)
    pitch_x = []
    pitch_y = []
    high = np.zeros(3)
    pin_diameter = np.zeros((0, 3))
    standoff = np.zeros((0, 3))
    pin_num_x = 0
    pin_num_y = 0
    acc = 0  # 记录几个是直接从表中读出来的
    for i in range(len(table_dic)):
        if body_x_yinXian == table_dic[i]['data']:
            body_x = table_dic[i]['max_medium_min']
        elif table_dic[i]['data'] == 'E':
            body_x = table_dic[i]['max_medium_min']
            acc += 1

        if body_y_yinXian == table_dic[i]['data']:
            body_y = table_dic[i]['max_medium_min']
        elif table_dic[i]['data'] == 'D':
            body_y = table_dic[i]['max_medium_min']
            acc += 1
        if pitch_x_yinXian == table_dic[i]['data']:
            pitch_x = table_dic[i]['max_medium_min']
        elif table_dic[i]['data'] == 'e':
            pitch_x = table_dic[i]['max_medium_min']
            acc += 1
        if pitch_y_yinXian == table_dic[i]['data']:
            pitch_y = table_dic[i]['max_medium_min']
        elif table_dic[i]['data'] == 'e1':
            pitch_y = table_dic[i]['max_medium_min']
            acc += 1
        if high_yinXian == table_dic[i]['data']:
            high = table_dic[i]['max_medium_min']
        elif table_dic[i]['data'] == 'A':
            high = table_dic[i]['max_medium_min']
            acc += 1
        if pin_diameter_yinXian == table_dic[i]['data']:
            pin_diameter = table_dic[i]['max_medium_min']
        elif table_dic[i]['data'] == 'b':
            pin_diameter = table_dic[i]['max_medium_min']
            acc += 1
        if standoff_yinXian == table_dic[i]['data']:
            standoff = table_dic[i]['max_medium_min']
        elif table_dic[i]['data'] == 'A1':
            standoff = table_dic[i]['max_medium_min']
            acc += 1
        if pin_num_x_serial != 0:
            pin_num_x = pin_num_x_serial
        elif table_dic[i]['data'] == 'n_x':
            pin_num_x = table_dic[i]['max_medium_min']
            acc += 1
        if pin_num_y_serial != 0:
            pin_num_y = pin_num_y_serial
        elif table_dic[i]['data'] == 'n_y':
            pin_num_y = table_dic[i]['max_medium_min']
            acc += 1

    if pitch_x == [] and pitch_y != []:
        pitch_x = pitch_y
    if pitch_y == [] and pitch_x != []:
        pitch_y = pitch_x

    if np.array_equal(body_x, np.zeros(3)) and not np.array_equal(body_y, np.zeros(3)):
        body_x = body_y
    if np.array_equal(body_y, np.zeros(3)) and not np.array_equal(body_x, np.zeros(3)):
        body_y = body_x

    if pin_num_x == 0 and pin_num_y != 0:
        pin_num_x = pin_num_y
    if pin_num_y == 0 and pin_num_x != 0:
        pin_num_y = pin_num_x
    return body_x, body_y, pitch_x, pitch_y, high, pin_diameter, standoff, pin_num_x, pin_num_y, pin_1_location


def process_dic_1(table_dic, pin_num_x_serial, pin_num_y_serial):
    """处理表格字典的第一阶段整理。"""
    if table_dic[5] == ['', '', '', '']:
        if pin_num_y_serial == 0:
            a = 0
            try:
                a = round(table_dic[10][2] / table_dic[7][2]) + 1
            except:
                print("D1和e关系不正确")
            if a != 0:
                pin_num_y_serial = a
        table_dic[5] = ['', '', int(pin_num_y_serial), '']
    if table_dic[6] == ['', '', '', '']:
        if pin_num_x_serial == 0:
            b = 0
            try:
                b = round(table_dic[11][2] / table_dic[8][2]) + 1
            except:
                print("E1和e关系不正确")
            if b != 0:
                pin_num_x_serial = b
        table_dic[6] = ['', '', int(pin_num_x_serial), '']
    return table_dic


def process_dic_2(table_dic):
    """处理表格字典的第二阶段整理。"""
    ze = ['', '', '', '']
    key = 0  # 0代表经验判断长宽不等，1代表经验判断长宽相等
    if table_dic[0] == table_dic[1] or table_dic[5] == table_dic[6] or table_dic[7] == table_dic[8]:
        key = 1
        # table_dic[0]和table_dic[0]中一个等于ze另一个不等于就把table_dic[0]的值赋给table_dic[1]或者table_dic[1]的值赋给table_dic[0]
        if table_dic[0] == ze and table_dic[1] != ze:
            table_dic[0] = table_dic[1]
        if table_dic[1] == ze and table_dic[0] != ze:
            table_dic[1] = table_dic[0]
        # table_dic[5]和table_dic[6]中一个等于ze另一个不等于就把table_dic[5]的值赋给table_dic[6]或者table_dic[6]的值赋给table_dic[5]
        if table_dic[5] == ze and table_dic[6] != ze:
            table_dic[5] = table_dic[6]
        if table_dic[6] == ze and table_dic[5] != ze:
            table_dic[6] = table_dic[5]
        # table_dic[7]和table_dic[8]中一个等于ze另一个不等于就把table_dic[7]的值赋给table_dic[8]或者table_dic[8]的值赋给table_dic[7]

        if table_dic[7] == ze and table_dic[8] != ze:
            table_dic[7] = table_dic[8]
        if table_dic[8] == ze and table_dic[7] != ze:
            table_dic[8] = table_dic[7]
        # 当行列pitch都没值时，利用E1/(num - 1) = pitch计算出pitch
        if table_dic[7] == ze and table_dic[8] == ze:
            if table_dic[10][2] != 0 and table_dic[5][2] != 0:
                b = 0
                try:
                    b = round(table_dic[10][2] / (table_dic[5][2] - 1))
                except:
                    print("E1和e关系不正确")
                if b != 0:
                    pitch_x = b
                    table_dic[7] = ['', '', int(pitch_x), '']

            if table_dic[11][2] != 0 and table_dic[6][2] != 0:
                b = 0
                try:
                    b = round(table_dic[11][2] / (table_dic[6][2] - 1))
                except:
                    print("E1和e关系不正确")
                if b != 0:
                    pitch_y = b
                    table_dic[8] = ['', '', int(pitch_y), '']

    return table_dic


def process_dic_3(table_dic):
    """处理表格字典的第三阶段整理。"""
    ze = ['', '', '', '']
    ze1 = ['', 0, 0, 0]
    if table_dic[7] != ze and table_dic[8] == ze:
        table_dic[8] = table_dic[7]
    if table_dic[8] != ze and table_dic[7] == ze:
        table_dic[7] = table_dic[8]
    if table_dic[7] != ze1 and table_dic[7] != ze and table_dic[8] == ze1:
        table_dic[8] = table_dic[7]
    if table_dic[8] != ze1 and table_dic[8] != ze and table_dic[7] == ze1:
        table_dic[7] = table_dic[8]

    return table_dic


def process_dic_4(table_dic):
    """处理表格字典的第四阶段整理。"""
    ze = ['', 0, 0, 0]
    if table_dic[0] != ze:
        body_x_yinXian = np.array([[table_dic[0][3], table_dic[0][2], table_dic[0][1]]])
    else:
        body_x_yinXian = np.zeros((1, 3))
    if table_dic[1] != ze:
        body_y_yinXian = np.array([[table_dic[1][3], table_dic[1][2], table_dic[1][1]]])
    else:
        body_y_yinXian = np.zeros((1, 3))
    if table_dic[2] != ze:
        high_yinXian = np.array([table_dic[2][3], table_dic[2][2], table_dic[2][1]])
    else:
        high_yinXian = np.zeros(3)
    if table_dic[3] != ze:
        standoff_yinXian = np.array([[table_dic[3][3], table_dic[3][2], table_dic[3][1]]])
    else:
        standoff_yinXian = np.zeros((1, 3))
    if table_dic[4] != ze:
        pin_diameter_yinXian = np.array([[table_dic[4][3], table_dic[4][2], table_dic[4][1]]])
    else:
        pin_diameter_yinXian = np.zeros((1, 3))
    if table_dic[5] != ze:
        pin_num_y_serial = table_dic[5][2]
    else:
        pin_num_y_serial = 0
    if table_dic[6] != ze:
        pin_num_x_serial = table_dic[6][2]
    else:
        pin_num_x_serial = 0
    if table_dic[7] != ze:
        pitch_x_yinXian = [table_dic[7][2]]
    else:
        pitch_x_yinXian = []
    if table_dic[8] != ze:
        pitch_y_yinXian = [table_dic[8][2]]
    else:
        pitch_y_yinXian = []

    return body_x_yinXian, body_y_yinXian, \
        pitch_x_yinXian, pitch_y_yinXian, \
        high_yinXian, pin_diameter_yinXian, standoff_yinXian, \
        pin_num_x_serial, pin_num_y_serial


def find_pin_diameter_last(pin_diameter, side_data_np, high):
    """在多轮计算后给出最终 pin 直径。"""

    '''
    likely_pin_diameter = np.zeros((0, 3))
    side_1 = 0
    side_2 = 0
    key = -1
    for i in range(len(side_data_np)):
        if side_data_np[i][0] == 1:
            side_1 += 1
        if side_data_np[i][0] == 0:
            side_2 += 1
    if side_1 > side_2 and side_2 != 0:
        key = 0
    if side_2 > side_1 and side_1 != 0:
        key = 1

    if key != -1:
        for i in range(len(side_data_np)):
            if side_data_np[i][0] == key:
                # 如果side_data_np中某一行的后三位都大于0.2并小于0.7
                if 0.2 < side_data_np[i][1] < 1 and 0.2 < side_data_np[i][2] < 1 and 0.2 < side_data_np[i][3] < 1 and \
                        side_data_np[i][1] < high[0]:
                    likely_pin_diameter = np.r_[likely_pin_diameter, [side_data_np[i, 1: 4]]]
        print("like", likely_pin_diameter)
        if len(likely_pin_diameter) == 0:
            return pin_diameter
        if len(likely_pin_diameter) == 1:
            pin_diameter = likely_pin_diameter
            print("根据side视图特殊位置确定pin_diameter:", pin_diameter)

            return pin_diameter
        if len(likely_pin_diameter) > 1:
            # 选最大的一个作为pin_diameter
            pin_diameter = likely_pin_diameter[np.argmax(likely_pin_diameter[:, 0])]
            print("根据side视图特殊位置确定pin_diameter:", pin_diameter)
            return pin_diameter
    else:
        for i in range(len(side_data_np)):
            if 0.2 < side_data_np[i][1] < 1 and 0.2 < side_data_np[i][2] < 1 and 0.2 < side_data_np[i][3] < 1 and \
                    side_data_np[i][1] < high[0]:
                likely_pin_diameter = np.r_[likely_pin_diameter, [side_data_np[i, 1: 4]]]
        try:
            pin_diameter = likely_pin_diameter[np.argmax(likely_pin_diameter[:, 0])]
        except:
            pin_diameter = np.zeros((0, 3))
        print("根据side视图pin直径的固定大小范围确定pin_diameter:", pin_diameter)
        return pin_diameter


def yinXian_begain_get_data_present(test_mode, letter_or_number, table_dic):
    """封装整个流程的入口，执行匹配与展示。"""
    # empty_folder(r'opencv_output')
    # os.makedirs(r'opencv_output')
    # 1.将备份的三视图重新载入data
    # 清空文件夹
    empty_folder(DATA)
    # 创建文件夹
    os.makedirs(DATA)
    filePath = DATA_COPY
    file_name_list = os.listdir(filePath)
    for file_name in file_name_list:
        shutil.copy(f'{DATA_COPY}/{file_name}', f'{DATA}/{file_name}')
    # 2.检测标尺线与尺寸数字并匹配
    yolox_pairs_top, yolox_pairs_bottom, yolox_pairs_side, \
        top_yolox_pairs_length, bottom_yolox_pairs_length, side_yolox_pairs_length, \
        serial_numbers_data, serial_letters_data, serial_numbers, serial_letters, \
        letter_or_number, top_ocr_data, bottom_ocr_data, side_ocr_data, pin_map, top_border, bottom_border, bottom_pin, color = \
        yolox_dbnet_ocr_match(test_mode, letter_or_number)
    if letter_or_number == 'number':
        yolox_pairs_top_copy = yolox_pairs_top.copy()
        yolox_pairs_bottom_copy = yolox_pairs_bottom.copy()
        yolox_pairs_side_copy = yolox_pairs_side.copy()

        # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]
        # top_yolox_pairs_length,np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
        # 根据serilal_numbers和serial_letters输出pin行列数和pin1位置
        pin_num_x_serial, pin_num_y_serial, pin_1_location = find_pin_num_pin_1(serial_numbers_data,
                                                                                serial_letters_data,
                                                                                serial_numbers, serial_letters)
        print("pin_num_x_serial, pin_num_y_serial, pin_1_location", pin_num_x_serial, pin_num_y_serial, pin_1_location)
        # bottom视图用引线下引方法给出参数
        body_x_yinXian, body_y_yinXian = yinXinan_find_body(yolox_pairs_top, top_yolox_pairs_length, yolox_pairs_bottom,
                                                            bottom_yolox_pairs_length, top_border, bottom_border)
        print('body_x_yinXian,body_y_yinXian', body_x_yinXian, body_y_yinXian)
        '''
        onnx未集成PIN球，无法使用函数
        '''
        # bottom根据引线找pitch
        pitch_x_yinXian, pitch_y_yinXian = yinXinan_find_pitch(yolox_pairs_bottom, bottom_yolox_pairs_length, bottom_pin)
        print('pitch_x_yinXian, pitch_y_yinXian', pitch_x_yinXian, pitch_y_yinXian)
        # bottom和side视图，根据引线找pin_diameter
        pin_diameter_yinXian = yinXinan_find_pin_diameter(yolox_pairs_bottom, bottom_yolox_pairs_length, bottom_pin)
        print("pin_diameter_yinXian", pin_diameter_yinXian)
        # pitch_x_yinXian = []
        # pitch_y_yinXian = []
        # pin_diameter_yinXian = []
        '''
        onnx未集成BGAside
        '''
        # # side视图根据引线找high和standoff
        # high_yinXian, standoff_yinXian = yinXian_find_side_high_standoff(yolox_pairs_side, side_yolox_pairs_length)
        # print('high_yinXian, standoff_yinXian', high_yinXian, standoff_yinXian)
        high_yinXian = np.zeros((0, 3))
        standoff_yinXian = np.zeros((0, 3))

        return body_x_yinXian, body_y_yinXian, pitch_x_yinXian, pitch_y_yinXian, high_yinXian, pin_diameter_yinXian, standoff_yinXian, \
            pin_num_x_serial, pin_num_y_serial, pin_1_location, \
            yolox_pairs_top_copy, yolox_pairs_bottom_copy, yolox_pairs_side_copy, letter_or_number, \
            top_ocr_data, bottom_ocr_data, side_ocr_data, pin_map, color
    if letter_or_number == 'table':

        yolox_pairs_top_copy = ''
        # yolox_pairs_top np.(,9)[x1,y1,x2,y2,x1,y1,x2,y2,'e1']
        yolox_pairs_bottom_copy = ''
        yolox_pairs_side_copy = ''

        # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2,'e1']
        # top_yolox_pairs_length,np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
        # 根据serilal_numbers和serial_letters输出pin行列数和pin1位置
        pin_num_x_serial, pin_num_y_serial, pin_1_location = find_pin_num_pin_1(serial_numbers_data,
                                                                                serial_letters_data,
                                                                                serial_numbers, serial_letters)
        print("pin_num_x_serial, pin_num_y_serial, pin_1_location", pin_num_x_serial, pin_num_y_serial, pin_1_location)
        # 取行pin数和列pin数
        pin_num_txt = r'yolox_data\pin_num.txt'
        pin_num_x_y = get_np_array_in_txt(pin_num_txt)
        pin_x_num = int(pin_num_x_y[0][0])  # 行pin数
        pin_y_num = int(pin_num_x_y[1][0])  # 列pin数
        # 缺补存正pin_num
        if pin_num_x_serial == 0:
            pin_num_x_serial = pin_x_num
        if pin_num_y_serial == 0:
            pin_num_y_serial = pin_y_num
        if pin_x_num < pin_num_x_serial:
            pin_x_num = pin_num_x_serial
        if pin_y_num < pin_num_y_serial:
            pin_y_num = pin_num_y_serial
        if pin_x_num == 0:
            pin_x_num = pin_num_x_serial
        if pin_y_num == 0:
            pin_y_num = pin_num_y_serial
        # body_x_yinXian = ''
        # body_y_yinXian = ''
        # pitch_x_yinXian = ''
        # pitch_y_yinXian = ''
        # high_yinXian = ''
        # pin_diameter_yinXian = ''
        # standoff_yinXian = ''

        '''
        table_dic = list(10, 4)
        [['', 10, 10, 10], 实体长
        ['', 10, 10, 10], 实体宽
        ['', 10, 10, 10], 实体高
        ['', 10, 10, 10], 支撑高
        ['', 10, 10, 10], 球直径
        ['', 10, 10, 10], 行数
        ['', 10, 10, 10], 列数
        ['', 10, 10, 10], 行pitch
        ['', 10, 10, 10], 列pitch
        ['', '', '', '']] 缺pin
        '''
        print("table_dic", table_dic)
        # 填充行数,列数
        # table_dic = process_dic_1(table_dic, pin_num_x_serial, pin_num_y_serial)
        table_dic = process_dic_1(table_dic, pin_x_num, pin_y_num)
        print("table_dic", table_dic)
        # 补充长和宽，行数和列数，横向pitch和竖向pitch
        table_dic = process_dic_2(table_dic)
        print("table_dic", table_dic)
        # 修正数据，将''改为0
        for i in range(len(table_dic)):
            for j in range(len(table_dic[i])):
                if j > 0:
                    if table_dic[i][j] == '':
                        table_dic[i][j] = 0
        print("table_dic", table_dic)
        # 逻辑矫正
        table_dic = process_dic_3(table_dic)
        print("table_dic", table_dic)
        # 输出结果
        body_x_yinXian, body_y_yinXian, \
            pitch_x_yinXian, pitch_y_yinXian, \
            high_yinXian, pin_diameter_yinXian, standoff_yinXian, \
            pin_num_x_serial, pin_num_y_serial = process_dic_4(table_dic)

        return body_x_yinXian, body_y_yinXian, pitch_x_yinXian, pitch_y_yinXian, high_yinXian, pin_diameter_yinXian, standoff_yinXian, pin_num_x_serial, pin_num_y_serial, pin_1_location, yolox_pairs_top_copy, yolox_pairs_bottom_copy, yolox_pairs_side_copy, letter_or_number, top_ocr_data, bottom_ocr_data, side_ocr_data, pin_map


def begain_get_pairs_data_present2(body_x_yinXian, body_y_yinXian, pitch_x_yinXian, pitch_y_yinXian, high_yinXian,
    """在数据齐备情况下汇总 BGA 参数。"""
                                   pin_diameter_yinXian, standoff_yinXian, pin_num_x_serial, pin_num_y_serial,
                                   yolox_pairs_top_copy, yolox_pairs_bottom_copy, yolox_pairs_side_copy, pin_1_location,
                                   test_mode, top_ocr_data, bottom_ocr_data, side_ocr_data):
    zer = np.zeros(3)
    # 1.将备份的三视图重新载入data
    # 清空文件夹
    empty_folder(DATA)
    # 创建文件夹
    os.makedirs(DATA)
    filePath = DATA_COPY
    file_name_list = os.listdir(filePath)
    for file_name in file_name_list:
        shutil.copy(f'{DATA_COPY}/{file_name}', f'{DATA}/{file_name}')

    yolox_pairs_top = yolox_pairs_top_copy
    yolox_pairs_bottom = yolox_pairs_bottom_copy
    yolox_pairs_side = yolox_pairs_side_copy
    # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]

    # 先取行pin数和列pin数
    pin_num_txt = f'{YOLOX_DATA}\pin_num.txt'
    pin_num_x_y = get_np_array_in_txt(pin_num_txt)
    pin_x_num = int(pin_num_x_y[0][0])  # 行pin数
    pin_y_num = int(pin_num_x_y[1][0])  # 列pin数
    pin_txt = f'{YOLOX_DATA}\pin.txt'
    pin = get_np_array_in_txt(pin_txt)

    top_data_np = tf(yolox_pairs_top)  # 位数缩小
    # top_data_np:np(,5)[标尺线方向1 = 竖向 0 = 横向 0.5 = 未匹配到标尺线, 0 , max , medium , min]
    top_data_np = correct_top_data(top_data_np)  # 修正top视图中的data数据,如果top视图没有data则三视图的矫正不会起作用

    side_data_np = tf(yolox_pairs_side)  # 位数缩小
    # side_data_np:np(,5)[标尺线方向1 = 竖向 0 = 横向 0.5 = 未匹配到标尺线, 0 , max , medium , min]
    side_data_np = correct_bottom_side_data(top_data_np, side_data_np)  # 修正side视图中的data数据

    bottom_data_np = tf(yolox_pairs_bottom)  # 位数缩小
    # bottom_data_np:np(,5)[标尺线方向1 = 竖向 0 = 横向 0.5 = 未匹配到标尺线, 0 , max , medium , min]
    bottom_data_np = correct_bottom_side_data(top_data_np, bottom_data_np)  # 修正bottom视图中的data数据

    top_data_np = np.delete(top_data_np, 1, 1)
    # top_data_np:np(,4)[标尺线方向1 = 竖向 0 = 横向 0.5 = 未匹配到标尺线, max , medium , min]
    bottom_data_np = np.delete(bottom_data_np, 1, 1)
    # bottom_data_np:np(,4)[标尺线方向1 = 竖向 0 = 横向 0.5 = 未匹配到标尺线, max , medium , min]
    side_data_np = np.delete(side_data_np, 1, 1)
    # side_data_np:np(,4)[标尺线方向1 = 竖向 0 = 横向 0.5 = 未匹配到标尺线, max , medium , min]
    print("top_data_np:np(,4)[标尺线方向1 = 竖向 0 = 横向 0.5 = 未匹配到标尺线, max , medium , min]\n", top_data_np)
    print("bottom_data_np:np(,4)[标尺线方向1 = 竖向 0 = 横向 0.5 = 未匹配到标尺线, max , medium , min]\n",
          bottom_data_np)
    print("side_data_np:np(,4)[标尺线方向1 = 竖向 0 = 横向 0.5 = 未匹配到标尺线, max , medium , min]\n", side_data_np)
    print("############输出参数############")
    print("********************总结********************")
    print("body_x_yinXian, body_y_yinXian(max_medium_min)\n", body_x_yinXian, body_y_yinXian)
    print("pin_num_x_serial, pin_num_y_serial, pin_1_location\n", pin_num_x_serial, pin_num_y_serial,
          pin_1_location)
    print("pitch_x_yinXian, pitch_y_yinXian(max_medium_min)\n", pitch_x_yinXian, pitch_y_yinXian)
    print("high_yinXian(max_medium_min)\n", high_yinXian)
    print("pin_diameter_yinXian(max_medium_min)\n", pin_diameter_yinXian)
    print("standoff_yinXian(max_medium_min)\n", standoff_yinXian)

    # 缺补存正pin_num
    if pin_num_x_serial == 0:
        pin_num_x_serial = pin_x_num
    if pin_num_y_serial == 0:
        pin_num_y_serial = pin_y_num
    if pin_x_num < pin_num_x_serial:
        pin_x_num = pin_num_x_serial
    if pin_y_num < pin_num_y_serial:
        pin_y_num = pin_num_y_serial
    if pin_x_num == 0:
        pin_x_num = pin_num_x_serial
    if pin_y_num == 0:
        pin_y_num = pin_num_y_serial
    # pin_x_num = pin_num_x_serial
    # pin_y_num = pin_num_y_serial

    print("每行最大pin数,每列最大pin数", pin_x_num, pin_y_num)

    if pin_x_num != 0 and pin_y_num != 0:  # 不存在整列或者整行缺失时

        average_pitch_x_y_txt = f'{YOLOX_DATA}/average_x_y.txt'
        average_pitch_x_y = get_np_array_in_txt(average_pitch_x_y_txt)
        average_pitch_x = average_pitch_x_y[0][0]
        average_pitch_y = average_pitch_x_y[1][0]

        # 对比top和bottom中最长的pairs来判断长和宽在哪个视图中
        global top_or_bottom  # 标记长和宽在哪个视图中

        if compare_top_bottom(top_data_np, bottom_data_np):
            top_or_bottom = 'top'
            body_x, body_y = get_body_x_y(top_data_np)  # 长和宽
        else:
            top_or_bottom = 'bottom'
            body_x, body_y = get_body_x_y(bottom_data_np)  # 长和宽
        print("body_x, body_y(max,medium,min)", body_x, body_y)
        # print("body_x_shiti,body_y_shiti",body_x_shiti,body_y_shiti)
        # 缺补存正body
        try:
            if (body_x != body_y).any():
                if (body_x != body_x_yinXian).any():
                    body_x_yinXian = body_x
                if (body_y != body_y_yinXian).any():
                    body_y_yinXian = body_y
            if (body_x_yinXian != body_y_yinXian).any():
                if (body_x != body_x_yinXian).any():
                    body_x = body_x_yinXian
                if (body_y != body_y_yinXian).any():
                    body_y = body_y_yinXian
            if (body_x_yinXian == zer).all():
                body_x_yinXian = body_x
            if (body_y_yinXian == zer).all():
                body_y_yinXian = body_y
        except:
            print("报错，可能是body_x, body_y(max,medium,min) [] []")
        body_x = body_x_yinXian
        body_y = body_y_yinXian

        pitch_x, pitch_y, pin_x_num_new, pin_y_num_new, bottom_ocr_data = get_pitch_x_y(bottom_data_np, pin_x_num,
                                                                                        pin_y_num, body_x,
                                                                                        body_y,
                                                                                        bottom_ocr_data)  # 算出行和列的pitch值

        if len(pitch_x) != 0 or len(pitch_y) != 0:
            print("pitch_x,pitch_y", pitch_x, pitch_y)
        if pin_x_num_new != pin_x_num or pin_y_num_new != pin_y_num:
            print("修正之后的每行最大pin数,每列最大pin数", pin_x_num_new, pin_y_num_new)
            pin_x_num = pin_x_num_new
            pin_y_num = pin_y_num_new
            # 缺补存正pin_num
            if pin_num_x_serial == 0 or pin_num_x_serial < pin_x_num:
                pin_num_x_serial = pin_x_num
            if pin_num_y_serial == 0 or pin_num_y_serial < pin_y_num:
                pin_num_y_serial = pin_y_num
            pin_x_num = pin_num_x_serial
            pin_y_num = pin_num_y_serial
        # 根据BGA的绝对公式找pin_num
        pin_x_num, pin_y_num = get_absolute_pin_num(pin_x_num, pin_y_num, bottom_ocr_data)
        print("每行最大pin数,每列最大pin数", pin_x_num, pin_y_num)

        if len(pitch_x) == len(pitch_y) == 0:  # 当等式失效时用不等式求pitch
            pitch_x, pitch_y = get_pitch_when_lone(bottom_data_np, pin_x_num, pin_y_num, body_x, body_y)
        # 缺补存正pitch
        if (pitch_x_yinXian == np.array([0])).all():
            pitch_x_yinXian = pitch_x
        pitch_x = pitch_x_yinXian
        if (pitch_y_yinXian == np.array([0])).all():
            pitch_y_yinXian = pitch_y
        pitch_y = pitch_y_yinXian
        try:
            if len(pitch_x) == 0 and len(pitch_y) != 0:
                pitch_x = pitch_y
            if len(pitch_y) == 0 and len(pitch_x) != 0:
                pitch_y = pitch_x
        except:
            pass
        # 根据BGA的绝对公式找pitch
        pitch_x, pitch_y = get_absolute_pitch(pitch_x, pitch_y, bottom_ocr_data)

        # 提取high
        high = np.zeros(3)
        zer = np.zeros(3)
        if (body_x != zer).any() and (body_y != zer).any():
            high = get_high_pin_high_max_1(side_data_np, body_x, body_y)
            print("1.high(max,medium,min)", high)
        if (high == zer).all():
            high = get_high_pin_high_max(side_data_np)
            print("2.high(max,medium,min)", high)
        # 缺补存正high
        if np.array_equal(high, np.zeros(3)):
            high = high_yinXian
        high_yinXian = high
        # 根据BGA的绝对特征找high
        high = get_absolute_high(high, side_ocr_data)

        # 提取球直径
        pin_diameter = np.zeros((0, 3))
        zer = np.zeros((0, 3))
        # 根据BGA的绝对特征找pin_diameter
        pin_diameter_absolute = get_absolute_pin_diameter(pin_diameter, top_ocr_data, bottom_ocr_data, side_ocr_data)
        pin_diameter = pin_diameter_absolute.copy()
        standoff = np.zeros((0, 3))
        # if (standoff_yinXian != zer).all() and (pin_diameter_yinXian == zer).all():
        if (not np.array_equal(standoff_yinXian, zer)) and np.array_equal(pin_diameter_yinXian, zer):
            pin_diameter = pin_diameter_1(pitch_x, pitch_y, pin_x_num, pin_y_num, body_x, body_y, bottom_data_np,
                                          side_data_np, top_data_np, standoff_yinXian, high)
            print("1.找到支撑高和高的情况下寻找pin直径", pin_diameter)
        if len(pin_diameter) == 0:
            pin_diameter = get_pin_diameter(pitch_x, pitch_y, pin_x_num, pin_y_num, body_x, body_y, bottom_data_np,
                                            side_data_np, top_data_np)
        if len(pin_diameter) != 0:
            print("2.找到可能是pin直径的数据(max,medium,min)\n", pin_diameter)
        else:
            print("2.算法暂未找到pin直径数据")

        if len(pin_diameter) != 0:
            if np.array_equal(pin_diameter_absolute, zer):
                standoff = find_standoff(side_data_np, pin_diameter, high)
            else:
                standoff = find_standoff(side_data_np, pin_diameter_absolute, high)
            print("1.找到可能是支撑高的数据\n", standoff)

        if len(pin_diameter) == 0:
            pin_diameter = find_pin_diameter(pin_diameter, high, top_data_np, bottom_data_np, side_data_np, pitch_x,
                                             pitch_y)
            print("3.找到可能是pin直径的数据(max,medium,min)\n", pin_diameter)
        if len(pin_diameter) == 0:
            '''
            side视图中寻找少数方向种类中满足大概范围的标注作为pin直径       
            '''
            pin_diameter = find_pin_diameter_last(pin_diameter, side_data_np, high)

        pin_map_present = show_lost_pin_when_full(pin, pin_x_num, pin_y_num, average_pitch_x, average_pitch_y)
        if len(pin_diameter) == 0:
            standoff = np.zeros((0, 3))
        if len(pin_diameter) != 0:
            if np.array_equal(pin_diameter_absolute, zer):
                standoff = find_standoff(side_data_np, pin_diameter, high)
            else:
                standoff = find_standoff(side_data_np, pin_diameter_absolute, high)
            print("2.找到可能是支撑高的数据\n", standoff)

        dimension = pin_diameter_absolute.ndim
        # print(pin_diameter_absolute, pin_diameter_absolute.shape)
        if dimension == 1:
            zer = np.zeros((0, 3))
            if not ((
                    pin_diameter_absolute != np.zeros(3)).all()):
                pin_diameter = pin_diameter_absolute
                print("1.矫正为正确的pin_diameter:\n", pin_diameter)

        elif dimension == 2:
            if not (np.array_equal(pin_diameter_absolute,
                                   np.zeros((0, 3)))):
                pin_diameter = pin_diameter_absolute
                print("2.矫正为正确的pin_diameter:\n", pin_diameter)
        # 修正格式
        if pin_diameter.ndim == 1:
            zer = np.zeros((0, 3))
            pin_diameter = np.r_[zer, [pin_diameter]]
            print("1.矫正为统一的格式pin_diameter:\n", pin_diameter)

        if standoff_yinXian.ndim == 1:
            zer = np.zeros((0, 3))
            standoff_yinXian = np.r_[zer, [standoff_yinXian]]
            print("2.矫正为统一的格式standoff_yinXian:\n", standoff_yinXian)
        # 缺补存正 standoff
        print("standoff", standoff)
        print("standoff_yinXian", standoff_yinXian)
        if np.array_equal(standoff_yinXian, np.zeros((0, 3))) or np.array_equal(standoff_yinXian, np.zeros((1, 3))) or (
                standoff_yinXian > pin_diameter).any() or (
                0.1 >= standoff_yinXian[0]).any() or (standoff_yinXian[0] >= 0.65).any():
            standoff_yinXian = standoff
        standoff = standoff_yinXian
        # 当standoff不止一个时，挑选最合适的
        standoff = select_best_stanoff(standoff)

        return body_x, body_y, pin_x_num, pin_y_num, pitch_x, pitch_y, high, pin_diameter, standoff, pin_map_present
    if pin_x_num == 0 or pin_y_num == 0:  # 整行或整列缺pin时
        if pin_x_num == 0:
            key = 1
        if pin_y_num == 1:
            key = 0

        # 对比top和bottom中最长的pairs来判断长和宽在哪个视图中
        # global top_or_bottom  # 标记长和宽在哪个视图中

        if compare_top_bottom(top_data_np, bottom_data_np):
            top_or_bottom = 'top'
            body_x, body_y = get_body_x_y(top_data_np)  # 长和宽
        else:
            top_or_bottom = 'bottom'
            body_x, body_y = get_body_x_y(bottom_data_np)  # 长和宽
        print("body_x, body_y(max,medium,min)", body_x, body_y)
        # print("body_x_shiti,body_y_shiti",body_x_shiti,body_y_shiti)
        # 缺补存正body
        try:
            if (body_x != body_y).any():
                if (body_x != body_x_yinXian).any():
                    body_x_yinXian = body_x
                if (body_y != body_y_yinXian).any():
                    body_y_yinXian = body_y
            if (body_x_yinXian != body_y_yinXian).any():
                if (body_x != body_x_yinXian).any():
                    body_x = body_x_yinXian
                if (body_y != body_y_yinXian).any():
                    body_y = body_y_yinXian
            if (body_x_yinXian == zer).all():
                body_x_yinXian = body_x
            if (body_y_yinXian == zer).all():
                body_y_yinXian = body_y
        except:
            print("报错，可能是body_x, body_y(max,medium,min) [] []")
        body_x = body_x_yinXian
        body_y = body_y_yinXian

        pitch_x, pitch_y, pin_num_x, pin_num_y = get_pitch_x_y_when_absence_pin(bottom_data_np, pin_x_num, pin_y_num)
        # 缺补存正pitch
        if (pitch_x_yinXian == np.array([0])).all():
            pitch_x_yinXian = pitch_x
        pitch_x = pitch_x_yinXian
        if (pitch_y_yinXian == np.array([0])).all():
            pitch_y_yinXian = pitch_y
        pitch_y = pitch_y_yinXian
        try:
            if len(pitch_x) == 0 and len(pitch_y) != 0:
                pitch_x = pitch_y
            if len(pitch_y) == 0 and len(pitch_x) != 0:
                pitch_y = pitch_x
        except:
            pass
        # 缺补存正pin_num
        if pin_num_x_serial == 0:
            pin_num_x_serial = pin_num_x
        if pin_num_y_serial == 0:
            pin_num_y_serial = pin_num_y
        pin_x_num = pin_num_x_serial
        pin_y_num = pin_num_y_serial
        # 提取high
        high = np.zeros(3)
        zer = np.zeros(3)
        if (body_x != zer).any() and (body_y != zer).any():
            high = get_high_pin_high_max_1(side_data_np, body_x, body_y)
            print("high(max,medium,min)", high)
        if (high == zer).all():
            high = get_high_pin_high_max(side_data_np)
            print("high(max,medium,min)", high)
        # 缺补存正high
        if (high == zer).all():
            high = high_yinXian
        high_yinXian = high
        high = get_absolute_high(high, side_ocr_data)

        pin_txt = r'yolox_data\pin.txt'
        pin = get_np_array_in_txt(pin_txt)

        data_set_txt = r'yolox_data\hang_or_lie_set.txt'
        data_set = get_np_array_in_txt(data_set_txt)
        # print(data_set)

        average_pitch_x_y_txt = r'yolox_data\average_x_y.txt'
        average_pitch_x_y = get_np_array_in_txt(average_pitch_x_y_txt)
        average_pitch_x = average_pitch_x_y[0][0]
        average_pitch_y = average_pitch_x_y[1][0]
        # print(average_pitch_x, average_pitch_y)

        # 提取球直径
        pin_diameter = np.zeros((0, 3))
        zer = np.zeros((0, 3))
        # 根据BGA的绝对特征找pin_diameter
        pin_diameter_absolute = get_absolute_pin_diameter(pin_diameter, top_ocr_data, bottom_ocr_data, side_ocr_data)
        pin_diameter = pin_diameter_absolute.copy()
        standoff = np.zeros((0, 3))
        if (not np.array_equal(standoff_yinXian, zer)) and np.array_equal(pin_diameter_yinXian, zer):
            pin_diameter = pin_diameter_1(pitch_x, pitch_y, pin_x_num, pin_y_num, body_x, body_y, bottom_data_np,
                                          side_data_np, top_data_np, standoff_yinXian, high)
            print("1.找到支撑高和高的情况下寻找pin直径", pin_diameter)
        if len(pin_diameter) == 0:
            pin_diameter = get_pin_diameter(pitch_x, pitch_y, pin_x_num, pin_y_num, body_x, body_y, bottom_data_np,
                                            side_data_np, top_data_np)
        if len(pin_diameter) != 0:
            print("2.找到可能是pin直径的数据(max,medium,min)\n", pin_diameter)
        else:
            print("2.算法暂未找到pin直径数据")

        if len(pin_diameter) != 0:
            if np.array_equal(pin_diameter_absolute, zer):
                standoff = find_standoff(side_data_np, pin_diameter, high)
            else:
                standoff = find_standoff(side_data_np, pin_diameter_absolute, high)
            print("1.找到可能是支撑高的数据\n", standoff)

        if len(pin_diameter) == 0:
            pin_diameter = find_pin_diameter(pin_diameter, high, top_data_np, bottom_data_np, side_data_np, pitch_x,
                                             pitch_y)
            print("3.找到可能是pin直径的数据(max,medium,min)\n", pin_diameter)
        pin_map_present = show_lost_pin_when_full(pin, pin_x_num, pin_y_num, average_pitch_x, average_pitch_y)
        if len(pin_diameter) == 0:
            '''
            side视图中寻找少数方向种类中满足大概范围的标注作为pin直径       
            '''
            pin_diameter = find_pin_diameter_last(pin_diameter, side_data_np)
        if len(pin_diameter) == 0:
            standoff = np.zeros((0, 3))
        if len(pin_diameter) != 0:
            if np.array_equal(pin_diameter_absolute, zer):
                standoff = find_standoff(side_data_np, pin_diameter, high)
            else:
                standoff = find_standoff(side_data_np, pin_diameter_absolute, high)
            print("2.找到可能是支撑高的数据\n", standoff)
        dimension = pin_diameter_absolute.ndim
        if dimension == 1:
            if not ((
                    pin_diameter_absolute != np.zeros(3)).all()):
                pin_diameter = pin_diameter_absolute
        elif dimension == 2:
            if not (np.array_equal(pin_diameter_absolute, zer)):
                pin_diameter = pin_diameter_absolute
        # 修正格式
        if pin_diameter.ndim == 1:
            zer = np.zeros((0, 3))
            pin_diameter = np.r_[zer, [pin_diameter]]
        if standoff_yinXian.ndim == 1:
            zer = np.zeros((0, 3))
            standoff_yinXian = np.r_[zer, [standoff_yinXian]]
        # 缺补存正 standoff
        if np.array_equal(standoff_yinXian, np.zeros((0, 3))) or np.array_equal(standoff_yinXian, np.zeros((1, 3))) or (
                standoff_yinXian > pin_diameter).any():
            standoff_yinXian = standoff
        standoff = standoff_yinXian
        # 当standoff不止一个时，挑选最合适的
        standoff = select_best_stanoff(standoff)
        # 将三视图中的数据展示
        print("top_data_np\n", top_data_np)
        print("bottom_data_np\n", bottom_data_np)
        print("side_data_np\n", side_data_np)
        print("############输出参数############")
        print("每行最大pin数,每列最大pin数", pin_num_x, pin_num_y)
        print("body_x, body_y(max,medium,min)", body_x, body_y)
        print("pitch_x,pitch_y", pitch_x, pitch_y)
        print("high(max,medium,min)", high)
        if len(pin_diameter) != 0:
            print("找到可能是pin直径的数据(max,medium,min)\n", pin_diameter)
        else:
            print("暂未找到pin直径数据")
        if len(pin_diameter) != 0:
            standoff = find_standoff(side_data_np, pin_diameter, high)
            print("找到可能是支撑高的数据\n", standoff)
        if len(pin_diameter) == 0:
            standoff = np.zeros((1, 3))
        try:
            pin_map_present = show_lost_pin(pin, data_set, average_pitch_x, average_pitch_y, key, pin_num_x, pin_num_y)
        except Exception as e:
            print("************报错*********\n", e)

        # return body_x, body_y, pin_x_num, pin_y_num, pitch_x, pitch_y,high, pin_diameter, standoff, pin_map_present
        return body_x, body_y, pin_x_num, pin_y_num, pitch_x, pitch_y, high, pin_diameter, standoff, pin_map_present


def get_pinmap_table():
    """输出 pinmap 表格展示。"""
    # 先取行pin数和列pin数
    pin_num_txt = r'yolox_data\pin_num.txt'
    pin_num_x_y = get_np_array_in_txt(pin_num_txt)
    pin_x_num = int(pin_num_x_y[0][0])  # 行pin数
    pin_y_num = int(pin_num_x_y[1][0])  # 列pin数
    pin_txt = r'yolox_data\pin.txt'
    pin = get_np_array_in_txt(pin_txt)
    average_pitch_x_y_txt = r'yolox_data\average_x_y.txt'
    average_pitch_x_y = get_np_array_in_txt(average_pitch_x_y_txt)
    average_pitch_x = average_pitch_x_y[0][0]
    average_pitch_y = average_pitch_x_y[1][0]
    pin_map_present = show_lost_pin_when_full(pin, pin_x_num, pin_y_num, average_pitch_x, average_pitch_y)
    return pin_map_present


def tf(pairs_data):
    """以可视化方式展示 top 视图的匹配结果。"""
    data_np = np.zeros((0, 5))
    data_np_arr = np.zeros(5)
    for i in range(len(pairs_data)):
        if (pairs_data[i][2] - pairs_data[i][0]) > (pairs_data[i][3] - pairs_data[i][1]):
            data_np_arr[0] = 0
        if (pairs_data[i][2] - pairs_data[i][0]) < (pairs_data[i][3] - pairs_data[i][1]):
            data_np_arr[0] = 1
        if pairs_data[i][0] == pairs_data[i][1] == 0:
            data_np_arr[0] = 0.5
        data_np_arr[1] = 0
        data_np_arr[2:] = pairs_data[i, -3:]
        data_np = np.r_[data_np, [data_np_arr]]
    return data_np


def tfbottom(pairs_data):
    """以可视化方式展示 bottom 视图的匹配结果。"""
    data_np = np.zeros((0, 9))
    data_np_arr = np.zeros(9)
    for i in range(len(pairs_data)):
        if (pairs_data[i][2] - pairs_data[i][0]) > (pairs_data[i][3] - pairs_data[i][1]):
            data_np_arr[0] = 0
        if (pairs_data[i][2] - pairs_data[i][0]) < (pairs_data[i][3] - pairs_data[i][1]):
            data_np_arr[0] = 1
        if pairs_data[i][0] == pairs_data[i][1] == 0:
            data_np_arr[0] = 0.5
        data_np_arr[1] = 0
        data_np_arr[2:6] = pairs_data[i][0:4]
        data_np_arr[6:] = pairs_data[i, -3:]
        data_np = np.r_[data_np, [data_np_arr]]
    return data_np

def extract_BGA_PIN():
    """运行 BGA PIN 提取流程的入口。"""
    print("开始提取BGA的PIN")
    package_classes = 'BGA'
    package_path = page_path
    img_path = f'{package_path}/bottom.jpg'
    # 如果目标文件存在，则先删除
    if os.path.exists(f'{DATA}/bottom.jpg'):
        os.remove(f'{DATA}/bottom.jpg')

    # 移动文件
    shutil.copy2(img_path, f'{DATA}/bottom.jpg')

    bottom_dbnet_data = dbnet_get_text_box(img_path)
    bottom_yolox_pairs, bottom_yolox_num, bottom_yolox_serial_num, bottom_pin, bottom_other, bottom_pad, bottom_border, bottom_angle_pairs, bottom_BGA_serial_num, bottom_BGA_serial_letter = yolo_classify(
        img_path, package_classes)

    pin_map, color = time_save_find_pinmap(bottom_border)
    print("pin_map", pin_map)

    # pin_map的行数为列pin数， 列数为行pin数
    rows, cols = pin_map.shape


    serial_letters = bottom_BGA_serial_letter
    serial_numbers = bottom_BGA_serial_num
    bottom_dbnet_data_serial = bottom_dbnet_data.copy()
    serial_numbers_data, serial_letters_data, bottom_dbnet_data = find_serial_number_letter(serial_numbers,
                                                                                            serial_letters,
                                                                                            bottom_dbnet_data)
    path = f'{DATA}/bottom.jpg'
    bottom_ocr_data = ocr_data(path, bottom_dbnet_data_serial)
    serial_numbers_data, serial_letters_data, bottom_ocr_data = filter_bottom_ocr_data(bottom_ocr_data,
                                                                                       bottom_dbnet_data_serial,
                                                                                       serial_numbers_data,
                                                                                       serial_letters_data,
                                                                                       bottom_dbnet_data)
    pin_num_x_serial, pin_num_y_serial, pin_1_location = find_pin_num_pin_1(serial_numbers_data,
                                                                            serial_letters_data,
                                                                            serial_numbers, serial_letters)
    try:
        from packagefiles.PackageExtract.BGA_extract_old import Is_Loss_Pin
        loss_pin, loss_color = Is_Loss_Pin(pin_map, pin_1_location, color)
    except:
        loss_pin = []
        loss_color = []
    if len(loss_pin) == 0:
        loss_pin1 = 'None'
    else:
        loss_pin1 = loss_pin

    pin_num_x_serial = int(pin_num_x_serial)
    pin_num_y_serial = int(pin_num_y_serial)

    if pin_num_x_serial == 0:
        pin_num_x_serial = rows
    if pin_num_y_serial == 0:
        pin_num_y_serial = cols
    return pin_num_x_serial, pin_num_y_serial, loss_pin1, loss_color

# if __name__ == "__main__":