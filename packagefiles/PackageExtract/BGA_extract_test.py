import os
import queue
import threading

import numpy as np
from BGA_cal_pin import find_pin
# 外部文件：
from packagefiles.PackageExtract.common_pipeline import (
    get_data_location_by_yolo_dbnet,
    prepare_workspace,
)
from packagefiles.PackageExtract.function_tool import *
from packagefiles.PackageExtract.get_pairs_data_present5_test import *
#全局路径
DATA = 'Result/Package_extract/data'
DATA_BOTTOM_CROP = 'Result/Package_extract/data_bottom_crop'
DATA_COPY = 'Result/Package_extract/data_copy'
ONNX_OUTPUT = 'Result/Package_extract/onnx_output'
OPENCV_OUTPUT = 'Result/Package_extract/opencv_output'
OPENCV_OUTPUT_LINE = 'Result/Package_extract/opencv_output_yinXian'


def extract_package(package_classes):
    # 完成图片大小固定、清空建立文件夹等各种操作
    prepare_workspace(
        DATA,
        DATA_COPY,
        DATA_BOTTOM_CROP,
        ONNX_OUTPUT,
        OPENCV_OUTPUT,
    )
    test_mode = 0  # 0: 正常模式，1: 测试模式
    key = test_mode
    '''
        默认图片型封装
    '''
    letter_or_number = 'number'
    '''
    YOLO检测
    DBnet检测
    SVTR识别
    数据整理
    输出参数
    '''
    # BGA独有流程：获取pinmap，找到缺pin位置信息
    pin_map = time_save_find_pinmap()
    pin_output = 1
    # (1)在各个视图中用yolox识别图像元素LOCATION，dbnet识别文本location
    L3 = get_data_location_by_yolo_dbnet(DATA, package_classes)

    # (2)在yolo和dbnet的标注文本框中去除OTHER类型文本框
    L3 = data_delete_other(L3)

    # (3)为尺寸线寻找尺寸界限
    L3 = for_pairs_find_lines(L3, key)

    # 处理数据
    L3 = resize_data_1(L3, key)

    # (4)SVTR识别标注内容
    L3 = SVTR_get_data(L3)

    # (5)SVTR后处理数据
    L3 = get_max_medium_min(L3, key)

    # (6)提取并分离出yolo和dbnet检测出的标注中的序号
    L3 = get_Pin_data(L3, package_classes)

    # (7)匹配pairs和data
    L3 = MPD_data(L3, key)

    # 处理数据
    L3 = resize_data_2(L3)

    '''
        输出QFP参数
        nx,ny
        pitch
        high(A)
        standoff(A1)
        span_x,span_y
        body_x,body_y
        b
        pad_x,pad_y
    '''
    # 语义对齐

    # QFP_parameter_list, nx, ny = find_QFP_parameter(L3)
    # # 整理获得的参数
    # parameter_list = get_QFP_parameter_data(QFP_parameter_list, nx, ny)
    # print("修改前的参数列表", parameter_list)
    # # 参数检查与修改
    # parameter_list = alter_QFP_parameter_data(parameter_list)
    # print("修改后的参数列表", parameter_list)
    return parameter_list

def time_save_find_pinmap():
    result_queue = queue.Queue()
    thread = threading.Thread(target=long_running_task, args=(result_queue,))
    thread.start()

    thread.join(timeout=6)  # 设置超时时间为5秒
    if thread.is_alive():
        print("读取pinmap进程花费时间过长，跳过")
        pin_map = np.ones((10, 10))
        # 记录pin的行列数
        pin_num_x_y = np.array([0, 0])
        pin_num_x_y = pin_num_x_y.astype(int)
        path = r'yolox_data\pin_num.txt'
        np.savetxt(path, pin_num_x_y)
    else:
        try:
            pin_map = result_queue.get_nowait()  # 尝试获取结果
            # print("Result:", pin_map)
        except queue.Empty:
            print("Queue is empty, no result available.")
            pin_map = np.ones((10, 10))
            # 记录pin的行列数
            pin_num_x_y = np.array([0, 0])
            pin_num_x_y = pin_num_x_y.astype(int)
            path = r'yolox_data\pin_num.txt'
            np.savetxt(path, pin_num_x_y)
    # 记录pin的行列数
    pin_num_x_y = np.array([pin_map.shape[1], pin_map.shape[0]])
    pin_num_x_y = pin_num_x_y.astype(int)
    path = r'yolox_data\pin_num.txt'
    np.savetxt(path, pin_num_x_y)

    return pin_map

def long_running_task(result_queue):
    print()
    print("***/开始检测pin/***")
    result = find_pin()
    # try:
    #     result = find_pin()
    # except:
    #     print("pinmap没有正常读取，请记录pdf并反馈")
    #     result = np.ones((10, 10))
    result_queue.put(result)
    print("***/结束检测pin/***")
    print()


def data_delete_other(L3):
    """
    在yolo和dbnet的标注文本框中去除OTHER类型文本框
    :param L3:包含所有图像元素和文本的坐标范围集合(外框)
    :return: L3:包含所有图像元素和文本的坐标范围集合(外框)
    """
    top_yolox_num = find_list(L3, 'top_yolox_num')
    top_dbnet_data = find_list(L3, 'top_dbnet_data')
    top_other = find_list(L3, 'top_other')

    bottom_yolox_num = find_list(L3, 'bottom_yolox_num')
    bottom_dbnet_data = find_list(L3, 'bottom_dbnet_data')
    bottom_other = find_list(L3, 'bottom_other')

    side_yolox_num = find_list(L3, 'side_yolox_num')
    side_dbnet_data = find_list(L3, 'side_dbnet_data')
    side_other = find_list(L3, 'side_other')

    detailed_yolox_num = find_list(L3, 'detailed_yolox_num')
    detailed_dbnet_data = find_list(L3, 'detailed_dbnet_data')
    detailed_other = find_list(L3, 'detailed_other')

    top_yolox_num = delete_other(top_other, top_yolox_num)
    top_dbnet_data = delete_other(top_other, top_dbnet_data)

    bottom_yolox_num = delete_other(bottom_other, bottom_yolox_num)
    bottom_dbnet_data = delete_other(bottom_other, bottom_dbnet_data)

    side_yolox_num = delete_other(side_other, side_yolox_num)
    side_dbnet_data = delete_other(side_other, side_dbnet_data)

    detailed_yolox_num = delete_other(detailed_other, detailed_yolox_num)
    detailed_dbnet_data = delete_other(detailed_other, detailed_dbnet_data)

    recite_data(L3, 'top_yolox_num', top_yolox_num)
    recite_data(L3, 'top_dbnet_data', top_dbnet_data)
    recite_data(L3, 'bottom_yolox_num', bottom_yolox_num)
    recite_data(L3, 'bottom_dbnet_data', bottom_dbnet_data)
    recite_data(L3, 'side_yolox_num', side_yolox_num)
    recite_data(L3, 'side_dbnet_data', side_dbnet_data)
    recite_data(L3, 'detailed_yolox_num', detailed_yolox_num)
    recite_data(L3, 'detailed_dbnet_data', detailed_dbnet_data)

    return L3




def for_pairs_find_lines(L3, test_mode):
    """
    为尺寸线寻找尺寸界限
    :param L3:
    :param test_mode:
    :return:
    """
    top_yolox_pairs = find_list(L3, 'top_yolox_pairs')
    bottom_yolox_pairs = find_list(L3, 'bottom_yolox_pairs')
    side_yolox_pairs = find_list(L3, 'side_yolox_pairs')
    detailed_yolox_pairs = find_list(L3, 'detailed_yolox_pairs')
    empty_data = np.empty((0, 13))
    img_path = f'{DATA}/top.jpg'
    if not os.path.exists(img_path):
        top_yolox_pairs_length = empty_data
    else:
        top_yolox_pairs_length = find_pairs_length(img_path, top_yolox_pairs, test_mode)
        # top_yolox_pairs_length np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
    img_path = f'{DATA}/bottom.jpg'
    if not os.path.exists(img_path):
        bottom_yolox_pairs_length = empty_data
    else:
        bottom_yolox_pairs_length = find_pairs_length(img_path, bottom_yolox_pairs, test_mode)
    img_path = f'{DATA}/side.jpg'
    if not os.path.exists(img_path):
        side_yolox_pairs_length = empty_data
    else:
        side_yolox_pairs_length = find_pairs_length(img_path, side_yolox_pairs, test_mode)
    img_path = f'{DATA}/detailed.jpg'
    if not os.path.exists(img_path):
        detailed_yolox_pairs_length = empty_data
    else:
        detailed_yolox_pairs_length = find_pairs_length(img_path, detailed_yolox_pairs, test_mode)

    recite_data(L3, 'top_yolox_pairs_length', top_yolox_pairs_length)
    recite_data(L3, 'bottom_yolox_pairs_length', bottom_yolox_pairs_length)
    recite_data(L3, 'side_yolox_pairs_length', side_yolox_pairs_length)
    recite_data(L3, 'detailed_yolox_pairs_length', detailed_yolox_pairs_length)
    return L3


def resize_data_1(L3, key):
    """
    处理数据
    :param L3:
    :param key:
    :return:
    """
    top_yolox_pairs = find_list(L3, 'top_yolox_pairs')
    top_dbnet_data = find_list(L3, 'top_dbnet_data')
    bottom_yolox_pairs = find_list(L3, 'bottom_yolox_pairs')
    bottom_dbnet_data = find_list(L3, 'bottom_dbnet_data')
    side_yolox_pairs = find_list(L3, 'side_yolox_pairs')
    side_dbnet_data = find_list(L3, 'side_dbnet_data')
    detailed_yolox_pairs = find_list(L3, 'detailed_yolox_pairs')
    detailed_dbnet_data = find_list(L3, 'detailed_dbnet_data')

    top_yolox_pairs, bottom_yolox_pairs, side_yolox_pairs, detailed_yolox_pairs, top_yolox_pairs_copy, bottom_yolox_pairs_copy, side_yolox_pairs_copy, detailed_yolox_pairs_copy, top_dbnet_data_all, bottom_dbnet_data_all \
        = get_better_data_1(top_yolox_pairs, bottom_yolox_pairs, side_yolox_pairs, detailed_yolox_pairs, key,
                            top_dbnet_data, bottom_dbnet_data, side_dbnet_data, detailed_dbnet_data)

    recite_data(L3, 'top_yolox_pairs', top_yolox_pairs)
    recite_data(L3, 'top_dbnet_data', top_dbnet_data)
    recite_data(L3, 'bottom_yolox_pairs', bottom_yolox_pairs)
    recite_data(L3, 'bottom_dbnet_data', bottom_dbnet_data)
    recite_data(L3, 'side_yolox_pairs', side_yolox_pairs)
    recite_data(L3, 'side_dbnet_data', side_dbnet_data)
    recite_data(L3, 'detailed_yolox_pairs', detailed_yolox_pairs)
    recite_data(L3, 'detailed_dbnet_data', detailed_dbnet_data)
    recite_data(L3, 'top_yolox_pairs_copy', top_yolox_pairs_copy)
    recite_data(L3, 'bottom_yolox_pairs_copy', bottom_yolox_pairs_copy)
    recite_data(L3, 'side_yolox_pairs_copy', side_yolox_pairs_copy)
    recite_data(L3, 'detailed_yolox_pairs_copy', detailed_yolox_pairs_copy)
    recite_data(L3, 'top_dbnet_data_all', top_dbnet_data_all)
    recite_data(L3, 'bottom_dbnet_data_all', bottom_dbnet_data_all)

    return L3



def SVTR_get_data(L3):
    """

    :param L3:
    :return:
    """
    top_dbnet_data_all = find_list(L3, 'top_dbnet_data_all')
    bottom_dbnet_data_all = find_list(L3, 'bottom_dbnet_data_all')
    side_dbnet_data = find_list(L3, 'side_dbnet_data')
    detailed_dbnet_data = find_list(L3, 'detailed_dbnet_data')

    start, end, top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data = SVTR(top_dbnet_data_all,
                                                                                        bottom_dbnet_data_all,
                                                                                        side_dbnet_data,
                                                                                        detailed_dbnet_data)
    recite_data(L3, 'top_ocr_data', top_ocr_data)
    recite_data(L3, 'bottom_ocr_data', bottom_ocr_data)
    recite_data(L3, 'side_ocr_data', side_ocr_data)
    recite_data(L3, 'detailed_ocr_data', detailed_ocr_data)

    return L3



def get_max_medium_min(L3, key):
    """

    :param L3:
    :return:
    """
    top_dbnet_data = find_list(L3, 'top_dbnet_data')
    bottom_dbnet_data = find_list(L3, 'bottom_dbnet_data')
    side_dbnet_data = find_list(L3, 'side_dbnet_data')
    detailed_dbnet_data = find_list(L3, 'detailed_dbnet_data')
    top_ocr_data = find_list(L3, 'top_ocr_data')
    bottom_ocr_data = find_list(L3, 'bottom_ocr_data')
    side_ocr_data = find_list(L3, 'side_ocr_data')
    detailed_ocr_data = find_list(L3, 'detailed_ocr_data')
    top_yolox_num = find_list(L3, 'top_yolox_num')
    bottom_yolox_num = find_list(L3, 'bottom_yolox_num')
    side_yolox_num = find_list(L3, 'side_yolox_num')
    detailed_yolox_num = find_list(L3, 'detailed_yolox_num')

    top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data = data_wrangling(key, top_dbnet_data,
                                                                                     bottom_dbnet_data,
                                                                                     side_dbnet_data,
                                                                                     detailed_dbnet_data,
                                                                                     top_ocr_data,
                                                                                     bottom_ocr_data,
                                                                                     side_ocr_data,
                                                                                     detailed_ocr_data,
                                                                                     top_yolox_num,
                                                                                     bottom_yolox_num,
                                                                                     side_yolox_num,
                                                                                     detailed_yolox_num)
    recite_data(L3, 'top_ocr_data', top_ocr_data)
    recite_data(L3, 'bottom_ocr_data', bottom_ocr_data)
    recite_data(L3, 'side_ocr_data', side_ocr_data)
    recite_data(L3, 'detailed_ocr_data', detailed_ocr_data)

    return L3


def get_Pin_data(L3):
    top_yolox_serial_num = find_list(L3, 'top_yolox_serial_num')
    bottom_yolox_serial_num = find_list(L3, 'bottom_yolox_serial_num')
    top_ocr_data = find_list(L3, 'top_ocr_data')
    bottom_ocr_data = find_list(L3, 'bottom_ocr_data')

    top_serial_numbers_data, bottom_serial_numbers_data, top_ocr_data, bottom_ocr_data = find_PIN(top_yolox_serial_num,
                                                                   bottom_yolox_serial_num, top_ocr_data,
                                                                   bottom_ocr_data)

    recite_data(L3, 'top_serial_numbers_data', top_serial_numbers_data)
    recite_data(L3, 'bottom_serial_numbers_data', bottom_serial_numbers_data)
    recite_data(L3, 'top_ocr_data', top_ocr_data)
    recite_data(L3, 'bottom_ocr_data', bottom_ocr_data)
    return L3

def MPD_data(L3, key):
    # 从L3中获取数据
    top_yolox_pairs = find_list(L3, 'top_yolox_pairs')
    bottom_yolox_pairs = find_list(L3, 'bottom_yolox_pairs')
    side_yolox_pairs = find_list(L3, 'side_yolox_pairs')
    detailed_yolox_pairs = find_list(L3, 'detailed_yolox_pairs')
    side_angle_pairs = find_list(L3, 'side_angle_pairs')
    detailed_angle_pairs = find_list(L3, 'detailed_angle_pairs')
    top_border = find_list(L3, 'top_border')
    bottom_border = find_list(L3, 'bottom_border')
    top_ocr_data = find_list(L3, 'top_ocr_data')
    bottom_ocr_data = find_list(L3, 'bottom_ocr_data')
    side_ocr_data = find_list(L3, 'side_ocr_data')
    detailed_ocr_data = find_list(L3, 'detailed_ocr_data')
    top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data = MPD(key, top_yolox_pairs,
                                                                          bottom_yolox_pairs, side_yolox_pairs,
                                                                          detailed_yolox_pairs,
                                                                          side_angle_pairs,
                                                                          detailed_angle_pairs, top_border,
                                                                          bottom_border, top_ocr_data,
                                                                          bottom_ocr_data, side_ocr_data,
                                                                          detailed_ocr_data)
    recite_data(L3, 'top_ocr_data', top_ocr_data)
    recite_data(L3, 'bottom_ocr_data', bottom_ocr_data)
    recite_data(L3, 'side_ocr_data', side_ocr_data)
    recite_data(L3, 'detailed_ocr_data', detailed_ocr_data)
    return L3


def resize_data_2(L3):


    top_ocr_data = find_list(L3, 'top_ocr_data')
    bottom_ocr_data = find_list(L3, 'bottom_ocr_data')
    side_ocr_data = find_list(L3, 'side_ocr_data')
    detailed_ocr_data = find_list(L3, 'detailed_ocr_data')
    top_yolox_pairs_length = find_list(L3, 'top_yolox_pairs_length')
    bottom_yolox_pairs_length = find_list(L3, 'bottom_yolox_pairs_length')
    side_yolox_pairs_length = find_list(L3, 'side_yolox_pairs_length')
    detailed_yolox_pairs_length = find_list(L3, 'detailed_yolox_pairs_length')
    top_yolox_pairs_copy = find_list(L3, 'top_yolox_pairs_copy')
    bottom_yolox_pairs_copy = find_list(L3, 'bottom_yolox_pairs_copy')
    side_yolox_pairs_copy = find_list(L3, 'side_yolox_pairs_copy')
    detailed_yolox_pairs_copy = find_list(L3, 'detailed_yolox_pairs_copy')




    top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data, yolox_pairs_top, yolox_pairs_bottom, yolox_pairs_side, yolox_pairs_detailed = get_better_data_2(
        top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data, top_yolox_pairs_length,
        bottom_yolox_pairs_length, side_yolox_pairs_length, detailed_yolox_pairs_length, top_yolox_pairs_copy,
        bottom_yolox_pairs_copy, side_yolox_pairs_copy, detailed_yolox_pairs_copy)

    recite_data(L3, 'top_ocr_data', top_ocr_data)
    recite_data(L3, 'bottom_ocr_data', bottom_ocr_data)
    recite_data(L3, 'side_ocr_data', side_ocr_data)
    recite_data(L3, 'detailed_ocr_data', detailed_ocr_data)
    recite_data(L3, 'yolox_pairs_top', yolox_pairs_top)
    recite_data(L3, 'yolox_pairs_bottom', yolox_pairs_bottom)
    recite_data(L3, 'yolox_pairs_side', yolox_pairs_side)
    recite_data(L3, 'yolox_pairs_detailed', yolox_pairs_detailed)
    # 总结
    print("***/数据整理结果/***")
    print("top视图数据整理结果:\n", *top_ocr_data, sep='\n')
    print("bottom视图数据整理结果:\n", *bottom_ocr_data, sep='\n')
    print("side视图数据整理结果:\n", *side_ocr_data, sep='\n')
    print("detailed视图数据整理结果:\n", *detailed_ocr_data, sep='\n')
    # print("top视图中的PIN,pad,Border:\n", top_pin, top_pad, top_border)
    # print("bottom视图中的PIN,pad,Border:\n", bottom_pin, bottom_pad, bottom_border)
    # print("side视图中的PIN,pad,Border:\n", side_pin, side_pad, side_border)
    # print("detailed视图中的PIN,pad,Border:\n", detailed_pin, detailed_pad, detailed_border)
    return L3

def find_QFP_parameter(L3):
    top_serial_numbers_data = find_list(L3, 'top_serial_numbers_data')
    bottom_serial_numbers_data = find_list(L3, 'bottom_serial_numbers_data')
    top_ocr_data = find_list(L3, 'top_ocr_data')
    bottom_ocr_data = find_list(L3, 'bottom_ocr_data')
    side_ocr_data = find_list(L3, 'side_ocr_data')
    detailed_ocr_data = find_list(L3, 'detailed_ocr_data')
    yolox_pairs_top = find_list(L3, 'yolox_pairs_top')
    yolox_pairs_bottom = find_list(L3, 'yolox_pairs_bottom')
    top_yolox_pairs_length = find_list(L3, 'top_yolox_pairs_length')
    bottom_yolox_pairs_length = find_list(L3, 'bottom_yolox_pairs_length')
    top_border = find_list(L3, 'top_border')
    bottom_border = find_list(L3, 'bottom_border')



    # (9)输出序号nx,ny和body_x、body_y
    nx, ny = get_serial(top_serial_numbers_data, bottom_serial_numbers_data)
    body_x, body_y = get_QFP_body(yolox_pairs_top, top_yolox_pairs_length, yolox_pairs_bottom,
                                  bottom_yolox_pairs_length, top_border, bottom_border, top_ocr_data,
                                  bottom_ocr_data)
    get_QFP_body(yolox_pairs_top, top_yolox_pairs_length, yolox_pairs_bottom,
                                  bottom_yolox_pairs_length, top_border, bottom_border, top_ocr_data,
                                  bottom_ocr_data)
    # (10)初始化参数列表
    QFP_parameter_list = get_QFP_parameter_list(top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data,
                                                body_x, body_y)
    # (11)整理参数列表
    QFP_parameter_list = resort_parameter_list_2(QFP_parameter_list)
    # 输出高

    if len(QFP_parameter_list[4]) > 1:
        high = get_QFP_high(QFP_parameter_list[4]['maybe_data'])
        if len(high) > 0:
            QFP_parameter_list[4]['maybe_data'] = high
            QFP_parameter_list[4]['maybe_data_num'] = len(high)
    # 输出pitch
    if len(QFP_parameter_list[5]['maybe_data']) > 1 or len(QFP_parameter_list[6]['maybe_data']) > 1:
        pitch_x, pitch_y = get_QFP_pitch(QFP_parameter_list[5]['maybe_data'], body_x, body_y, nx, ny)
        if len(pitch_x) > 0:
            QFP_parameter_list[5]['maybe_data'] = pitch_x
            QFP_parameter_list[5]['maybe_data_num'] = len(pitch_x)
        if len(pitch_y) > 0:
            QFP_parameter_list[6]['maybe_data'] = pitch_y
            QFP_parameter_list[6]['maybe_data_num'] = len(pitch_y)
    # 整理参数列表
    QFP_parameter_list = resort_parameter_list_2(QFP_parameter_list)
    # # 补全相同参数的x、y
    # QFP_parameter_list = Completion_QFP_parameter_list(QFP_parameter_list)
    # # 输出参数列表，给出置信度
    # QFP = output_QFP_parameter(QFP_parameter_list, nx, ny)
    return QFP_parameter_list, nx, ny
# if __name__ == '__main__':
#     extract_package(package_classes='QFP')