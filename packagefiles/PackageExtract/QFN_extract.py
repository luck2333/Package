import os

# 外部文件：
from packagefiles.PackageExtract.function_tool import *
from packagefiles.PackageExtract.get_pairs_data_present5_test import *
from packagefiles.PackageExtract.onnx_use import Run_onnx_det, Run_onnx
from packagefiles.PackageExtract.yolox_onnx_py.onnx_QFP_pairs_data_location2 import begain_output_pairs_data_location
from packagefiles.PackageExtract.QFP_extract import *
#全局路径
DATA = 'Result/Package_extract/data'
DATA_BOTTOM_CROP = 'Result/Package_extract/data_bottom_crop'
DATA_COPY = 'Result/Package_extract/data_copy'
ONNX_OUTPUT = 'Result/Package_extract/onnx_output'
OPENCV_OUTPUT = 'Result/Package_extract/opencv_output'
OPENCV_OUTPUT_LINE = 'Result/Package_extract/opencv_output_yinXian'

def extract_QFN(package_classes, page_num):
    # 完成图片大小固定、清空建立文件夹等各种操作
    front_loading_work()
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
    
    '''
    # 语义对齐
    QFN_parameter_list, nx, ny = find_QFN_parameter(L3)
    # 20250722添加
    # 指定要查找的 page_num
    target_page_num = page_num
    json_file = 'output.json'
    result = []
    # 读取 JSON 文件
    print('正在读取JSON文件...')
    # with open(json_file, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    with open(json_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content:
            try:
                data = json.loads(content)
                print("解析成功")
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
                if abs(result[4] * result[5] - result[0]) < 1e-9 and abs(nx * ny - result[4] * result[5]) > 1e-9:
                    nx = result[4]
                    ny = result[5]
                if nx == 0 and result[4] != None:
                    nx = result[4]
                if ny == 0 and result[5] != None:
                    ny = result[5]
    # # 整理获得的参数
    parameter_list = get_QFN_parameter_data(QFN_parameter_list, nx, ny)
    try:
        length = float(parameter_list[0][2])
    except:
        print("无法转化为浮点数length", parameter_list[0][2])
    try:
        weight = float(parameter_list[1][2])
    except:
        print("无法转化为浮点数weight", parameter_list[1][2])
    try:
        height = float(parameter_list[2][2])
    except:
        print("无法转化为浮点数height", parameter_list[2][2])
    if result != []:
        if result[1] != None and result[1] != length and (result[1] != weight and result[2] != length):
            parameter_list[0][1] = ''
            parameter_list[0][2] = result[1]
            parameter_list[0][3] = ''
        if result[2] != None and result[2] != weight and (result[1] != weight and result[2] != length):
            parameter_list[1][1] = ''
            parameter_list[1][2] = result[2]
            parameter_list[1][3] = ''
        if result[3] != None and result[3] != height:
            parameter_list[2][1] = ''
            parameter_list[2][2] = result[3]
            parameter_list[2][3] = ''
    # #20250723改变顺序
    # QFN_TABLE = ['实体长D', '实体宽E', '实体高A', '支撑高A1', '端子高A3', 'PIN长L', 'PIN宽b', '行PIN数', '列PIN数', '行/列PIN间距e',
    #              '散热盘长D2', '散热盘宽E2', '削角否', '削角长度', '端子圆角否']
    # QFP_TABLE = ['实体长D1', '实体宽E1', '实体高A', '支撑高A1', '端子高A3', '外围长D', '外围宽E', 'PIN长L', 'PIN宽b', '行PIN数',
    #              '列PIN数', '行/列PIN间距e', '散热盘长D2', '散热盘宽E2', '削角长度', '端子厚度', '接触角度', '端腿角度', '主体顶部绘制角度']
    # SON_TABLE = ['实体长D', '实体宽E', '实体高A', 'PIN长L', 'PIN宽b', 'PIN行数', 'PIN列数', '行PIN数', '列PIN数',
    #              'PIN_Pitche', 'PIN端距', 'layout建议值']
    # SOP_TABLE = ['实体长D', '实体宽E', '实体高A', 'PIN长L', 'PIN宽b', 'PIN行数', 'PIN列数', '行PIN数', '列PIN数',
    #              'PIN_Pitche', 'PIN端距', 'layout建议值']
    new_parameter_list = []
    new_parameter_list.append(parameter_list[9])
    new_parameter_list.append(parameter_list[9])
    new_parameter_list.append(parameter_list[7])
    new_parameter_list.append(parameter_list[8])
    new_parameter_list.append(parameter_list[2])
    new_parameter_list.append(parameter_list[3])
    new_parameter_list.append([0,'-','-','-'])
    new_parameter_list.append(parameter_list[0])
    new_parameter_list.append(parameter_list[1])
    new_parameter_list.append([0, '-', '-', '-'])
    new_parameter_list.append(parameter_list[5])
    new_parameter_list.append(parameter_list[6])
    new_parameter_list.append([0, '-', '-', '-'])
    new_parameter_list.append([0, '-', '-', '-'])
    new_parameter_list.append(parameter_list[10])
    new_parameter_list.append(parameter_list[11])
    print(new_parameter_list)
    return new_parameter_list

def find_QFN_parameter(L3):
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
    # (10)初始化参数列表
    QFP_parameter_list = get_QFN_parameter_list(top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data,
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


def get_QFN_parameter_list(top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data, body_x, body_y):
    '''
    D/E 10~35
    D1/E1
    D2/E2
    A
    A1
    e
    b
    θ
    L
    c:引脚厚度
    '''
    QFN_parameter_list = []
    dic = {'parameter_name': [], 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_D = {'parameter_name': 'D', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_E = {'parameter_name': 'E', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    # D_max = 35
    # D_min = 8.75
    # E_max = D_max
    # E_min = D_min

    dic_D1 = {'parameter_name': 'D1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_E1 = {'parameter_name': 'E1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    D1_max = 30
    D1_min = 6.9
    E1_max = D1_max
    E1_min = D1_min

    dic_L = {'parameter_name': 'L', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    L_max = 0.75
    L_min = 0.45


    dic_GAGE_PLANE = {'parameter_name': 'GAGE_PLANE', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    GAGE_PLANE_max = 0.25
    GAGE_PLANE_min = 0.25

    dic_c = {'parameter_name': 'c', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    c_max = 0.20
    c_min = 0.09

    dic_θ = {'parameter_name': 'θ', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    θ_max = 10
    θ_min = 0

    dic_θ1 = {'parameter_name': 'θ1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    θ1_max = 14
    θ1_min = 0

    dic_θ2 = {'parameter_name': 'θ2', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    θ2_max = 16
    θ2_min = 11

    dic_θ3 = {'parameter_name': 'θ3', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    θ3_max = 16
    θ3_min = 11

    # dic_D2 = {'parameter_name': 'D2', 'maybe_data': [], 'possible': []}
    # dic_E2 = {'parameter_name': 'E2', 'maybe_data': [], 'possible': []}

    dic_A = {'parameter_name': 'A', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    A_max = 4.5
    A_min = 1.1
    dic_A1 = {'parameter_name': 'A1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    A1_max = 0.3
    A1_min = 0
    dic_e = {'parameter_name': 'e', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    e_max = 1.3
    e_min = 0.35
    dic_b = {'parameter_name': 'b', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    b_max = 0.83
    b_min = 0.13
    dic_D2 = {'parameter_name': 'D2', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_E2 = {'parameter_name': 'E2', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    D2_max = 7.2
    D2_min = 3.15
    E2_max = 7.2
    E2_min = 3.15
    QFN_parameter_list.append(dic_D)
    QFN_parameter_list.append(dic_E)
    QFN_parameter_list.append(dic_D1)
    QFN_parameter_list.append(dic_E1)
    # QFN_parameter_list.append(dic_D2)
    # QFN_parameter_list.append(dic_E2)
    QFN_parameter_list.append(dic_A)
    QFN_parameter_list.append(dic_A1)
    QFN_parameter_list.append(dic_e)
    QFN_parameter_list.append(dic_b)
    QFN_parameter_list.append(dic_D2)
    QFN_parameter_list.append(dic_E2)
    QFN_parameter_list.append(dic_L)
    QFN_parameter_list.append(dic_GAGE_PLANE)
    QFN_parameter_list.append(dic_c)
    QFN_parameter_list.append(dic_θ)
    QFN_parameter_list.append(dic_θ1)
    QFN_parameter_list.append(dic_θ2)
    QFN_parameter_list.append(dic_θ3)

    for i in range(len(top_ocr_data)):
        # if D_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= D_max:
        #     QFN_parameter_list[0]['maybe_data'].append(top_ocr_data[i])
        #     QFN_parameter_list[0]['maybe_data_num'] += 1
        # if E_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= E_max:
        #     QFN_parameter_list[1]['maybe_data'].append(top_ocr_data[i])
        #     QFN_parameter_list[1]['maybe_data_num'] += 1
        if D1_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= D1_max:
            if len(body_x) > 0:
                QFN_parameter_list[2]['maybe_data'] = body_x
            else:
                QFN_parameter_list[2]['maybe_data'].append(top_ocr_data[i])
                QFN_parameter_list[2]['maybe_data_num'] += 1
        if E1_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= E1_max:
            if len(body_y) > 0:
                QFN_parameter_list[3]['maybe_data'] = body_y
            else:
                QFN_parameter_list[3]['maybe_data'].append(top_ocr_data[i])
                QFN_parameter_list[3]['maybe_data_num'] += 1
        if e_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= e_max:
            QFN_parameter_list[6]['maybe_data'].append(top_ocr_data[i])
            QFN_parameter_list[6]['maybe_data_num'] += 1
        if b_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= b_max:
            QFN_parameter_list[7]['maybe_data'].append(top_ocr_data[i])
            QFN_parameter_list[7]['maybe_data_num'] += 1
        if D2_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= D2_max:
            QFN_parameter_list[8]['maybe_data'].append(top_ocr_data[i])
            QFN_parameter_list[8]['maybe_data_num'] += 1
        if E2_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= E2_max:
            QFN_parameter_list[9]['maybe_data'].append(top_ocr_data[i])
            QFN_parameter_list[9]['maybe_data_num'] += 1
    for i in range(len(bottom_ocr_data)):
        # if D_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= D_max:
        #     QFN_parameter_list[0]['maybe_data'].append(bottom_ocr_data[i])
        #     QFN_parameter_list[0]['maybe_data_num'] += 1
        # if E_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= E_max:
        #     QFN_parameter_list[1]['maybe_data'].append(bottom_ocr_data[i])
        #     QFN_parameter_list[1]['maybe_data_num'] += 1
        if D1_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= D1_max:
            if len(body_x) > 0:
                QFN_parameter_list[2]['maybe_data'] = body_x
            else:
                QFN_parameter_list[2]['maybe_data'].append(bottom_ocr_data[i])
                QFN_parameter_list[2]['maybe_data_num'] += 1
        if E1_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= E1_max:
            if len(body_y) > 0:
                QFN_parameter_list[3]['maybe_data'] = body_y
            else:
                QFN_parameter_list[3]['maybe_data'].append(bottom_ocr_data[i])
                QFN_parameter_list[3]['maybe_data_num'] += 1
        if e_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= e_max:
            QFN_parameter_list[6]['maybe_data'].append(bottom_ocr_data[i])
            QFN_parameter_list[6]['maybe_data_num'] += 1
        if b_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= b_max:
            QFN_parameter_list[7]['maybe_data'].append(bottom_ocr_data[i])
            QFN_parameter_list[7]['maybe_data_num'] += 1
        if D2_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= D2_max:
            QFN_parameter_list[8]['maybe_data'].append(bottom_ocr_data[i])
            QFN_parameter_list[8]['maybe_data_num'] += 1
        if E2_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= E2_max:
            QFN_parameter_list[9]['maybe_data'].append(bottom_ocr_data[i])
            QFN_parameter_list[9]['maybe_data_num'] += 1
    for i in range(len(side_ocr_data)):
        # if D_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= D_max:
        #     QFN_parameter_list[0]['maybe_data'].append(side_ocr_data[i])
        #     QFN_parameter_list[0]['maybe_data_num'] += 1
        # if E_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= E_max:
        #     QFN_parameter_list[1]['maybe_data'].append(side_ocr_data[i])
        #     QFN_parameter_list[1]['maybe_data_num'] += 1
        if D1_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= D1_max:
            if len(body_x) > 0:
                QFN_parameter_list[2]['maybe_data'] = body_x
            else:
                QFN_parameter_list[2]['maybe_data'].append(side_ocr_data[i])
                QFN_parameter_list[2]['maybe_data_num'] += 1
        if E1_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= E1_max:
            if len(body_y) > 0:
                QFN_parameter_list[3]['maybe_data'] = body_y
            else:
                QFN_parameter_list[3]['maybe_data'].append(side_ocr_data[i])
                QFN_parameter_list[3]['maybe_data_num'] += 1
        if A_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= A_max:
            QFN_parameter_list[4]['maybe_data'].append(side_ocr_data[i])
            QFN_parameter_list[4]['maybe_data_num'] += 1
        if A1_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= A1_max:
            QFN_parameter_list[5]['maybe_data'].append(side_ocr_data[i])
            QFN_parameter_list[5]['maybe_data_num'] += 1
        if e_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= e_max:
            QFN_parameter_list[6]['maybe_data'].append(side_ocr_data[i])
            QFN_parameter_list[6]['maybe_data_num'] += 1
        if b_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= b_max:
            QFN_parameter_list[7]['maybe_data'].append(side_ocr_data[i])
            QFN_parameter_list[7]['maybe_data_num'] += 1
        if side_ocr_data[i]['Absolutely'] == 'angle':
            if θ2_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= θ2_max:
                QFN_parameter_list[15]['maybe_data'].append(side_ocr_data[i])
                QFN_parameter_list[15]['maybe_data_num'] += 1
    for i in range(len(detailed_ocr_data)):
        if A_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= A_max:
            QFN_parameter_list[4]['maybe_data'].append(detailed_ocr_data[i])
            QFN_parameter_list[4]['maybe_data_num'] += 1
        if L_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= L_max:
            QFN_parameter_list[10]['maybe_data'].append(detailed_ocr_data[i])
            QFN_parameter_list[10]['maybe_data_num'] += 1
        if GAGE_PLANE_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= GAGE_PLANE_max:
            QFN_parameter_list[11]['maybe_data'].append(detailed_ocr_data[i])
            QFN_parameter_list[11]['maybe_data_num'] += 1
        if c_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= c_max:
            QFN_parameter_list[12]['maybe_data'].append(detailed_ocr_data[i])
            QFN_parameter_list[12]['maybe_data_num'] += 1

        if detailed_ocr_data[i]['Absolutely'] == 'angle':
            if θ_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= θ_max:
                QFN_parameter_list[13]['maybe_data'].append(detailed_ocr_data[i])
                QFN_parameter_list[13]['maybe_data_num'] += 1
            if θ1_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= θ1_max:
                QFN_parameter_list[14]['maybe_data'].append(detailed_ocr_data[i])
                QFN_parameter_list[14]['maybe_data_num'] += 1

            if θ2_min < detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= θ2_max:
                QFN_parameter_list[15]['maybe_data'].append(detailed_ocr_data[i])
                QFN_parameter_list[15]['maybe_data_num'] += 1
            if θ3_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= θ3_max:
                QFN_parameter_list[16]['maybe_data'].append(detailed_ocr_data[i])
                QFN_parameter_list[16]['maybe_data_num'] += 1


    for i in range(len(QFN_parameter_list)):
        print("***/", QFN_parameter_list[i]['parameter_name'],"/***")

        for j in range(len(QFN_parameter_list[i]['maybe_data'])):
            print(QFN_parameter_list[i]['maybe_data'][j]['max_medium_min'])

    for i in range(len(QFN_parameter_list)):
        print(QFN_parameter_list[i]['maybe_data_num'])

    return QFN_parameter_list