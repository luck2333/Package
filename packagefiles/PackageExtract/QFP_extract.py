
# 外部文件：
from packagefiles.PackageExtract.function_tool import *
from packagefiles.PackageExtract.get_pairs_data_present5_test import *
from packagefiles.PackageExtract.onnx_use import Run_onnx_det, Run_onnx
from packagefiles.PackageExtract.yolox_onnx_py.onnx_QFP_pairs_data_location2 import begain_output_pairs_data_location
from packagefiles.PackageExtract.DETR_BGA import DETR_BGA
import json
#全局路径
DATA = 'Result/Package_extract/data'
DATA_BOTTOM_CROP = 'Result/Package_extract/data_bottom_crop'
DATA_COPY = 'Result/Package_extract/data_copy'
ONNX_OUTPUT = 'Result/Package_extract/onnx_output'
OPENCV_OUTPUT = 'Result/Package_extract/opencv_output'



def extract_package(package_classes, page_num):
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
    L3 = get_Pin_data(L3,package_classes)

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
    QFP_parameter_list, nx, ny = find_QFP_parameter(L3)
    # 整理获得的参数
    parameter_list = get_QFP_parameter_data(QFP_parameter_list, nx, ny)

    # 20250722添加
    # 指定要查找的 page_num
    target_page_num = page_num
    json_file = 'output.json'
    result = []
    # 读取 JSON 文件
    print("开始读取json文件")
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
    print("修改前的参数列表", parameter_list)
    # 参数检查与修改
    parameter_list = alter_QFP_parameter_data(parameter_list)
    print("修改后的参数列表", parameter_list)
    # if result != []:
    #     if result[1] != None:
    #         parameter_list[0][1] = result[1]
    #         parameter_list[0][2] = result[1]
    #         parameter_list[0][3] = result[1]
    #     if result[2] != None:
    #         parameter_list[1][1] = result[2]
    #         parameter_list[1][2] = result[2]
    #         parameter_list[1][3] = result[2]
    #     if result[3] != None:
    #         parameter_list[2][1] = result[3]
    #         parameter_list[2][2] = result[3]
    #         parameter_list[2][3] = result[3]
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


    #20250621修改顺序
    # ['实体长D1', '实体宽E1', '实体高A', '支撑高A1', '端子高A3', '外围长D', '外围宽E', 'PIN长L', 'PIN宽b', '行PIN数',
    #  '列PIN数', '行/列PIN间距e', '散热盘长D2', '散热盘宽E2', '削角长度', '端子厚度', '接触角度', '端腿角度',
    #  '主体顶部绘制角度']

    # new_parameter_list = []
    # new_parameter_list.append(parameter_list[0])
    # new_parameter_list.append(parameter_list[1])
    # new_parameter_list.append(parameter_list[2])
    # new_parameter_list.append(parameter_list[3])
    # new_parameter_list.append([0,'-','-','-'])
    # new_parameter_list.append(parameter_list[5])
    # new_parameter_list.append(parameter_list[6])
    # new_parameter_list.append(parameter_list[7])
    # new_parameter_list.append(parameter_list[8])
    # new_parameter_list.append(parameter_list[9])
    # new_parameter_list.append(parameter_list[10])
    # new_parameter_list.append(parameter_list[11])
    # new_parameter_list.append(parameter_list[12])
    # new_parameter_list.append(parameter_list[13])
    # new_parameter_list.append(parameter_list[14])
    # new_parameter_list.append(parameter_list[15])
    # new_parameter_list.append(parameter_list[16])
    # new_parameter_list.append(parameter_list[17])
    # new_parameter_list.append(parameter_list[18])
    # 20250621修改顺序
    new_parameter_list = []
    new_parameter_list.append(parameter_list[9])
    new_parameter_list.append(parameter_list[10])
    new_parameter_list.append(parameter_list[2])
    new_parameter_list.append(parameter_list[3])
    new_parameter_list.append(parameter_list[5])
    new_parameter_list.append(parameter_list[6])
    new_parameter_list.append(parameter_list[0])
    new_parameter_list.append(parameter_list[1])
    new_parameter_list.append(parameter_list[16])
    new_parameter_list.append([0, '-', '-', '-'])
    new_parameter_list.append(parameter_list[7])
    new_parameter_list.append(parameter_list[8])
    new_parameter_list.append(parameter_list[15])
    new_parameter_list.append([0, '-', '-', '-'])
    new_parameter_list.append(parameter_list[12])
    new_parameter_list.append(parameter_list[13])
    return new_parameter_list

def front_loading_work():
    """
    预处理，将data文件夹中的所有文件复制到data_copy文件夹中,同时清空上一个封装的数据
    :return:
    """

    # 1.初始化文件夹
    # 清空文件夹
    empty_folder(ONNX_OUTPUT)
    # 创建文件夹
    os.makedirs(ONNX_OUTPUT)
    # 清空文件夹
    empty_folder(DATA_BOTTOM_CROP)
    # 创建文件夹
    os.makedirs(DATA_BOTTOM_CROP)
    '''
    这里将分割好的视图放入data文件夹即可提取QFP
    '''

    # 2.将分割好的三视图备份
    # 固定三视图的尺寸，并增强画质规定最长的一边为1000
    filein = f'{DATA}/top.jpg'
    fileout = filein
    try:
        set_Image_size(filein, fileout)
    except:
        print('文件', filein, '不存在')
    filein = f'{DATA}/bottom.jpg'
    fileout = filein
    try:
        set_Image_size(filein, fileout)
    except:
        print('文件', filein, '不存在')
    filein = f'{DATA}/side.jpg'
    fileout = filein
    try:
        set_Image_size(filein, fileout)
    except:
        print('文件', filein, '不存在')
    filein = f'{DATA}/detailed.jpg'
    fileout = filein
    try:
        set_Image_size(filein, fileout)
    except:
        print('文件', filein, '不存在')

    # 清空文件夹
    empty_folder(DATA_COPY)
    # 创建文件夹
    os.makedirs(DATA_COPY)
    filePath = DATA
    file_name_list = os.listdir(filePath)
    for file_name in file_name_list:
        shutil.copy(f'{DATA}/{file_name}', f'{DATA_COPY}/{file_name}')

    empty_folder(OPENCV_OUTPUT)
    os.makedirs(OPENCV_OUTPUT)
    # 清空文件夹
    empty_folder(DATA)
    # 创建文件夹
    os.makedirs(DATA)
    filePath = DATA_COPY
    file_name_list = os.listdir(filePath)
    for file_name in file_name_list:
        shutil.copy(f'{DATA_COPY}/{file_name}', f'{DATA}/{file_name}')



    # 2.检测图像中的文本信息和图像信息
def opencv_get_outline(package_path):
    """
    使用OPENCV提取器件外框线
    :param package_path: data 文件夹所在路径，包含分割好的每个视图
    :return: 外框线
    """

def dbnet_get_text_box(img_path):
    """
    DBNet提取所有文本框坐标范围
    :param img_path: data 图片所在路径
    :return: 包含所有文本框坐标范围(外框)
    """
    location_cool = Run_onnx_det(img_path)
    dbnet_data = np.empty((len(location_cool), 4))  # [x1,x2,x3,x4]
    for i in range(len(location_cool)):
        dbnet_data[i][0] = min(location_cool[i][2], location_cool[i][0])
        dbnet_data[i][1] = min(location_cool[i][3], location_cool[i][1])
        dbnet_data[i][2] = max(location_cool[i][2], location_cool[i][0])
        dbnet_data[i][3] = max(location_cool[i][3], location_cool[i][1])

    dbnet_data = np.around(dbnet_data, decimals=2)

    return dbnet_data

def yolo_classify(img_path, package_classes):
    """
    Yolo分类，并提取图像元素坐标范围
    :param img_path: data 图片所在路径
    :return: yolox_pairs:标尺线的坐标范围(外框),yolox_num:尺寸标注的坐标范围(外框),yolox_serial_num:行列序号的坐标范围(外框),pin:pin的坐标范围(外框),other:无关图像元素的坐标范围(外框),pad:散热盘的坐标范围(外框),border:本体外框的坐标范围(外框), angle_pairs:角度标尺线的坐标范围(外框)
    """
    if package_classes == 'BGA':
        yolox_pairs, yolox_num, yolox_serial_num, pin, other, pad, border, angle_pairs, BGA_serial_num, BGA_serial_letter = begain_output_pairs_data_location(
            img_path, package_classes)
        # yolox_pairs, yolox_num, yolox_serial_num, pin, other, pad, border, angle_pairs, BGA_serial_num, BGA_serial_letter = DETR_BGA(img_path, package_classes)
        yolox_pairs_null, yolox_num_null, yolox_serial_num_null, pin, other_null, pad_null, border, angle_pairs_null, BGA_serial_num, BGA_serial_letter = DETR_BGA(img_path, package_classes)
        print("yolox_pairs", yolox_pairs)
        print("yolox_num", yolox_num)
        print("yolox_serial_num", yolox_serial_num)
        print("pin", pin)
        print("other", other)
        print("pad", pad)
        print("border", border)
        print("angle_pairs", angle_pairs)
        print("BGA_serial_num", BGA_serial_num)
        print("BGA_serial_letter", BGA_serial_letter)

    else:
        yolox_pairs, yolox_num, yolox_serial_num, pin, other, pad, border, angle_pairs,BGA_serial_num, BGA_serial_letter = begain_output_pairs_data_location(
            img_path, package_classes)

        yolox_pairs = np.around(yolox_pairs, decimals=2)
        yolox_num = np.around(yolox_num, decimals=2)
        angle_pairs = np.around(angle_pairs, decimals=2)

    return yolox_pairs, yolox_num, yolox_serial_num, pin, other, pad, border, angle_pairs, BGA_serial_num, BGA_serial_letter


def get_data_location_by_yolo_dbnet(package_path, package_classes):
    """
    使用Yolo和DBNet提取各个视图的图像元素和文本的的坐标范围(外框)
    :param package_path: data 文件夹所在路径，包含分割好的每个视图
    :return: L3:包含所有图像元素和文本的坐标范围(外框)集合
    """
    L3 = []
    empty_data = np.empty((0, 4))
    img_path = f'{package_path}/top.jpg'
    if not os.path.exists(img_path):
        top_dbnet_data = empty_data
        top_yolox_pairs = empty_data
        top_yolox_num = empty_data
        top_yolox_serial_num = empty_data
        top_pin = empty_data
        top_other = empty_data
        top_pad = empty_data
        top_border = empty_data
        top_angle_pairs = empty_data
    else:
        top_dbnet_data = dbnet_get_text_box(img_path)
        top_yolox_pairs, top_yolox_num, top_yolox_serial_num, top_pin, top_other, top_pad, top_border, top_angle_pairs, top_BGA_serial_num, top_BGA_serial_letter = yolo_classify(img_path, package_classes)
    img_path = f'{package_path}/bottom.jpg'
    if not os.path.exists(img_path):
        bottom_dbnet_data = empty_data
        bottom_yolox_pairs = empty_data
        bottom_yolox_num = empty_data
        bottom_yolox_serial_num = empty_data
        bottom_pin = empty_data
        bottom_other = empty_data
        bottom_pad = empty_data
        bottom_border = empty_data
        bottom_angle_pairs = empty_data
    else:
        bottom_dbnet_data = dbnet_get_text_box(img_path)
        bottom_yolox_pairs, bottom_yolox_num, bottom_yolox_serial_num, bottom_pin, bottom_other, bottom_pad, bottom_border, bottom_angle_pairs, bottom_BGA_serial_num, bottom_BGA_serial_letter = yolo_classify(img_path, package_classes)
    img_path = f'{package_path}/side.jpg'
    if not os.path.exists(img_path):
        side_dbnet_data = empty_data
        side_yolox_pairs = empty_data
        side_yolox_num = empty_data
        side_yolox_serial_num = empty_data
        side_pin = empty_data
        side_other = empty_data
        side_pad = empty_data
        side_border = empty_data
        side_angle_pairs = empty_data
    else:
        side_dbnet_data = dbnet_get_text_box(img_path)
        side_yolox_pairs, side_yolox_num, side_yolox_serial_num, side_pin, side_other, side_pad, side_border, side_angle_pairs, side_BGA_serial_num, side_BGA_serial_letter = yolo_classify(img_path, package_classes)
    img_path = f'{package_path}/detailed.jpg'
    if not os.path.exists(img_path):
        detailed_dbnet_data = empty_data
        detailed_yolox_pairs = empty_data
        detailed_yolox_num = empty_data
        detailed_yolox_serial_num = empty_data
        detailed_pin = empty_data
        detailed_other = empty_data
        detailed_pad = empty_data
        detailed_border = empty_data
        detailed_angle_pairs = empty_data
    else:
        detailed_dbnet_data = dbnet_get_text_box(img_path)
        detailed_yolox_pairs, detailed_yolox_num, detailed_yolox_serial_num, detailed_pin, detailed_other, detailed_pad, detailed_border, detailed_angle_pairs, detailed_BGA_serial_num, detailed_BGA_serial_letter = yolo_classify(img_path, package_classes)
    # 将所有列表添加到L3中
    L3.append({'list_name': 'top_dbnet_data', 'list': top_dbnet_data})
    L3.append({'list_name': 'top_yolox_pairs', 'list': top_yolox_pairs})
    L3.append({'list_name': 'top_yolox_num', 'list': top_yolox_num})
    L3.append({'list_name': 'top_yolox_serial_num', 'list': top_yolox_serial_num})
    L3.append({'list_name': 'top_pin', 'list': top_pin})
    L3.append({'list_name': 'top_other', 'list': top_other})
    L3.append({'list_name': 'top_pad', 'list': top_pad})
    L3.append({'list_name': 'top_border', 'list': top_border})
    L3.append({'list_name': 'top_angle_pairs', 'list': top_angle_pairs})
    L3.append({'list_name': 'bottom_dbnet_data', 'list': bottom_dbnet_data})
    L3.append({'list_name': 'bottom_yolox_pairs', 'list': bottom_yolox_pairs})
    L3.append({'list_name': 'bottom_yolox_num', 'list': bottom_yolox_num})
    L3.append({'list_name': 'bottom_yolox_serial_num', 'list': bottom_yolox_serial_num})
    L3.append({'list_name': 'bottom_pin', 'list': bottom_pin})
    L3.append({'list_name': 'bottom_other', 'list': bottom_other})
    L3.append({'list_name': 'bottom_pad', 'list': bottom_pad})
    L3.append({'list_name': 'bottom_border', 'list': bottom_border})
    L3.append({'list_name': 'bottom_angle_pairs', 'list': bottom_angle_pairs})
    L3.append({'list_name': 'bottom_BGA_serial_letter', 'list': bottom_BGA_serial_letter})
    L3.append({'list_name': 'bottom_BGA_serial_num', 'list': bottom_BGA_serial_num})
    L3.append({'list_name': 'side_dbnet_data', 'list': side_dbnet_data})
    L3.append({'list_name': 'side_yolox_pairs', 'list': side_yolox_pairs})
    L3.append({'list_name': 'side_yolox_num', 'list': side_yolox_num})
    L3.append({'list_name': 'side_yolox_serial_num', 'list': side_yolox_serial_num})
    L3.append({'list_name': 'side_pin', 'list': side_pin})
    L3.append({'list_name': 'side_other', 'list': side_other})
    L3.append({'list_name': 'side_pad', 'list': side_pad})
    L3.append({'list_name': 'side_border', 'list': side_border})
    L3.append({'list_name': 'side_angle_pairs', 'list': side_angle_pairs})
    L3.append({'list_name': 'detailed_dbnet_data', 'list': detailed_dbnet_data})
    L3.append({'list_name': 'detailed_yolox_pairs', 'list': detailed_yolox_pairs})
    L3.append({'list_name': 'detailed_yolox_num', 'list': detailed_yolox_num})
    L3.append({'list_name': 'detailed_yolox_serial_num', 'list': detailed_yolox_serial_num})
    L3.append({'list_name': 'detailed_pin', 'list': detailed_pin})
    L3.append({'list_name': 'detailed_other', 'list': detailed_other})
    L3.append({'list_name': 'detailed_pad', 'list': detailed_pad})
    L3.append({'list_name': 'detailed_border', 'list': detailed_border})
    L3.append({'list_name': 'detailed_angle_pairs', 'list': detailed_angle_pairs})
    return L3


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


def get_Pin_data(L3, package_classes):
    top_yolox_serial_num = find_list(L3, 'top_yolox_serial_num')
    bottom_yolox_serial_num = find_list(L3, 'bottom_yolox_serial_num')
    top_ocr_data = find_list(L3, 'top_ocr_data')
    bottom_ocr_data = find_list(L3, 'bottom_ocr_data')
    if package_classes == 'QFP' or package_classes == 'QFN' or package_classes == 'SOP' or package_classes == 'SON':
        top_serial_numbers_data, bottom_serial_numbers_data, top_ocr_data, bottom_ocr_data = find_PIN(top_yolox_serial_num,
                                                                       bottom_yolox_serial_num, top_ocr_data,
                                                                       bottom_ocr_data)

        recite_data(L3, 'top_serial_numbers_data', top_serial_numbers_data)
        recite_data(L3, 'bottom_serial_numbers_data', bottom_serial_numbers_data)
        recite_data(L3, 'top_ocr_data', top_ocr_data)
        recite_data(L3, 'bottom_ocr_data', bottom_ocr_data)
    if package_classes == 'BGA':
        bottom_BGA_serial_number = find_list(L3, 'bottom_BGA_serial_number')
        bottom_BGA_serial_letter = find_list(L3, 'bottom_BGA_serial_letter')
        bottom_ocr_data = find_list(L3, 'bottom_ocr_data')

        bottom_BGA_serial_number, bottom_BGA_serial_letter, bottom_ocr_data = find_BGA_PIN(
            bottom_BGA_serial_number, bottom_BGA_serial_letter, bottom_ocr_data)
        #数据整理适应旧函数
        # serial_numbers_data: np.(, 4)['x1', 'y1', 'x2', 'y2', 'str']ocr之后得到的单个数字
        # serial_letters_data: np.(, 4)['x1', 'y1', 'x2', 'y2', 'str']
        # serial_numbers: np.(, 4)[x1, y1, x2, y2)yolo得到的一组数字序号位置
        # serial_letters: np.(, 4)[x1, y1, x2, y2)
        serial_numbers_data = np.empty((0,4))
        for i in range(len(bottom_BGA_serial_number)):
            mid = np.empty(5)
            mid[0:4] = bottom_BGA_serial_number[i]['location'].astype(str)
            mid[4] = bottom_BGA_serial_number[i]['key_info'][0]
            serial_numbers_data = np.r_[serial_numbers_data, [mid]]
        serial_numbers_data = np.empty((0,4))
        for i in range(len(bottom_BGA_serial_letter)):
            mid = np.empty(5)
            mid[0:4] = bottom_BGA_serial_letter[i]['location'].astype(str)
            mid[4] = bottom_BGA_serial_letter[i]['key_info'][0]
            serial_letters_data = np.r_[serial_letters_data, [mid]]
        serial_numbers = bottom_BGA_serial_number
        serial_letters = bottom_BGA_serial_letter
        pin_num_x_serial, pin_num_y_serial, pin_1_location = find_pin_num_pin_1(serial_numbers_data,
                                                                                serial_letters_data,
                                                                                serial_numbers, serial_letters)
        recite_data(L3, 'bottom_BGA_serial_number', bottom_BGA_serial_number)
        recite_data(L3, 'bottom_BGA_serial_letter', bottom_BGA_serial_letter)
        recite_data(L3, 'bottom_ocr_data', bottom_ocr_data)
        recite_data(L3, 'pin_num_x_serial', pin_num_x_serial)
        recite_data(L3, 'pin_num_y_serial', pin_num_y_serial)
        recite_data(L3, 'pin_1_location', pin_1_location)
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