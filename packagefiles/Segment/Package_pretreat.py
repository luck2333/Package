#F2.分装图预处理流程模块
import os.path
from PIL import Image,ImageEnhance
import matplotlib.pyplot as plt
import matplotlib
import copy

matplotlib.use('TkAgg')
# 外部文件
from packagefiles.Segment.Segment_function import *
from packagefiles.PackageExtract import QFP_extract
from packagefiles.PackageExtract import SOP_extract
from packagefiles.PackageExtract import QFN_extract
from packagefiles.PackageExtract import SON_extract
from packagefiles.PackageExtract import BGA_extract_old
from packagefiles.PackageExtract.get_pairs_data_present5 import extract_BGA_PIN
YOLO_RESULT = 'Result/Package_view/YoloPage'
SEGMENT_RESULT = 'Result/Package_view/page'
TEMP_SIDE = 'model/yolo_model/ExtractPackage/side.jpg'
TEMP_BOTTOM = 'model/yolo_model/ExtractPackage/bottom.jpg'
TEMP_TOP = 'model/yolo_model/ExtractPackage/top.jpg'
SEGMENT_SIDE = 'Result/Package_view/page/side.jpg'
SEGMENT_BOTTOM = 'Result/Package_view/page/bottom.jpg'
SEGMENT_TOP = 'Result/Package_view/page/top.jpg'
YOLO_IMG = 'Result/PDF_extract/yolo_img'

BGA_TABLE = ['Pitch x (el)', 'Pitch y (e)', 'Number of pins along X', 'Number of pins along Y',
             'Package Height (A)', 'Standoff (A1)', 'Body X (E)', 'Body Y (D)', 'Edge Fillet Radius',
             'Ball Diameter Normal (b)', 'Exclude Pins']
QFN_TABLE = ['Pitch x (el)', 'Pitch y (e)', 'Number of pins along X', 'Number of pins along Y',
             'Package Height (A)', 'Standoff (A1)', 'Pull Back (p)', 'Body X (E)', 'Body Y (D)',
             'Lead style', 'Pin Length (L)', 'Lead width (b)', 'Lead Height (c)', 'Exclude Pins',
             'Thermal X (E2)', 'Thermal Y (D2)']
QFP_TABLE = ['Number of pins along X', 'Number of pins along Y', 'Package Height (A)', 'Standoff (A1)',
             'Span X (E)', 'Span Y (D)', 'Body X (E1)', 'Body Y (D1)', 'Body draft (θ)', 'Edge Fillet radius',
             'Lead Length (L)', 'Lead width (b)', 'Lead Thickness (c)', 'Lead Radius (r)', 'Thermal X (E2)', 'Thermal Y (D2)']
SON_TABLE = ['Pitch (e)', 'Number of pins', 'Package Height (A)', 'Standoff (A1)', 'Pull Back (p)', 'Body X (E)',
             'Body Y (D)', 'Lead style', 'Lead Length (L)', 'Lead width (b)', 'Lead Height (c)', 'Exclude Pins', 'Thermal X (E2)', 'Thermal Y (D2)']

def package_coordinate_process(current_page,package_information):
    """
    此函数用来消除封装图上的表格、长文本信息像素，提取Package各视图
    :param current_page: 当前页图片坐标
    :param package_information: 整理后的封装信息
    :return: 返回干净的只包含封装信息的图片package_img
    """
    img_path = os.path.join(YOLO_IMG,f'{current_page+1}.png')
    original_img = Image.open(img_path)
    # 图像增强
    enhancer = ImageEnhance.Contrast(original_img)
    enhance_image = enhancer.enhance(factor=7)
    first_img = np.array(enhance_image)
    original_img = Image.fromarray(first_img)
    # 获取封装图各部分坐标信息
    class_information = package_information.get(current_page)
    print(class_information)
    table_coordinate = []
    class_dic = {}
    if not os.path.exists(f'{YOLO_RESULT}{current_page}'):
        os.makedirs(f'{YOLO_RESULT}{current_page}')
    for item in class_information:
        if 'Form' in item:
            table_coord = item['Form']
            table_x1,table_y1 = int(table_coord[0]),int(table_coord[1])
            table_x2,table_y2 = int(table_coord[2]),int(table_coord[3])
        
            for table_y in range(table_y1, table_y2 + 2):
                for table_x in range(table_x1, table_x2 + 2):
                    try:
                        first_img[table_y, table_x] = [255, 255, 255]
                    except:
                        pass
            table_coordinate.append(table_x1 // 5 - 10)
            table_coordinate.append(table_y1 // 5 - 10)
            table_coordinate.append(table_x2 // 5 + 10)
            table_coordinate.append(table_y2 // 5 + 30)
        elif 'Note' in item:
            note_coord = item['Note']
            note_x1,note_y1 = int(note_coord[0]),int(note_coord[1])
            note_x2,note_y2 = int(note_coord[2]),int(note_coord[3])

            for note_y in range(note_y1, note_y2 + 2):
                for note_x in range(note_x1, note_x2 + 2):
                    first_img[note_y, note_x] = [255, 255, 255]
        elif 'package' in item:
            person_coord = item['package']
            x1,y1 = int(person_coord[0]),int(person_coord[1])
            x2,y2 = int(person_coord[2]),int(person_coord[3])
            pred_box = (x1, y1, x2 + 30, y2)
            correct_box = jz(pred_box, first_img)
            x11,y11 = int(correct_box[0]),int(correct_box[1])
            x21,y21 = int(correct_box[2]),int(correct_box[3])
            # 裁剪获得封装图坐标
            croppend_image = original_img.crop((int(x11), int(y11) - 10, int(x21), int(y21)))
        elif 'Side' in item:
            # 区域中点坐标
            sx = item['Side'][0] + (item['Side'][2] - item['Side'][0]) // 2
            sy = item['Side'][1] + (item['Side'][3] - item['Side'][1]) // 2
            mix_s = [sx, sy]
            if class_dic.get('side') is None:
                side = original_img.crop(
                    (int(item['Side'][0]) - 20, int(item['Side'][1]) - 10, int(item['Side'][2] + 20),
                     int(item['Side'][3]) + 10))
                class_dic['side'] = mix_s
                cv2.imwrite(f'{YOLO_RESULT}{current_page}/side.jpg', np.array(side))
            else:
                class_dic['side1'] = mix_s
                side = original_img.crop(
                    (int(item['Side'][0]) - 20, int(item['Side'][1]) - 10, int(item['Side'][2]) + 20,
                     int(item['Side'][3]) + 10))
                cv2.imwrite(f'{YOLO_RESULT}{current_page}/side1.jpg', np.array(side))
        elif 'Top' in item or 'TOPVIEW' in item:
            # 区域中点坐标
            tx = item['Top'][0] + (item['Top'][2] - item['Top'][0]) // 2
            ty = item['Top'][1] + (item['Top'][3] - item['Top'][1]) // 2
            mix_t = [tx, ty]
            class_dic['top'] = mix_t
            top = original_img.crop(
                (int(item['Top'][0]) - 20, int(item['Top'][1]) - 10, int(item['Top'][2]) + 20,
                 int(item['Top'][3]) + 10))
            cv2.imwrite(f'{YOLO_RESULT}{current_page}/top.jpg', np.array(top))
        elif 'Detail' in item:
            # 区域中点坐标
            dx = item['Detail'][0] + (item['Detail'][2] - item['Detail'][0]) // 2
            dy = item['Detail'][1] + (item['Detail'][3] - item['Detail'][1]) // 2
            mix_t = [dx, dy]
            class_dic['detailed'] = mix_t
            top = original_img.crop(
                (int(item['Detail'][0]) - 20, int(item['Detail'][1]) - 10, int(item['Detail'][2]) + 20,
                 int(item['Detail'][3]) + 10))
            cv2.imwrite(f'{YOLO_RESULT}{current_page}/detailed.jpg', np.array(top))
        else:
            for key in item.keys():
                if key == 'BGA':
                    bx = item['BGA'][0] + (item['BGA'][2] - item['BGA'][0]) // 2
                    by = item['BGA'][1] + (item['BGA'][3] - item['BGA'][1]) // 2
                    mix_b = [bx, by]
                    if class_dic.get('bottom') is None:
                        class_dic['bottom'] = mix_b
                    else:
                        continue
                    bga = original_img.crop(
                        (int(item['BGA'][0]) - 20, int(item['BGA'][1]) - 10, int(item['BGA'][2]) + 20,
                         int(item['BGA'][3]) + 10))
                    cv2.imwrite(f'{YOLO_RESULT}{current_page}/bottom.jpg', np.array(bga))
                elif key == 'DFN_SON':
                    qx = item['DFN_SON'][0] + (item['DFN_SON'][2] - item['DFN_SON'][0]) // 2
                    qy = item['DFN_SON'][1] + (item['DFN_SON'][3] - item['DFN_SON'][1]) // 2
                    mix_q = [qx, qy]
                    class_dic['bottom'] = mix_q
                    qfn = original_img.crop(
                        (int(item['DFN_SON'][0]), int(item['DFN_SON'][1]) - 10, int(item['DFN_SON'][2]),
                         int(item['DFN_SON'][3])))
                    cv2.imwrite(f'{YOLO_RESULT}{current_page}/bottom.jpg', np.array(qfn))
                elif key == 'QFP':
                    qx = item['QFP'][0] + (item['QFP'][2] - item['QFP'][0]) // 2
                    qy = item['QFP'][1] + (item['QFP'][3] - item['QFP'][1]) // 2
                    mix_q = [qx, qy]
                    class_dic['bottom'] = mix_q
                    qfn = original_img.crop(
                        (int(item['QFP'][0]), int(item['QFP'][1]) - 10, int(item['QFP'][2]),
                         int(item['QFP'][3])))
                    cv2.imwrite(f'{YOLO_RESULT}{current_page}/bottom.jpg', np.array(qfn))

    for key,value in class_dic.items():
        value[0] = value[0] - x11
        value[1] = value[1] - y11

    # print(class_dic)

    return croppend_image, class_dic

def segment_package(package_img,class_dic, current_page):
    """
    :param package_img: 只包含封装信息的图片
    :param class_dic:YOLO检测框的中点值{'bottom': [892, 227], 'side': [877, 580], 'side1': [281, 617], 'top': [305, 218]}
    :param current_page:当前页
    :return:
    """
    image1 = np.array(package_img)
    # 定义扩充的像素数，例如每个边缘都扩充10个像素
    top, bottom, left, right = 20, 20, 20, 20
    # 使用白色像素扩充边缘
    image = cv2.copyMakeBorder(image1, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    edges, dilated_image = morphological_treatment(image)
    # plt.imshow(dilated_image)
    # plt.show()
    # plt.imshow(edges)
    # plt.show()
    # ret, thresh = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)
    result_image = np.zeros_like(image)
    kernel = np.ones((3, 3), np.uint8)
    view_flage = False  # 判断视图中是否有view命名的图片
    while True:
        y, x, found_white_pixel = np_where(edges)
        if found_white_pixel:
            start_x, start_y = y, x  # 选择一个多边形边上的像素点作为起点
            contour = find_contour(edges, start_x, start_y)  # 找到多边形的外轮廓
            processed_list = [[int(str(item[0]).lstrip()), item[1]] for item in contour]
            # 对最外层轮廓进行重写
            for coord in processed_list:
                x, y = coord
                result_image[x, y] = 255
            # 对上列表按照第二个元素从小到大排序
            x_list = sorted(processed_list, key=lambda x: x[1])  # 列排序
            y_list = sorted(processed_list, key=lambda y: y[0])  # 行排序
            x1,x2 = x_list[0][1],x_list[len(x_list) - 1][1]
            y1,y2 = y_list[0][0],y_list[len(y_list) - 1][0]

            area = (x2 - x1) * (y2 - y1)
            if area < 25000:
                # 清空该区域
                for i in range(y1, y2 + 1):
                    for j in range(x1, x2 + 1):
                        edges[i, j] = 0
                continue
            # ----------------------------------------------------
            result_image = cv2.dilate(result_image, kernel, iterations=2)  # 略微膨胀
            gray = cv2.cvtColor(result_image, cv2.COLOR_RGB2GRAY)
            # 应用阈值将图像二值化
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(gray)
            # 在掩码图像上绘制所有轮廓，并填充轮廓内部
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
            # result = cv2.bitwise_and(image, image, mask=mask)
            white_background = np.ones_like(image) * 255
            # 将原始图像中掩码区域的内容复制到全白背景图像上
            inverse_mask = cv2.bitwise_not(mask)
            white_background[inverse_mask == 255] = [255, 255, 255]
            white_background[mask == 255] = image[mask == 255]
            edges[mask == 255] = [0]  # 清除已切割区域
            img1 = Image.fromarray(white_background)
            crop_img = img1.crop((x1 - 5, y1 - 5, x2 + 5, y2 + 5))
            crop_img = np.array(crop_img)
            # 确保生成保存文件夹
            if not os.path.exists(SEGMENT_RESULT):
                os.makedirs(SEGMENT_RESULT)
            # plt.imshow(crop_img)
            # plt.show()
            # YOLO作为分类标准
            name_class = 'view'
            key_list = []
            test_list = ['bottom', 'side', 'side1', 'top']
            for key, value in class_dic.items():
                key_list.append(key)
                if key in test_list:
                    test_list.remove(key)
                if x1 < value[0] < x2 and y1 < value[1] < y2:
                    name_class = key
                    break
                else:
                    continue

            if name_class == 'view':
                view_flage = True
            cv2.imwrite(f'{SEGMENT_RESULT}/{name_class}.jpg', crop_img)  # YOLOX分类保存
        else:
            break
    # 处理未命名view情况：
    if view_flage:
        # for name in os.listdir('Package_view/page'):
        if 'bottom.jpg' in os.listdir(SEGMENT_RESULT) and 'top.jpg' in os.listdir(
                SEGMENT_RESULT) and 'side.jpg' in os.listdir(SEGMENT_RESULT):
            pass
        elif 'bottom.jpg' in os.listdir(SEGMENT_RESULT) and 'top.jpg' in os.listdir(
                SEGMENT_RESULT):
            os.rename(f'{SEGMENT_RESULT}/view.jpg', SEGMENT_SIDE)  # side
        elif 'side.jpg' in os.listdir(SEGMENT_RESULT) and 'top.jpg' in os.listdir(
                SEGMENT_RESULT):
            os.rename(f'{SEGMENT_RESULT}/view.jpg', SEGMENT_BOTTOM)  # bottom

        elif 'bottom.jpg' in os.listdir(SEGMENT_RESULT) and 'side.jpg' in os.listdir(
                SEGMENT_RESULT):
            os.rename(f'{SEGMENT_RESULT}/view.jpg', SEGMENT_TOP)  # top

        elif 'side.jpg' in os.listdir(SEGMENT_RESULT) and 'top.jpg' not in os.listdir(
                SEGMENT_RESULT):
            os.rename(f'{SEGMENT_RESULT}/view.jpg', SEGMENT_BOTTOM)  # bottom
    # 添加top视图
    # if 'top.jpg' not in os.listdir('Package_view/page'):
    #     shutil.copy('yolox_onnx/top.jpg','Package_view/page/top.jpg')
    #     shutil.copy('yolox_onnx/top.jpg','data/top.jpg')
    if (not os.path.exists(SEGMENT_TOP)) and os.path.exists(
            f'{YOLO_RESULT}{current_page}/top.jpg'):
        shutil.move(f'{YOLO_RESULT}{current_page}/top.jpg', SEGMENT_TOP)
    elif (not os.path.exists(SEGMENT_TOP)) and not os.path.exists(
            f'{YOLO_RESULT}{current_page}/top.jpg'):
        shutil.copy(TEMP_TOP, SEGMENT_TOP)

    if (not os.path.exists(SEGMENT_BOTTOM)) and os.path.exists(
            f'{YOLO_RESULT}{current_page}/bottom.jpg'):
        shutil.move(f'{YOLO_RESULT}{current_page}/bottom.jpg', SEGMENT_BOTTOM)
    elif (not os.path.exists(SEGMENT_BOTTOM)) and not os.path.exists(
            f'{YOLO_RESULT}{current_page}/bottom.jpg'):
        shutil.copy(TEMP_BOTTOM, SEGMENT_BOTTOM)

    # 处理多side情况
    if 'side1.jpg' in os.listdir(SEGMENT_RESULT) and 'side.jpg' in os.listdir(SEGMENT_RESULT):
        side = f'{SEGMENT_RESULT}/side.jpg'
        side1 = f'{SEGMENT_RESULT}/side1.jpg'
        side_combined(side, side1, save_path=SEGMENT_RESULT)
    elif 'side1.jpg' in os.listdir(SEGMENT_RESULT) and 'side.jpg' not in os.listdir(
            SEGMENT_RESULT):
        side = f'R{YOLO_RESULT}{current_page}/side.jpg'
        side1 = f'{SEGMENT_RESULT}/side1.jpg'
        side_combined(side, side1, save_path=SEGMENT_RESULT)
def package_process(current_page,package_information):
    """
    对封装图片进行分割操作，并保存命名
    :param current_page: 当前页
    :param package_information: 整理后的封装信息
    :return:
    """
    if os.path.exists(SEGMENT_RESULT):
        shutil.rmtree('Result/Package_view')

    package_image, class_dic = package_coordinate_process(current_page,package_information)

    # 显示传入的图像
    # plt.imshow(package_image)
    # plt.show()
    # 分割函数
    segment_package(package_image,class_dic, current_page)

    # 处理意外分割情况
    if os.path.exists(f'{YOLO_RESULT}{current_page}/top.jpg'):
       shutil.move(f'{YOLO_RESULT}{current_page}/top.jpg',SEGMENT_TOP)

    if os.path.exists(f'{YOLO_RESULT}{current_page}/bottom.jpg'):
       shutil.move(f'{YOLO_RESULT}{current_page}/bottom.jpg',SEGMENT_BOTTOM)

    if ( not os.path.exists(SEGMENT_SIDE) )and os.path.exists(f'{YOLO_RESULT}{current_page}/side.jpg'):
        shutil.move(f'{YOLO_RESULT}{current_page}/side.jpg', SEGMENT_SIDE)
    elif ( not os.path.exists(SEGMENT_SIDE) )and not os.path.exists(f'{YOLO_RESULT}{current_page}/side.jpg'):
        shutil.copy(TEMP_SIDE, SEGMENT_SIDE)
    #
    # # 对分割后的图片按照祁新源要求进行放大处理
    # set_image_size(SEGMENT_BOTTOM)
    # set_image_size(SEGMENT_TOP)
    # set_image_size(SEGMENT_SIDE)


def package_indentify(package_type, current_page):
    """
    进行封装视图的信息提取
    :param package_type: 该封装类型
    :return:
    """
    destination_folder_path = "Result/Package_extract/data"
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
        print(f"文件夹 {destination_folder_path} 已创建")

    # 调用函数进行移动操作
    move_files(SEGMENT_RESULT, destination_folder_path)
    # 按照类型进行封装图信息提取
    if package_type == 'QFP':
        out_put = QFP_extract.extract_package(package_type, current_page)
    elif package_type == 'QFN':
        out_put = QFN_extract.extract_QFN(package_type, current_page)
    elif package_type == 'SOP':
        out_put = SOP_extract.extract_SOP(package_type, current_page)
    elif package_type == 'SON' or package_type == 'DFN' or package_type == 'DFN_SON':
        out_put = SON_extract.extract_SON(package_type, current_page)
    elif package_type == 'BGA':
        out_put = BGA_extract_old.extract_BGA(current_page, letter_or_number='number', table_dic=[])
    else:
        print("未定义的封装类型")
        out_put = []



    return out_put


def extract_BGA_pins():
    # 提取BGA引脚数量

    pin_num_x_serial, pin_num_y_serial, loss_pin,loss_color = extract_BGA_PIN()
    if pin_num_x_serial!=pin_num_y_serial:
        pin_num_x_serial = max(pin_num_x_serial,pin_num_y_serial)
        pin_num_y_serial = max(pin_num_x_serial,pin_num_y_serial)
    return pin_num_x_serial, pin_num_y_serial, loss_pin,loss_color

# 提供给迪浩的代码借口
def manage_result(out_put, package_type):
    # if len(out_put)==4:
    for out_list in out_put:
        del out_list[0]
    result = []
    for out_list in out_put:
        result1 =  []
        for item in out_list:
            if item == '' or item == '-':
                result1.append(None)
            else:
                result1.append(item)
        result.append(result1)

    record_json = {"pkg_type":None,"parameters":{}}
    # record_json[package_type] = {}
    if package_type == 'QFP':
        for i,key in enumerate(QFP_TABLE):
            record_json["pkg_type"] = package_type
            record_json["parameters"][key] = result[i]
    if package_type == 'QFN':
        for i, key in enumerate(QFN_TABLE):
            record_json["pkg_type"] = package_type
            record_json["parameters"][key] = result[i]
    if package_type == 'SON':
        for i, key in enumerate(SON_TABLE):
            record_json["pkg_type"] = package_type
            record_json["parameters"][key] = result[i]
    if package_type == 'BGA':
        for i, key in enumerate(BGA_TABLE):
            record_json["pkg_type"] = package_type
            record_json["parameters"][key] = result[i]

    return record_json
def reco_package(package_type,current_package,current_page,pdf_path):
    from packagefiles.TableProcessing import Table_extract

    Table_Coordinate_List = []
    page_Number_List = []
    result = None
    pin_num_x_serial = None
    pin_num_y_serial = None
    # 封装类型
  
    if package_type == 'DFN_SON' or package_type == 'DFN':
        package_type = 'SON'
    # 判断是自动搜索还是手动框选
    # 判断是否走数字流程有两个条件，一个是当前封装信息current_package内无Form；另一个是有Form但不是封装Form，这个就是在识别表格的时候才显示。
    if current_package['part_content'] is not None:
        manage_data = manage_json(current_package)
        package_process(current_page, manage_data[0])  # 分割流程
        exists = any(part['part_name'] == 'Form' for part in current_package['part_content'])
        if exists:
            # 表格提取
            current_table = manage_data[1][current_page]
            page_Number_List, Table_Coordinate_List = adjust_table_coordinates(current_page, current_table)
        else:
            print('数字提取')
            result = package_indentify(package_type, current_page)
    elif current_package['part_content'] is None and current_package['type'] == 'list':  # 说明是自动框表
        # 目前只考虑识别当前框选的表，暂不考虑识别多个框选的表
        Table_Coordinate_List = [[], current_package['rect'], []]
        page_Number_List = [current_page, current_page + 1, current_page + 2]
    elif current_package['part_content'] is None and current_package['type'] == 'img':  # 说明是自动框图
        # 框选图流程存在争议
        pass
    if len(page_Number_List) != 0 and len(Table_Coordinate_List) != 0:
        try:
            # 表格内容提取
            data = Table_extract.extract_table(pdf_path, page_Number_List, Table_Coordinate_List, package_type,
                                               current_page)
            if package_type == 'BGA':
                # 如果表格类型是BGA,运行数字提取BGA引脚数量
                pin_num_x_serial, pin_num_y_serial, loss_pin, loss_color = extract_BGA_pins()

            # 后续操作只考虑了BGA表格类型
            if package_type == 'QFP':
                if not data:
                    # 走数字提取流程
                    print("-----表格数据提取为空-----")
                    result = package_indentify(package_type, current_page)
                else:
                    result = data
            elif package_type == 'BGA':
                result = data[0:11]
                result[1] = copy.deepcopy(result[0])
                result[10][2] = str(loss_color)
                result[10][1] = str(loss_pin)
                if pin_num_x_serial != None and (result[2][2] == '' or result[2][2] == 0):
                    result[2] = ['', '', pin_num_x_serial, '']
                if pin_num_y_serial != None and (result[3][2] == '' or result[3][2] == 0):
                    result[3] = ['', '', pin_num_y_serial, '']

            elif package_type == 'SON':
                result = data[0:14]

            elif package_type == 'SOP':
                result = data[0:12]

            elif package_type == 'QFN':
                result = data
        except Exception as e:
            print(e)
            # 走数字提取流程
            result = package_indentify(package_type, current_page)
    out_put = manage_result(result, package_type)

    return out_put