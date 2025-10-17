import os
import shutil
import csv  # 导入csv模块
import json
from packagefiles.UI.detect import Detect
from packagefiles.UI import ui_class
from packagefiles.UI.detect import PreProcess
from packagefiles.UI.ui_class import RecoTableThread
from packagefiles.Segment.Segment_function import get_type

BGA_TABLE = ['Pitch x (el)', 'Pitch y (e)', 'Number of pins along X', 'Number of pins along Y',
             'Package Height (A)', 'Standoff (A1)', 'Body X (D)', 'Body Y (E)', 'Edge Fillet Radius',
             'Ball Diameter Normal (b)', 'Exclude Pins']
QFN_TABLE = ['Pitch x (el)', 'Pitch y (e)', 'Number of pins along X', 'Number of pins along Y',
             'Package Height (A)', 'Standoff (A1)', 'Pull Back (p)', 'Body X (D)', 'Body Y (E)',
             'Lead style', 'Pin Length (L)', 'Lead width (b)', 'Lead Height (c)', 'Exclude Pins',
             'Thermal X (D2)', 'Thermal Y (E2)']
QFP_TABLE = ['Number of pins along X', 'Number of pins along Y', 'Package Height (A)', 'Standoff (A1)',
             'Span X (E)', 'Span Y (D)', 'Body X (D1)', 'Body Y (E1)', 'Body draft (θ)', 'Edge Fillet radius',
             'Lead Length (L)', 'Lead width (b)', 'Lead Thickness (c)', 'Lead Radius (r)', 'Thermal X (D2)', 'Thermal Y (E2)']


# 自动搜索
def auto_detect_reco(pdf_path,test_type):
    page = PreProcess(pdf_path).page_list
    detect = Detect(pdf_path, page)
    detect.pre_process()
    detect.process()
    detect.post_process()
    # 自动识别
    type_dict = get_type(detect.data2)
    for index in range(0, len(detect.data2)):
        pdf_page_count = detect.data2[index]["page"]
        current_package = detect.data2[index]
        # 识别函数
        if detect.data2[index]['package_type'] is None:
            print(f"{pdf_page_count}页封装类型为空，跳过")
            continue
        elif detect.data2[index]['package_type'] != test_type:
            continue
        reco = RecoTableThread(pdf_path, detect.data2[index]["page"], detect.data2[index], detect.data2, type_dict)
        reco.run()

        # ------------------修改后的代码---------------------------
        table_mapping = {
            'BGA': BGA_TABLE,
            'QFN': QFN_TABLE,
            'QFP': QFP_TABLE
        }
        Table_type = table_mapping.get(test_type, None)
        pdf_name = os.path.basename(pdf_path)
        # 获取当前封装类型
        package_type = detect.data2[index]['package_type']
        row_data = [pdf_name, pdf_page_count + 1, package_type]

        if isinstance(reco.result, dict):
            data = [reco.result.get(f, []) for f in Table_type]
        else:
            data = (reco.result + [[]] * len(Table_type))[:len(Table_type)]

        # 对每个字段的值去掉首元素，转字符串，不足补空
        row_data.extend([str(v[1:]) if isinstance(v, list) and len(v) > 1 else ""
                         for v in data])

        # 写入CSV文件
        with open('bga_result.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 如果是第一次写入，先写表头（增加package_type字段）
            if os.path.getsize('bga_result.csv') == 0:
                header = ['PDF名称', '页码', 'package_type'] + Table_type
                writer.writerow(header)

            # 提取每个字段去掉第一个元素后的列表（增加package_type）
            row = [pdf_name, pdf_page_count + 1, package_type]
            for values in reco.result:
                trimmed = values[1:]  # 去掉第一个元素
                row.append(str(trimmed))  # 转为字符串写入
            writer.writerow(row)
        # ------------------新增：保存单个BGA封装为JSON文件---------------------------
        # 构建JSON数据（包含完整信息）
        bga_info = {
            "pdf_name": pdf_name,
            "page_number": pdf_page_count + 1,  # 保持和CSV中页码一致
            "package_type": package_type,
            # "raw_data": reco.result
            "processed_data": {  # 处理后的数据（去掉首元素）
                field: (values[1:] if isinstance(values, list) and len(values) > 1 else [])
                for field, values in zip(Table_type, data)
            }
        }
        result_folder = "bga_results"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        pdf_base = os.path.splitext(pdf_name)[0]
        json_filename = f"{pdf_base}_page_{pdf_page_count + 1}.json"
        json_path = os.path.join(result_folder, json_filename)

        # 写入JSON文件
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(bga_info, json_file, ensure_ascii=False, indent=4)
        print(f"已保存BGA封装信息到：{json_path}")

if __name__ == "__main__":
    completed_folder = 'completed_pdf'  # 存放已经测试过的文件
    if not os.path.exists(completed_folder):
        os.makedirs(completed_folder)

    # 从文件夹中获取所有PDF文件
    folder_path = 'bga'
    pdf_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf')]

    for pdf_path in pdf_paths:
        auto_detect_reco(pdf_path,test_type='BGA') # 测试什么类型就写什么类型-BGA、QFP、QFN

        # 构建目标路径
        destination_path = os.path.join(completed_folder, pdf_path)
        # 获取目标目录并确保其存在
        dest_dir = os.path.dirname(destination_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        # 移动文件
        shutil.move(pdf_path, destination_path)


