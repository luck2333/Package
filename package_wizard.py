
import os
import shutil
import csv  # 导入csv模块
import json

import copy

from packagefiles.UI.detect import Detect,PreProcess
from packagefiles.Segment.Segment_function import get_type,increment
from packagefiles.Segment.Package_pretreat import reco_package
from packagefiles.UI.ui_class import RecoTableThread



def auto_detect_package(pdf_path):
     """
     自动检测目标路径PDF，找出所含封装页
     :param pdf_path: pdf路径
     """
     page = PreProcess(pdf_path).page_list
     detect = Detect(pdf_path, page)
     detect.pre_process()
     detect.process()
     detect.post_process()

     return detect

def auto_reco_package(pdf_path, package_data,current_index):
     """
     自动识别PDF手册目标页封装图
    :param pdf_path: pdf路径
    :param package_data: 封装数据，该数据为auto_detect_package函数产生的task.data2
    :param current_index: PDF中存在的第current_index个封装图
    :return:result当前封装的提取信息,package_type当前封装类型
     """
     #pdf_path, pdf_page_count, current_package, package
     # 对封装信息进行处理
     package1 = copy.deepcopy(package_data)
     package1 = increment(package1)
     # 识别参数
     current_package = package1[current_index]
     current_page = current_package['page']
     # 封装类型
     package_type = current_package['package_type']
     out_put = {}
     # try:
     out_put= reco_package(package_type,current_package,current_page,pdf_path)
     # except Exception as e:
     #      print(f'识别出错:{e}')

     return out_put,package_type



if __name__ == '__main__':
     pdf_path = r"4091fa.pdf"
     detect = auto_detect_package(pdf_path)
     package_data = detect.data2
     for index in range(0,len(package_data)):
          package_result,package_type = auto_reco_package(pdf_path, package_data, index)



