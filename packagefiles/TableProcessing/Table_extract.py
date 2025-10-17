#F3.表格内容解析与判断 F4.表格规范化流程
from packagefiles.TableProcessing.Table_function.GetTable import *

def extract_table(pdfPath, page_Number_List, Table_Coordinate_List, packageType, page_num):
    """
    传入表格坐标，获得表格信息
    :param pdfPath: pdf路径
    :param page_Number_List: 存在表格页
    :param Table_Coordinate_List: 表格坐标
    :return: 当前表格信息
    """
    Table = []
    Type = []
    Integrity = []
    print(f'文件路径：{pdfPath}\n', f'存在表格页：{page_Number_List}\n', f'表格对应坐标：{Table_Coordinate_List}\n', f'封装类型：{packageType}')
    for pageNumber,TableCoordinate in zip(page_Number_List,Table_Coordinate_List):
        # 读取当前页的表格内容
        if TableCoordinate == []:
            Table.append([])
            Type.append(False)
            Integrity.append(False)
            continue

        table = get_table(pdfPath, pageNumber, TableCoordinate)
        Table.append(table)
        # 判断当前页表格是否是封装表``
        Type.append(judge_if_package_table(table, packageType))
        # 判断当前页封装表是否完全
        Integrity.append(judge_if_complete_table(table, packageType))
        print("##")
    # 根据封装表是否完整进行合并
    table = table_Select(Table, Type, Integrity)
    table = table_checked(table)
    # 提取表内信息
    data = postProcess(table, packageType, page_num)

    return data
