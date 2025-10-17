import fitz
import pdfplumber
import pandas as pd
import re
from sklearn.cluster import KMeans
import numpy as np


def extract_table_structure(TableImage):
    """
    输入表格图片，输出二维表格结构（无完全边框表格）
    Args:
        TableImage (np.ndarray): 原始图像
    Returns:
        list: 二维列表，表示识别出的表格内容
    """
    # 获取文本块信息
    blocks = get_text_blocks(TableImage)
    # 构建表格结构
    table = build_table_from_blocks(blocks)
    return table

def get_text_blocks(image):

    from packagefiles.TableProcessing.ocr_onnx.OCR_use import ONNX_Use
    words, coordinates = ONNX_Use(image, 'test')

    # 构造 blocks
    blocks = []
    for word, coord in zip(words, coordinates):
        text = word.strip()
        if text:
            x1, y1 = coord
            blocks.append({
                'text': text,
                'bbox': [x1, y1]
            })
    return blocks
def build_table_from_blocks(blocks, row_threshold=15):
    """
    表格构建算法，使用列中心聚类
    """
    if not blocks:
        return []

    # 按Y坐标排序
    blocks.sort(key=lambda b: b['bbox'][1])

    # 自适应行阈值计算
    y_coords = [b['bbox'][1] for b in blocks]
    y_diffs = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]
    if y_diffs:
        avg_diff = sum(y_diffs) / len(y_diffs)
        row_threshold = max(row_threshold, avg_diff * 0.5)

    # 行分组
    rows = []
    current_row = [blocks[0]]
    for block in blocks[1:]:
        last_block = current_row[-1]
        if abs(block['bbox'][1] - last_block['bbox'][1]) < row_threshold:
            current_row.append(block)
        else:
            rows.append(current_row)
            current_row = [block]
    rows.append(current_row)

    # 提取所有X坐标用于聚类
    all_x = [b['bbox'][0] for b in blocks]
    X = np.array(all_x).reshape(-1, 1)

    # 聚类数 = 最大列数
    max_cols = max(len(row) for row in rows)
    kmeans = KMeans(n_clusters=min(max_cols, len(X)), random_state=0).fit(X)
    col_centers = sorted(kmeans.cluster_centers_.flatten())

    # 每行按列中心分配文本块，并填充 "_"
    aligned_table = []
    for row in rows:
        row.sort(key=lambda b: b['bbox'][0])
        row_dict = {}
        for block in row:
            x = block['bbox'][0]
            closest_col = min(col_centers, key=lambda c: abs(c - x))
            col_idx = col_centers.index(closest_col)
            row_dict[col_idx] = block['text']

        # 插入 "_" 表示空白单元格
        aligned_row = [row_dict.get(i, '_') for i in range(len(col_centers))]
        aligned_table.append(aligned_row)

    return aligned_table

# 获取以xList, yList构成的坐标集合围成的单元格的坐标、合并单元格
def get_cells_coordinate(xList, yList, HorizontalLine, VerticalLine):
    # 将需要合并的单元格存在mergeCellsList的对应位置
    def add_coordinate_in_mergeCellsLsit(cell_1, cell_2, mergeCellsList):
        if len(mergeCellsList) == 0:
            mergeCellsList.append([cell_1, cell_2])
            return mergeCellsList
        tag = 0
        for mergeCellList in mergeCellsList:
            if cell_1 in mergeCellList:
                mergeCellList.append(cell_2)
                tag = 1
                break
        if tag == 0:
            mergeCellsList.append([cell_1, cell_2])

        return mergeCellsList
    
    # 判断是否需要合并单元格，返回True表示需要合并
    def merge_cells(cell, VerticalLine, direction):
        if direction == 0:
            # 找到与cell右框线处于同于x轴的所有线的纵坐标，横坐标比较时无用，直接舍弃
            Lines = [(line[1],line[3]) for line in VerticalLine if line[0] == cell[2]]
            cellLine = (cell[1],cell[3])
            for line in Lines:
                if cellLine[0] >= line[0] and cellLine[1] <= line[1]:
                    return False
            return True
        else:
            # 找到与cell下框线处于同一y轴的所有线的横坐标，纵坐标比较时无用，直接舍弃
            Lines = [(line[0],line[2]) for line in VerticalLine if line[1] == cell[3]]
            cellLine = (cell[0],cell[2])
            for line in Lines:
                if cellLine[0] >= line[0] and cellLine[1] <= line[1]:
                    return False
            return True
    
    Tablerows = len(yList) - 1  # 行数
    Tablecols = len(xList) - 1  # 列数
    cells_coordinate = [[[xList[j],yList[i],xList[j+1],yList[i+1]] for j in range(Tablecols)]
                        for i in range(Tablerows)]
    mergeCellsList = []
    # 行合并
    for i in range(Tablerows):
        for j in range(Tablecols):
            cell = cells_coordinate[i][j]
            if j != Tablecols - 1 and merge_cells(cell, VerticalLine, 0):
                # print(f"第{i+1}行{j+1}列需要右合并")
                mergeCellsList = add_coordinate_in_mergeCellsLsit((i,j), (i,j+1), mergeCellsList)
            if i != Tablerows - 1 and merge_cells(cell, HorizontalLine, 1):
                # print(f"第{i+1}行{j+1}列需要下合并")
                mergeCellsList = add_coordinate_in_mergeCellsLsit((i,j), (i+1,j), mergeCellsList)

    # 根据索引合并单元格坐标
    for mergeCellList in mergeCellsList:
        # 不管是先行合并还是先列合并，最后一个进入列表的总是右下角的单元格
        cellStart = cells_coordinate[mergeCellList[0][0]][mergeCellList[0][1]]
        cellEnd = cells_coordinate[mergeCellList[-1][0]][mergeCellList[-1][1]]
        truePoint = [cellStart[0],cellStart[1],cellEnd[2],cellEnd[3]]
        # 将所有被合并的单元格的坐标全设置为整个合并后的单元格的坐标
        for (i,j) in mergeCellList:
            cells_coordinate[i][j] = truePoint

    return cells_coordinate

# 根据文字坐标填充表格
def get_texts_coordinate(pdfPath, pageNumber,tableCoordinate, cellsCoordinate):
    with fitz.open(pdfPath) as pdfDoc:
        words = []
        coordinates = []
        page = pdfDoc[pageNumber -1]  # 获取第一页
        blocks = page.get_text("dict", clip=tableCoordinate)['blocks']
        blocks_filt = [x for x in blocks if x['type'] == 0]
        for block in blocks_filt:  # iterate through the text blocks
            for line in block["lines"]:  # iterate through the text lines
                string = ''
                for span in line["spans"]:  # iterate through the text spans
                    string += span['text']
                    coordinate = [round(x) for x in span['bbox']]
                    core = [(coordinate[0]+coordinate[2])/2,(coordinate[1]+coordinate[3])/2]
                    core = [round(x) for x in core]
                if string == ' ':
                    continue
                words.append(string)
                coordinates.append(core)

    zipped_list = list(zip(words, coordinates))
    zipped_list = sorted(zipped_list, key=lambda x: x[1][0])
    zipped_list = sorted(zipped_list, key=lambda x: x[1][1])
    words, coordinates = zip(*zipped_list)
    words, coordinates = list(words), list(coordinates)

    table = [['' for _ in  range(len(cellsCoordinate[0]))] for _ in range(len(cellsCoordinate))]
    for i in range(len(cellsCoordinate)):
        for j in range(len(cellsCoordinate[0])):
            for index in range(len(coordinates)):
                text = words[index].replace(' ','')
                x,y = coordinates[index]
                x1,y1,x2,y2 = cellsCoordinate[i][j]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if table[i][j] == '':
                        table[i][j]+=text
                    else:
                        table[i][j]+='_'+ text
    return table

def get_texts_from_coordinate(pdfPath, pageNumber,tableCoordinate, cellsCoordinate):
    table = []
    try:
        with pdfplumber.open(pdfPath) as pdf :
            pdfPage = pdf.pages[pageNumber - 1]
            for i in range(len(cellsCoordinate)):
                rowText = []
                for j in range(len(cellsCoordinate[0])):
                    x1,y1,x2,y2 = cellsCoordinate[i][j]
                    text = pdfPage.within_bbox((x1,y1-4,x2,y2+4)).extract_text(y_tolerance=5).replace("\n",",").replace(" ",",")
                    rowText.append(text)
                table.append(rowText)
    except ValueError:
        table = []
    return table

# 读取不可编辑表格
def get_texts_UsingOcr(TableImage, tableCoordinate, cellsCoordinate):
    
    from packagefiles.TableProcessing.ocr_onnx.OCR_use import ONNX_Use
    words, coordinates = ONNX_Use(TableImage, 'test')

    for index in range(coordinates.__len__()):
        coordinates[index][0] = tableCoordinate[0]+round(coordinates[index][0]*2/4)
        coordinates[index][1] = tableCoordinate[1]+round(coordinates[index][1]*2/4)

    zipped_list = list(zip(words, coordinates))
    zipped_list = sorted(zipped_list, key=lambda x: x[1][0])
    zipped_list = sorted(zipped_list, key=lambda x: x[1][1])
    words, coordinates = zip(*zipped_list)
    words, coordinates = list(words), list(coordinates)

    table = [['' for _ in  range(len(cellsCoordinate[0]))] for _ in range(len(cellsCoordinate))]
    for i in range(len(cellsCoordinate)):
        for j in range(len(cellsCoordinate[0])):
            for index in range(len(coordinates)):
                text = words[index]
                x,y = coordinates[index]
                x1,y1,x2,y2 = cellsCoordinate[i][j]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if table[i][j] == '':
                        table[i][j]+=text
                    else:
                        table[i][j]+='_'+ text

    return table

# 得到单元格合并信息
def return_Merge_info(cellsCoordinate):
    index = 0
    Is_merge_Cell = [[-1 for _ in  range(len(cellsCoordinate[0]))] for _ in range(len(cellsCoordinate))]
    i = 0
    j = 0
    merge_unit = []
    while i < len(cellsCoordinate):
        j = 0
        while j < len(cellsCoordinate[0]):
            if Is_merge_Cell[i][j] >= 0 :
                j += 1
                continue 
            # 横着找到合并边界
            HoriLen = 1
            while j + HoriLen <=  len(cellsCoordinate[0]) - 1:
                if cellsCoordinate[i][j] == cellsCoordinate[i][j+HoriLen]:
                    HoriLen += 1
                else:
                    break
            # 纵向找到合并边界
            VertLen = 1
            while i + VertLen <=  len(cellsCoordinate) -1:
                if cellsCoordinate[i][j] == cellsCoordinate[i+VertLen][j]:
                    VertLen += 1
                else:
                    break
            # 没有需要合并的单元格就开始检测下一个单元格
            if HoriLen == 1 and VertLen == 1:
                j+=1
            else:
                for tmp in range(VertLen):
                    Is_merge_Cell[i+tmp][j:j+HoriLen] = [index for _ in range(HoriLen)]
                merge_unit.append([[x,y] for y in range(j,j+HoriLen) for x in range(i,i+VertLen)])
                index += 1
                j+=HoriLen
        i += 1
    return Is_merge_Cell, merge_unit

# 特定情况拆分单元格
def split_cell(table, cellsCoordinate):
    Is_merge_Cell, merge_unit = return_Merge_info(cellsCoordinate)
    for unit in merge_unit:
        cell_0 = unit[0]
        Merge_cell_Num = len(unit)
        string = table[cell_0[0]][cell_0[1]]
        cell_split = string.split('_')
        if cell_split.__len__() != Merge_cell_Num:
            for cell in unit:
                table[cell[0]][cell[1]] = string.replace('_','')
        else:
            for index in range(unit.__len__()):
                cell = unit[index]
                table[cell[0]][cell[1]] = cell_split[index]
    return table

# 根据表头信息判断是否是封装表格
def judge_from_title(table, title_keyword_list):
    count = 0
    # 先判断前面几行
    limit = 4 if len(table) > 2 else 2
    for i in range(limit):
        for j in range(table[0].__len__()):
            if table[i][j] == None or table[i][j].__len__()>20:
                continue
            data = table[i][j].upper().replace(" ","").replace("\n","").replace(",","")
            if any(item in data for item in title_keyword_list) :
                count+=1
    # 在判断前面几列
    limit = 4 if len(table[0]) > 2 else 2
    for i in range(limit):
        for i in range(table.__len__()):
            if table[i][j] == None or table[i][j].__len__()>20:
                continue
            data = table[i][j].upper().replace(" ","").replace("\n","").replace(",","").replace('D','O')
            if any(item in data for item in title_keyword_list) :
                count+=1
    # 根据是否出现这几个关键字以及出现的数量进行判断是否是封装表格
    if count >= 2:
        return True
    else:
        return False

# 根据内容信息判断是否是封装表格
def judge_from_context(table, parameter_list):
    count = 0
    for i in range(len(table)):
        for j in range(len(table[0])):
            data = table[i][j].upper().replace(" ","").replace("\n","").replace(",","").replace('D','O')
            # 出现关键字
            if any(item in data for item in parameter_list):
                flag_1 = i+1<len(table) and len(re.findall(r'\d+\.\d+',table[i+1][j])) == 1
                flag_2 = j+2<len(table[0]) and (len(re.findall(r'\d+\.\d+',table[i][j+1])) == 1 or len(re.findall(r'\d+\.\d+',table[i][j+2])) == 1)
                # 并且后面跟着数字
                if flag_1 or flag_2:
                    count += 1
    if count >= 2 and count < 10:
        return True
    else:
        return False
# 判断需要旋转，以及旋转方向
def Is_Common_package(table):
    JudgeList = ['A','A1','A2','D','E','e','D1','E1']
    count = 0
    # 判断第一列是否是符合标准的，符合就不旋转
    for row in table:
        for tag in JudgeList:
            if tag in row[0]:
                count += 1
                break
    if count > 3:
        return 0
    # 判断是否能找到符合条件的行
    else:
        for index in range(len(table)):
            count = 0
            row = table[index]
            matched_elements = set()  # 用于跟踪已经匹配的元素
            for cell in row:
                for element in JudgeList:
                    if element in cell and element not in matched_elements:
                        matched_elements.add(element)
                        count += 1
            if count > 3:
                if index == 0:
                    return 0
                if index >= len(table)/2:
                   return 90
                # 转表不转图
                elif index == len(table)/2 - 1:
                    return -1
        # 没找到就直接顺时针旋转270，即逆时针旋转90
        return 270
# 转表不转图
def rotate_table(table):
    tmp = [[table[i][j] for i in range(len(table))] for j in range(len(table[0]))]
    return tmp
# 表格坐标变换
def Table_coordinate_transformation(tableCoordinate, cellsCoordinate, image , direction):
    if direction == 90:
        height =  image.shape[0]
        for i in range(cellsCoordinate.__len__()):
            for j in range(cellsCoordinate[0].__len__()):
                x1,y1,x2,y2 = cellsCoordinate[i][j]
                cellsCoordinate[i][j] = [height - y2, x1, height - y1, x2]
        tmp = cellsCoordinate
        cellsCoordinate = [[tmp[tmp.__len__() - 1-j][i] for j in  range(tmp.__len__())] for i in range(tmp[0].__len__())]
        x1,y1,x2,y2 = tableCoordinate
        tableCoordinate = [height - y2, x1, height - y1, x2]
    elif direction == 270:
        width = image.shape[1]
        for i in range(cellsCoordinate.__len__()):
            for j in range(cellsCoordinate[0].__len__()):
                x1,y1,x2,y2 = cellsCoordinate[i][j]
                cellsCoordinate[i][j] = [y1, width - x2, y2, width - x1]
        tmp = cellsCoordinate
        cellsCoordinate = [[tmp[j][tmp[0].__len__()-1 - i] for j in  range(tmp.__len__())] for i in range(tmp[0].__len__())]
        x1,y1,x2,y2 = tableCoordinate
        tableCoordinate = [y1, width - x2, y2, width - x1]
    return tableCoordinate, cellsCoordinate

# 含unit的表格处理
def get_data_from_unit_table(table, unit):
    data = []
    tableCols = len(table[0])
    table = [list(i) for i in zip(*table)]
    for index in range(len(unit)):
        col,row = unit[index]
        for i in range(row+1,tableCols):
            tmp = table[i][col+1].split('_')
            tmp = [x.replace(' ','') for x in tmp]
            if len(tmp) == 1:
                tmp.append(tmp[0])
                tmp.append(tmp[0])
            elif len(tmp) == 2:
                tmp.append('-')
            data.append(
                [
                    table[i][col].replace(' ',''),
                    tmp[0],
                    tmp[2],
                    tmp[1]
                ]
                )
    return data

# 找到min nom max所在列
def find_MIN_NOM_MAX(table):
    cols = []
    title = 0
    limit = 4 if len(table) > 2 else 2
    for i in range(limit):
        for j in range(table[0].__len__()):
            if table[i][j] == None or table[i][j].__len__()>20:
                continue
            data = table[i][j].upper().replace(" ","").replace("\n","").replace(",","")
            if any(item in data for item in ['NOM','MOM','MAX','MIN','TYP','MN','最小','最大','公称']) and j not in cols:
                cols.append(j)
                title = i+1
    if cols.__len__() == 1:
        table = [list(row) for row in zip(*table)]
        cols = []
        for i in range(limit):
            for j in range(table[0].__len__()):
                if table[i][j] == None or table[i][j].__len__()>20:
                    continue
                data = table[i][j].upper().replace(" ","").replace("\n","").replace(",","")
                if any(item in data for item in ['NOM','MOM','MAX','MIN','MN','TYP','最小','最大','公称']) and j not in cols:
                    cols.append(j)
                    title = i+1
                if data == 'A':
                    tmp = j
        if cols.__len__() == 3:
            # 无论怎么排布保证字母在数字前一列
            if tmp != cols[0] - 1:
                for row in table:
                    # 交换指定列的元素  
                    row[cols[0] - 1], row[tmp] = row[tmp], row[cols[0] - 1]
            return title, [cols], table
    if cols.__len__() == 3:
        return title, [cols], table
    elif cols.__len__() == 6 and cols[3]== cols[2] + 1:
        return title, [cols[:3]], table
    elif cols.__len__() == 6 and cols[3]!= cols[2] + 1:
        return title, [cols[:3],cols[3:]], table
    elif cols.__len__() == 4 and cols[2]== cols[1] + 1:
        return title, [cols], table
    elif cols.__len__() == 2:
        return title, [cols], table
    else:
        return 0,[], table

def delete_space_row(data):
    # 删除全为空字符串的行
    non_empty_rows = [row for row in data if any(cell != '' for cell in row)]
    
    transposed = list(zip(*non_empty_rows))  # 转置
    non_empty_cols = [col for col in transposed if any(cell != '' for cell in col)]
    
    # 再次转置回来得到最终的结果
    result = list(zip(*non_empty_cols))  # 转置回原始形状
    result = [list(row) for row in result]  # 将元组转换为列表
    return result


# 根据数值找到数据所在列
def find_number_col(table):
    cols = []
    title = []
    # 遍历每一列
    for i in range(table[0].__len__()):
        tmp = 0
        count = 0
        for j in range(table.__len__()):
            if bool(re.match(r'^[a-zA-Z][0-9]$', table[j][i])):
                continue
            if len(re.findall("\d+",table[j][i])) > 0 or table[j][i]=='' or table[j][i]=='-'or table[j][i]=='--':
                count += 1
                tmp = j if tmp == 0 else tmp
        if count/table.__len__() > 0.5:
             cols.append(i)
             title.append(tmp)
    title = min(title)
    if cols.__len__() == 3:
        return title, [cols]
    elif cols.__len__() == 6 and cols[3]== cols[2] + 1:
        return title, [cols[:3]]
    elif cols.__len__() == 6 and cols[3]!= cols[2] + 1:
        return title, [cols[:3],cols[3:]]
    elif cols.__len__() == 4 and cols[2]== cols[1] + 1:
        return title, [cols]
    elif cols.__len__() == 1:
        return title, [cols]

# 常规表格处理
def get_data_from_common_table(table):
    data = []
    table = delete_space_row(table)
    # 找到目标值所在行列，一般会有MIN NOM MAX等标识
    title, Paircols, table = find_MIN_NOM_MAX(table)
    if Paircols == []:
        title, Paircols = find_number_col(table)
    tableRows = len(table)
    # print(table)
    # 可能会有分成6列是并列关系的存在
    for cols in Paircols:
        for i in range(title,tableRows):
            if cols.__len__() == 3:
                if 'D1,E1' in table[i][cols[0]-1]:
                    table[i][cols[0]-1] = table[i][cols[0]-1].replace('D1,E1','D1E1')
                data.append(
                    [
                        table[i][cols[0]-1], 
                        table[i][cols[0]], 
                        table[i][cols[1]], 
                        table[i][cols[2]]
                    ]
                    )
            elif cols.__len__() == 1:
                
                data.append(
                    [
                        table[i][cols[0]-1], 
                        table[i][cols[0]]
                    ]
                    )
            elif cols.__len__() == 2:
                data.append(
                    [
                        table[i][cols[0]-1], 
                        table[i][cols[0]], 
                        '',
                        table[i][cols[1]]
                    ]
                    )
            else:
                    data.append(
                    [
                        table[i][cols[0]-1], 
                        table[i][cols[0]], 
                        table[i][cols[1]], 
                        table[i][cols[2]], 
                        table[i][cols[3]], 
                    ]
                    )
    count_same = 1
    for index in range(data.__len__()-1):
        if data[index][0] == data[index+1][0]:
            count_same += 1
    if count_same > 7 and count_same == len(data[0][0].split('_')):
        tmp = data[0][0].split('_')
        for rowIndex in range(count_same):
            data[rowIndex][0] = tmp[rowIndex]
    # if data[0][0].split(',').__len__() == data.__len__():
    #     tmp = data[0][0].split(',')
    #     for rowIndex in range(data.__len__()):
    #         data[rowIndex][0] = tmp[rowIndex]

    return data

# 从table中提取出需要的行和列 
def get_info_from_table(table):
    if table == []:
        return table
    unitTag = []
    symbolTag = []
    result = []
    tableCols = len(table[0])
    for i in range(2):
        for j in range(tableCols):
            tmp = table[i][j].upper().replace(" ","").replace("\n","").replace(",","")
            if 'UNIT' in tmp or 'UNT' in tmp:
                unitTag.append([i,j])
            if 'SYMBOL' in tmp or 'MIN' in tmp or 'NOM' in tmp or 'MAX' in tmp and '_MAX' not in tmp:
                symbolTag.append([i,j])
    if unitTag != [] and symbolTag == [] :
        result = get_data_from_unit_table(table, unitTag)
    # 多列
    else:
        result = get_data_from_common_table(table)

    return result
 
 # 对OCR提取的结果进行过滤

def filt_KeyInfo_data(lst):
    if lst.__len__() == 1:
        number = re.findall("[1-9]\d*.\d*|0\.\d*[1-9]\d*", lst[0])
        number = [float(x) for x in number]
        if len(number) == 1:
            return [number[0],number[0],number[0]]
        elif len(number) == 2:
            if number[0] - number[1] < 0:
                return [number[0],(number[0]+number[1])/2,number[1]]
            else:
                return [number[0]- number[1],number[0],number[1]+number[0]]
    data = []
    tmp = []
    count = 0
    # print(lst)
    lst = [str(num.replace(',', '.')) for num in lst]
    for index in range(lst.__len__()):
        try:
            # 表格读取代码不通用
            if len(lst[index].split('_')) >= 3 and len(re.findall("[1-9]\d*.\d*|0\.\d*[1-9]\d*", lst[index])) == 3:
                tmp = [lst[index].split('_')[0],lst[index].split('_')[1],lst[index].split('_')[2]]
                tmp = [float(x) for x in tmp]
                break
            lst[index] = lst[index].replace('B','').replace('S','').replace('C.','').replace(',','').replace('C','')
            str_list = re.findall("[1-9]\d*.\d*|0\.\d*[1-9]\d*", lst[index])
            number = ''
            for x in str_list:
                number += x
            tmp.append(float(number))
        except:
            count+=1
            tmp.append(0)
    if count < 2:
        if tmp[0] == 0:
            data = [max(0,round(2*tmp[1]-tmp[2],2)),tmp[1],tmp[2]]
        elif tmp[1] == 0:
            data = [tmp[0], round((tmp[0]+tmp[2])/2,2), tmp[2]]
        elif tmp[2] == 0:
            data = [tmp[0],tmp[1],round(2*tmp[1] - tmp[0],2)]
        else:
            data = [tmp[0],tmp[1],tmp[2]]
    elif count == 2:
        singleNumber = [x for x in tmp if x!=0][0]
        data = [singleNumber,singleNumber,singleNumber]
    else:
        data = ['','','']

    return data

def table_checked(table):
    # 遍历前两行
    for i in range(min(2, len(table))):
        row = table[i]
        # 查找包含 MIN, NOM, MAX 的列索引
        for j in range(len(row)):
            cell = row[j].strip().upper() if row[j] else ""
            if ("MIN" in cell) or ("MN" in cell):
                row[j] = "min"
                if j + 2 < len(row):
                    row[j + 1] = "nom"
                    row[j + 2] = "max"
                break
            elif "NOM" in cell:
                if j - 1 >= 0:
                    row[j - 1] = "min"
                row[j] = "nom"
                if j + 1 < len(row):
                    row[j + 1] = "max"
                break
            elif "MAX" in cell:
                if j - 2 >= 0:
                    row[j - 2] = "min"
                    row[j - 1] = "nom"
                row[j] = "max"
                break
                
    return table

# 根据标准字母语义从表中找出信息进行对应
def add_info_from_KeyInfo(data, KeyInfo,packageType):
    # BGA类型的语义匹配
    def BGA_add_info(data, row):
        row = [s.strip() for s in row]
        # print(row)
        row[0] = row[0].replace('_','')
        if row[0] == '0' or row[0] == '9' or row[0] == 'Φb':
            row[0] = 'b'
        row[0] = row[0].split(',')[0].replace(' ','')
        row = [x.replace('.BSC','').replace(',BSC','').replace('8SC','') for x in row]
        # 列Pitch
        if ('e' in row[0] or '0' in row[0]) and data[11][1] == '':
            data[11][1:] = filt_KeyInfo_data(row[1:])
        # 行Pitch
        if ('e' in row[0] or '0' in row[0]) and data[10][1] == '':
            data[10][1:]  = filt_KeyInfo_data(row[1:])
        # 列数
        # if row[0] =='eD' and data[8][1] == '':
        #     data[8][1:]  = filt_KeyInfo_data(row[1:])
        # 行数
        # if (row[0] =='e' or row[0]=='eE' or row[0] == 'e0')and data[7][1] == '':
        #     data[7][1:]  = filt_KeyInfo_data(row[1:])
        # # 列数
        # if ('MD' in row[0] or row[0] == 'M') and data[6][1] == '':
        #     data[6][1:] = filt_KeyInfo_data(row[1:])
        # 行数
        # if ('ME' in row[0] or row[0] == 'M') and data[5][1] == '':
        #     data[5][1:] = filt_KeyInfo_data(row[1:])
        # 球直径
        if 'b' in row[0] and data[4][1] == '':
            data[4][1:]  = filt_KeyInfo_data(row[1:])
        # 支撑高
        if ('A1' in row[0]) and data[3][1] == '':
            data[3][1:]  = filt_KeyInfo_data(row[1:])
        # 实体高
        if row[0]=='A'and data[2][1] == '':
            data[2][1:] = filt_KeyInfo_data(row[1:])
        # 实体宽
        if ('E' in row[0] or 'DE' in row[0]) and data[1][1] == '':
            # row[1:] = [f"{float(num)/10:.1f}" if int(num) > 100 else num for num in row[1:]]
            data[1][1:]  = filt_KeyInfo_data(row[1:])
        # 实体长
        if ('D' in row[0] or 'DE' in row[0]) and data[0][1] == '':
            data[0][1:]  = filt_KeyInfo_data(row[1:])
    # QFP类型的语义匹配
    def QFP_add_info(data, row):
        row = [s.strip() for s in row]
        # print(row)
        row[0] = row[0].replace('_','')
        if row[0] == '0' or row[0] == '9' or row[0] == 'Φb':
            row[0] = 'b'
        row[0] = row[0].split(',')[0].replace(' ','')
        row = [x.replace('.BSC','').replace(',BSC','').replace('8SC','') for x in row]
        # 引脚的厚度
        if 'c' in row[0] and data[12][1] == '':
            data[12][1:]  = filt_KeyInfo_data(row[1:])
        # 散热盘长
        if row[0] =='E2' and data[11][1] == '':
            data[11][1:]  = filt_KeyInfo_data(row[1:])
        # 散热盘长
        if row[0] =='D2' and data[10][1] == '':
            data[10][1:]  = filt_KeyInfo_data(row[1:])
        # 行/列Pin数
        if (row[0] =='e' or row[0]=='eE' or row[0] == 'e0')and data[9][1] == '':
            data[9][1:]  = filt_KeyInfo_data(row[1:])
        # pin宽
        if 'b' in row[0] and data[8][1] == '':
            data[8][1:]  = filt_KeyInfo_data(row[1:])
        # pin长
        if 'L' in row[0] and data[7][1] == '':
            data[7][1:]  = filt_KeyInfo_data(row[1:])
        # 外围宽
        if 'E' in row[0] and data[6][1] == '':
            data[6][1:]  = filt_KeyInfo_data(row[1:])
        # 外围长
        if 'D' in row[0] and data[5][1] == '':
            data[5][1:]  = filt_KeyInfo_data(row[1:])   
        # 端子高
        if 'A3' in row[0] and data[4][1] == '':
            data[4][1:]  = filt_KeyInfo_data(row[1:])
        # 支撑高
        if ('A1' in row[0]) and data[3][1] == '':
            data[3][1:]  = filt_KeyInfo_data(row[1:])
        # 实体高
        if row[0]=='A'and data[2][1] == '':
            data[2][1:] = filt_KeyInfo_data(row[1:])
        # 实体宽
        if 'E1' in row[0] and data[1][1] == '':
            # row[1:] = [f"{float(num)/10:.1f}" if int(num) > 100 else num for num in row[1:]]
            data[1][1:] = filt_KeyInfo_data(row[1:])
        # 实体长
        if 'D1' in row[0] and data[0][1] == '':
            data[0][1:] = filt_KeyInfo_data(row[1:])
    if KeyInfo == []:
        return KeyInfo
    for key in KeyInfo :
        for i, item in enumerate(key):
            # 检查是否以小数点开头并紧接数字
            if item.startswith('.') and len(item) > 1 and item[1:].isdigit():
                value_str = '0' + item  # 在小数前加0
                try:
                    value_inch = round(float(value_str) * 25.4, 2)
                    key[i] = str(value_inch)
                except ValueError:
                    continue
        print(key)
    # print(KeyInfo)
    if packageType == 'BGA':
        # print(KeyInfo)
        for row in KeyInfo:
            try:
                BGA_add_info(data,row)
            except:
                continue
    elif packageType == 'QFP':
        for row in KeyInfo:
            try:
                QFP_add_info(data,row)
            except:
                continue
    elif packageType == 'QFN':
        for row in KeyInfo:
            row = [s.strip() for s in row]
            # print(row)
            row[0] = row[0].replace('_','')
            if row[0] == '0' or row[0] == '9' or row[0] == 'Φb':
                row[0] = 'b'
            try:
                row[0] = row[0].split(',')[0].replace(' ','')
                row = [x.replace('.BSC','').replace(',BSC','').replace('8SC','') for x in row]
                # Pin_Pitch
                if 'e' in row[0] and data[0][1] == '':
                    data[0][1:] = filt_KeyInfo_data(row[1:])
                if 'e' in row[0] and data[1][1] == '':
                    data[1][1:] = filt_KeyInfo_data(row[1:])

                # 散热盘长
                if row[0] =='D2' and data[11][1] == '':
                    data[11][1:]  = filt_KeyInfo_data(row[1:])
                # 列Pin数
                if row[0] =='eD' and data[10][1] == '':
                    data[10][1:]  = filt_KeyInfo_data(row[1:])
                # 行Pin数
                if (row[0]=='eE' or row[0] == 'e0')and data[9][1] == '':
                    data[9][1:]  = filt_KeyInfo_data(row[1:])
                # 列数
                if ('MD' in row[0] or row[0] == 'M') and data[8][1] == '':
                    data[8][1:] = filt_KeyInfo_data(row[1:])
                # 行数
                if ('ME' in row[0] or row[0] == 'M') and data[7][1] == '':
                    data[7][1:] = filt_KeyInfo_data(row[1:])
                # pin宽
                if 'b' in row[0] and data[6][1] == '':
                    data[6][1:]  = filt_KeyInfo_data(row[1:])
                # pin长
                if 'L' in row[0] and data[5][1] == '':
                    data[5][1:]  = filt_KeyInfo_data(row[1:])
                # 端子高
                if 'A3' in row[0] and data[4][1] == '':
                    data[4][1:]  = filt_KeyInfo_data(row[1:])
                # 支撑高
                if ('A1' in row[0]) and data[3][1] == '':
                    data[3][1:]  = filt_KeyInfo_data(row[1:])
                # 实体高
                if row[0]=='A'and data[2][1] == '':
                    data[2][1:] = filt_KeyInfo_data(row[1:])
                # 实体宽
                if 'E' in row[0] and data[1][1] == '':
                    # row[1:] = [f"{float(num)/10:.1f}" if int(num) > 100 else num for num in row[1:]]
                    data[1][1:]  = filt_KeyInfo_data(row[1:])
                # 实体长
                if 'D' in row[0] and data[0][1] == '':
                    data[0][1:]  = filt_KeyInfo_data(row[1:])
            except:
                continue
    elif packageType == 'SON' or packageType == 'SOP':
        for row in KeyInfo:
            row = [s.strip() for s in row]
            # print(row)
            row[0] = row[0].replace('_','')
            if row[0] == '0' or row[0] == '9' or row[0] == 'Φb':
                row[0] = 'b'
            try:
                row[0] = row[0].split(',')[0].replace(' ','')
                row = [x.replace('.BSC','').replace(',BSC','').replace('8SC','') for x in row]
                # pin宽
                if 'b' in row[0] and data[4][1] == '':
                    data[4][1:]  = filt_KeyInfo_data(row[1:])
                # pin长
                if ('L' in row[0]) and data[3][1] == '':
                    data[3][1:]  = filt_KeyInfo_data(row[1:])
                # 实体高
                if row[0]=='A'and data[2][1] == '':
                    data[2][1:] = filt_KeyInfo_data(row[1:])
                if row[0] == 'Amax.' and data[2][1] == '':
                    data[2][1:] = filt_KeyInfo_data(row[1:])
                if row[0] == 'A(1)max.' and data[2][1] == '':
                    data[2][1:] = filt_KeyInfo_data(row[1:])
                # 实体宽
                if 'E' in row[0] and data[1][1] == '':
                    # row[1:] = [f"{float(num)/10:.1f}" if int(num) > 100 else num for num in row[1:]]
                    data[1][1:]  = filt_KeyInfo_data(row[1:])
                # 实体长
                if 'D' in row[0] and data[0][1] == '':
                    data[0][1:]  = filt_KeyInfo_data(row[1:])
                # Pin_Pitch
                if 'e' in row[0] and data[9][1] == '':
                    data[9][1:] = filt_KeyInfo_data(row[1:])
            except:
                continue
    return data