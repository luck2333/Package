from packagefiles.TableProcessing.Table_function.Tool import *
from packagefiles.TableProcessing.Table_function.getBoder import *
from packagefiles.TableProcessing.Table_function.dataProcess import *
from packagefiles.TableProcessing.Table_function.table_detect import find_best_match
# 提取封装表格


def get_table(pdfPath, pageNumber, Coordinate):
    scale = 2
    direction = -1
# 页面转图片
    try:
        image = pdf2img_WithoutText(pdfPath, pageNumber, scale)
        tableCoordinate = [round(x * scale) for x in Coordinate]
        xList, yList, HorizontalLine, VerticalLine = get_Border(image, tableCoordinate)
    except:
        image = pdf2img_WithText(pdfPath, pageNumber, scale)
        tableCoordinate = [round(x * scale) for x in Coordinate]
        xList, yList, HorizontalLine, VerticalLine = get_Border(image, tableCoordinate)
    # 判断矩形框区域是否为全白
#     if (is_all_white(image, Coordinate)):
#         image = pdf2img_WithText(pdfPath, pageNumber, scale)
#     tableCoordinate = [round(x*scale) for x in Coordinate]
# # F4.4 找到图中除开文字的所有框线,并提取表格内容
#     xList, yList, HorizontalLine, VerticalLine = get_Border(image, tableCoordinate)
    # 得到所有单元格的坐标
    cellsCoordinate = get_cells_coordinate(xList, yList, HorizontalLine, VerticalLine)
    # 单元格可视化
    # show_each_retangle(image, cellsCoordinate)
    # 单元格坐标转换为放大前的大小

    with fitz.open(pdfPath) as doc:
        page = doc.load_page(pageNumber-1)
        # 获取页面上的文本块
        blocks = page.get_text("blocks", clip=Coordinate)

    # 读取表格内容
    if blocks.__len__() < 5:
        TableImage = Get_Ocr_TableImage(pdfPath, pageNumber,Coordinate)
        
        table = get_texts_UsingOcr(TableImage, tableCoordinate, cellsCoordinate)
        rotate = Is_Common_package(table)
        # 转表不转图
        if rotate == -1:
            table = rotate_table(table)
        if rotate == 90:
            TableImage = cv2.rotate(TableImage, cv2.ROTATE_90_CLOCKWISE)
            tableCoordinate, cellsCoordinate = Table_coordinate_transformation(tableCoordinate, cellsCoordinate, image, direction = 90)
            table = get_texts_UsingOcr(TableImage, tableCoordinate, cellsCoordinate)
        elif rotate == 270:
            TableImage = cv2.rotate(TableImage, cv2.ROTATE_90_COUNTERCLOCKWISE) 
            tableCoordinate, cellsCoordinate = Table_coordinate_transformation(tableCoordinate, cellsCoordinate, image , direction = 270)
            table = get_texts_UsingOcr(TableImage, tableCoordinate, cellsCoordinate)
            if Is_Common_package(table) == 270:
                return []
    else:
        for i in range(len(cellsCoordinate)):
            for j in range(len(cellsCoordinate[0])):
                cellsCoordinate[i][j] = [round(x/scale) for x in cellsCoordinate[i][j]]

        table = get_texts_coordinate(pdfPath, pageNumber, Coordinate, cellsCoordinate)
        # 如果表格的编码格式有问题，转到特殊表格去识别
        found = False
        for row in table:
            for cell in row:
                if isinstance(cell, str) and any(char in cell for char in ['\\', '\x0c', '\x1d']):
                    found = True
                    break
        if found == True:
            TableImage = Get_Ocr_TableImage(pdfPath, pageNumber, Coordinate)
            table = extract_table_structure(TableImage)
    # 对没有框线分割但实际上需要进行单元格拆分的情况进行处理
    table = split_cell(table, cellsCoordinate)
    return table

# 根据字符判断是否是封装表格
def judge_if_package_table(table, packageType):
    if table == []:
        return False
    title_keyword_list = ['NOM','MAX','MIN','TYP','最小','最大','公称']
    if packageType == 'BGA':
        data = ['D','E','A','A1','b','e','e1','SD','SE']
    if packageType == 'QFN':
        data = ['D','E','A','A1','A3','L','b','e','D2','E2']
    if packageType == 'QFP':
        data = ['D1','E1','A','A1','A3','L','b','e','D2','E2','c']
    if packageType == 'SON' or packageType == 'SOP':
        data = ['D','E','A','L','b']
    # 进行两次判断，先判断根据关键词判断，再根据数值进行判断
    if judge_from_title(table, title_keyword_list) or judge_from_context(table, data):
        return True
    else:
        return False

# 判断表格是否完整
def judge_if_complete_table(table, packageType):
    if table == []:
        return False
    if packageType == 'BGA':
        data = ['D','E','A','A1','b','e']
    elif packageType == 'QFN':
        data = ['D','E','A','A1','L','b','e']
    elif packageType == 'QFP':
        data = ['D1','E1','A','A1','L','b','e','D2','E2']
    elif packageType == 'SOP' or packageType == 'SON':
        data = ['D','E','A','L','b']

    count = 0
    for j in range(len(table[0])):
        for i in range(len(table)):
            # 如果table[i][j]有数字编号带括号，将括号和内容删除
            if re.search(r'\(\d+\)', str(table[i][j])):
                table[i][j] = re.sub(r'\(\d+\)', '', str(table[i][j]))
            if table[i][j] == 'eD' or table[i][j] == 'eE': count += 1
            if table[i][j] == 'Φb': count += 1
            if table[i][j] == 'DE': count += 4
            if table[i][j] == 'D/E': count += 2
            if table[i][j] == '':
                continue
            if any(table[i][j] in item for item in data):
                count+=1
    if count>=len(data):
        return True
    else:
        return False
    
# 将表与其续表进行合并
def table_Merge(Table_1, Table_2):
    col_table_1 = len(Table_1[0])
    col_table_2 = len(Table_2[0])
    # 列数相同就直接合并
    if col_table_1 == col_table_2:
        return Table_1+ Table_2
    # 续表与前一页表格列不相同就先拓展再合并
    elif col_table_1 > col_table_2:
        extend_table = [['' for _ in range(len(Table_1[0]))]for _ in range(len(Table_2))]
        # 对表进行填充
        for i in range(len(extend_table)):
            for j in range(len(extend_table[0])):
                extend_table[i][j] = Table_2[i][j] if j < col_table_2 else Table_2[i][col_table_2 - 1]
        return Table_1 + extend_table

# 根据信息判断是否需要进行表格合并
def table_Select(Table, Type, Integrity):
    if len(Table) == 1:
        return Table[0]
    # 当页表有效完全
    if Type[1] == True:
        # 当前页有效且完全
        if Integrity[1] == True:
            return Table[1]
        # 当前页有效但是不完全
        else:
            # 相邻页表均无效
            if Type[0] != True and Type[2] != True:
                return Table[1]
            # 相邻页的表有效
            else:
                # 上一页的表有效
                if Type[0] == True:
                    # 且完全
                    if Integrity[0] == True:
                        return Table[1]
                    # 不完全
                    else:
                        return table_Merge(Table[0], Table[1])
                # 下一页的表有效
                elif Type[2] == True:
                     # 且完全
                    if Integrity[2] == True:
                        return Table[1]
                    # 不完全
                    else:
                        return table_Merge(Table[1], Table[2])
                # 上一页、下一页均为有效表
                else:
                    return Table[1]
    # 当前页的表无效
    else:
        # 上一页有效且完全
        if Type[0] == True and Integrity[0] == True:
            return Table[0]
        # 下一页有效且完全
        elif Type[2] == True and Integrity[2] == True:
            return Table[2]
        else:
            return []
# 后处理
def get_nx_ny_from_title(page_num, nx, ny):
    import json
    if nx == '':
        nx = 0
    if ny == '':
        ny = 0
    pin_nums = ''
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

                    nx = str(result[4])
                    ny = str(result[5])
            pin_nums = result[0]

    return nx, ny,pin_nums
def postProcess(table, packageType, page_num):
    data = [['' for _ in  range(4)] for _ in range(13)]
    KeyInfo = get_info_from_table(table)
    data = add_info_from_KeyInfo(data,KeyInfo,packageType)
    if packageType == 'BGA':
        print("开始提取title参数", data)
        data[5][2], data[6][2],_ = get_nx_ny_from_title(page_num, data[5][2], data[6][2])
        data[5][1] = data[5][2]
        data[5][3] = data[5][2]
        data[6][1] = data[6][2]
        data[6][3] = data[6][2]
        print("提取title参数完毕", data)
        # # 20250723改变顺序
        new_result_list = []
        new_result_list.append(data[11])
        new_result_list.append(data[10])
        new_result_list.append(data[5])
        new_result_list.append(data[6])
        new_result_list.append(data[2])
        new_result_list.append(data[3])
        new_result_list.append(data[0])
        new_result_list.append(data[1])
        new_result_list.append([0, '-', '-', '-'])
        new_result_list.append(data[4])
        new_result_list.append([0, '-', '-', '-'])

        data = new_result_list
        print("BGA后处理完毕", data)
    if packageType == 'QFN':
        print("开始提取title参数", data)
        data[7][2], data[8][2],_ = get_nx_ny_from_title(page_num, data[7][2], data[8][2])
        data[7][1] = data[7][2]
        data[7][3] = data[7][2]
        data[8][1] = data[8][2]
        data[8][3] = data[8][2]
        print("提取title参数完毕", data)
        new_parameter_list = []
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append(data[7])
        new_parameter_list.append(data[8])
        new_parameter_list.append(data[2])
        new_parameter_list.append(data[3])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append(data[0])
        new_parameter_list.append(data[1])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append(data[5])
        new_parameter_list.append(data[6])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append(data[11])
        new_parameter_list.append(data[11])

        data = new_parameter_list
        print("QFN后处理完毕", data)
    if packageType == 'QFP':
        print("开始提取title参数", data)
        data[9][2], data[9][2],_ = get_nx_ny_from_title(page_num, data[9][2], data[9][2])
        data[9][1] = data[9][2]
        data[9][3] = data[9][3]
        print("提取title参数完毕", data)

        new_parameter_list = []
        new_parameter_list.append(data[9])
        new_parameter_list.append(data[9])
        new_parameter_list.append(data[2])
        new_parameter_list.append(data[3])
        new_parameter_list.append(data[5])
        new_parameter_list.append(data[6])
        new_parameter_list.append(data[0])
        new_parameter_list.append(data[1])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append(data[7])
        new_parameter_list.append(data[8])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append(data[10])
        new_parameter_list.append(data[11])
        new_parameter_list.append(data[12])
        data = new_parameter_list
        print("QFP后处理完毕", data)
    if packageType == 'SON' or packageType == 'DFN' or packageType == 'DFN_SON':
        print("开始提取title参数", data)
        data[5][2], data[6][2],pin_nums = get_nx_ny_from_title(page_num, data[5][2], data[6][2])
        data[5][1] = data[5][2]
        data[5][3] = data[5][2]
        data[6][1] = data[6][2]
        data[6][3] = data[6][2]
        print("提取title参数完毕", data)
        new_parameter_list = []
        new_parameter_list.append(data[9])
        new_parameter_list.append([0, '-', pin_nums, '-'])
        new_parameter_list.append(data[2])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append(data[0])
        new_parameter_list.append(data[1])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append(data[3])
        new_parameter_list.append(data[4])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append([0, '-', '-', '-'])
        new_parameter_list.append([0, '-', '-', '-'])

        data = new_parameter_list
        print("SON后处理完毕", data)
    if packageType == 'SOP':
        print("开始提取title参数", data)
        data[5][2], data[6][2] = get_nx_ny_from_title(page_num, data[5][2], data[6][2])
        print("提取title参数完毕", data)
    print("postProcess:", data)
    return data