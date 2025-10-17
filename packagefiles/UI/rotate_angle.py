import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageEnhance
import cv2
# 将PDF文件的指定页面转换为图像，并移除其中的所有文本内容
def pdf2img_WithoutText(pdfPath, pageNumber, scale):

    with fitz.open(pdfPath) as pdfDoc:

        page = pdfDoc.load_page(pageNumber-1)
        page.add_redact_annot([-500, -500, 3000, 3000])  # 删除该区域的所有文字
        page.apply_redactions()
        mat = fitz.Matrix(scale, scale).prerotate(0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        # pix.save('tmp.png')  # 将图片写入指定的文件夹内
        img = Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        enhancer = ImageEnhance.Contrast(img)
        enhance_image = enhancer.enhance(factor=6) # factor 是增强因子，小于1时为减弱因子
        image_np = np.array(enhance_image)
        # image_np = np.array(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), dtype=np.uint8)
        
    return image_np
# 在图像上检测非白色区域的外接矩形边界，并在图像上绘制这个矩形
def draw_rectangle(image):
        
    # 初始化边界坐标  
    min_x, min_y = float('inf'), float('inf')  
    max_x, max_y = -float('inf'), -float('inf')  
    
    # 遍历图像寻找非白色像素的边界  
    for y in range(0,image.shape[0],2):  
        for x in range(0,image.shape[1],2):  
            # 检查像素是否为非白色（这里我们假设白色是[255, 255, 255]）  
            if not np.array_equal(image[y, x], [255, 255, 255]):  
                min_x = min(min_x, x)  
                min_y = min(min_y, y)  
                max_x = max(max_x, x)  
                max_y = max(max_y, y)  
    
    # 确保找到了非白色像素  
    if min_x == float('inf') or min_y == float('inf') or max_x == -float('inf') or max_y == -float('inf'):  
        x1,x2 = 0,0

    else:  
        # 计算外接矩形的左上角和右下角坐标  
        x1,y1 = (int(min_x), int(min_y))  
        x2,y2 = (int(max_x), int(max_y))  
        
        # 在图像上绘制矩形
        cv2.rectangle(image, (x1,y1), (x2,y2), (0, 0, 0), 1)
    # cv_show_img(image)
    return image

def get_threshold(image_np):
    # 转化成灰度图
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)  
    # 图像像素取反
    imgBitwise = cv2.bitwise_not(image)
    # 根据阈值二值化灰度图
    Binari_image = cv2.adaptiveThreshold(imgBitwise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -2)
    return Binari_image

# 标准化框线
def delete_redundanceLine_reset_net(xList, yList, HorizontalLine, VerticalLine):
    # 找到最接近的网格点作为线段的起始点
    def find_closest_number_Index(arr, target):  
        left, right = 0, len(arr) - 1  
        closest = arr[0]  # 初始化为数组的第一个元素  

        while left <= right:  
            mid = (left + right) // 2  
            if arr[mid] == target:
                return mid  # 如果找到目标值，直接返回  
            elif arr[mid] < target:  
                closest = arr[mid] if abs(target - arr[mid]) < abs(target - closest) else closest  
                left = mid + 1
            else:
                closest = arr[mid] if abs(target - arr[mid]) < abs(target - closest) else closest  
                right = mid - 1

        return arr.index(closest)  # 返回与目标值最接近的元素的位置
    # 遍历每一条横线
    redundance_HorizontalLine = []
    for index in range(len(HorizontalLine)):
        # 记录线段的初始位置
        Line = HorizontalLine[index]
        LineStart, LineEnd = Line[0], Line[2]
        # 找到线段在初始网格中的起止位置
        IndexStart = find_closest_number_Index(xList, LineStart) 
        IndexEnd = find_closest_number_Index(xList, LineEnd)
        if IndexEnd - IndexStart == 0:
            redundance_HorizontalLine.append(Line)
        elif IndexEnd - IndexStart == 1:
            # 设置偏移量，适当扩大约束条件
            offset = (LineEnd - LineStart)*0.05
            # 线段太短则判断为是冗余线段，需要删除
            if (LineStart - offset) > xList[IndexStart] and (LineEnd + offset) < xList[IndexEnd]:
                redundance_HorizontalLine.append(Line)
        else:
            HorizontalLine[index][0], HorizontalLine[index][2] = LineStart, LineEnd
            # HorizontalLine[index][1] = yList[find_closest_number_Index(yList, Line[1])]
            # HorizontalLine[index][3] = HorizontalLine[index][1]
    # 剔除冗余线段
    HorizontalLine = [Line for Line in HorizontalLine if Line not in redundance_HorizontalLine]
    yList = [Line[1] for Line in HorizontalLine]
    yList = list(set(yList))
    yList = sorted(yList, key = lambda x:x)
    i = 0
    while i < len(yList) - 1:
        if yList[i + 1] - yList[i] < 10:
            yList.pop(i + 1)
        else:
            i += 1
            
    # 遍历每一条竖线
    redundance_VerticalLine = []
    for index in range(len(VerticalLine)):
        Line = VerticalLine[index]
        # 记录线段的初始位置
        LineStart, LineEnd = Line[1], Line[3]
        # 找到线段在初始网格中的起止位置
        IndexStart = find_closest_number_Index(yList, LineStart) 
        IndexEnd = find_closest_number_Index(yList, LineEnd)
        if IndexEnd - IndexStart == 0:
            redundance_VerticalLine.append(Line)
        elif IndexEnd - IndexStart == 1: 
            # 设置偏移量，适当扩大约束条件
            offset = round((LineEnd - LineStart)*0.08)
            # 线段太短则判断为是冗余线段，需要删除
            if (LineStart - offset) > yList[IndexStart] or (LineEnd + offset*2) < yList[IndexEnd]:
                redundance_VerticalLine.append(Line)
        else:
            VerticalLine[index][1], VerticalLine[index][3] = LineStart, LineEnd
            # VerticalLine[index][0] = xList[find_closest_number_Index(xList, Line[0])]
            # VerticalLine[index][2] = VerticalLine[index][0]
    VerticalLine = [Line for Line in VerticalLine if Line not in redundance_VerticalLine]
    xList = [Line[0] for Line in VerticalLine]
    xList = list(set(xList))
    xList = sorted(xList, key = lambda x:x)
    # 对网格线进行后处理，合并过近的竖线，保留谁
    i = 0
    while i < len(xList) - 1:
        if xList[i + 1] - xList[i] < 15:
            xList.pop(i + 1)
        else:
            i += 1
    return xList, yList, HorizontalLine, VerticalLine

# 找到所有的线
def findLines(BinaryThreshold, tableCoordinate = []):  # img_path
    # 标准化框线，找到最接近的网格点作为线段的起始点
    def find_closest_number(arr, target):  
        left, right = 0, len(arr) - 1  
        closest = arr[0]  # 初始化为数组的第一个元素  
    
        while left <= right:  
            mid = (left + right) // 2  
            if arr[mid] == target:  
                return arr[mid]  # 如果找到目标值，直接返回  
            elif arr[mid] < target:  
                closest = arr[mid] if abs(target - arr[mid]) < abs(target - closest) else closest  
                left = mid + 1  
            else:
                closest = arr[mid] if abs(target - arr[mid]) < abs(target - closest) else closest  
                right = mid - 1
    
        return closest  # 返回与目标值最接近的元素 
    # 得到每每条框线的起止点，并返回横向或纵向有哪些坐标
    def get_HorizontalLine_Coordinate(image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yList = []
        Line = []
        # 遍历轮廓并获取每个轮廓的起止点
        for contour in contours:
            # 轮廓的点是按顺序排列的，所以可以取第一个点和最后一个点作为起止点
            combined_array = np.vstack(contour)  
            max_x = np.max(combined_array[:, 0])
            min_x = np.min(combined_array[:, 0])
            max_y = np.max(combined_array[:, 1])
            yList.append(max_y)
            Line.append([min_x,max_y,max_x,max_y])
        # 对得到的List和Line进行后处理
        yList = list(set(sorted(yList)))
        yList = sorted(yList, key = lambda x:x)
        sum_of_diffs = 0
        for i in range(1, len(yList)):  
            # 计算当前元素和前一个元素的差值，并累加到sum_of_diffs  
            sum_of_diffs += yList[i] - yList[i-1]
        i = 0
        try:
            if len(yList) > 1:
                if yList[i+1] - yList[i] < 7:
                    yList.pop(i)
                while i < len(yList) - 1:
                    if yList[i + 1] - yList[i] < 8:
                        yList.pop(i + 1)
                    else:
                        i += 1
        except IndexError:
            pass
        return yList, Line
    def get_VerticalLine_Coordinate(image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        List = []
        Line = []
        # 遍历轮廓并获取每个轮廓的起止点
        for contour in contours:
            # 轮廓的点是按顺序排列的，所以可以取第一个点和最后一个点作为起止点
            combined_array = np.vstack(contour)  
            max_x = np.max(combined_array[:, 0])
            min_y = np.min(combined_array[:, 1])
            max_y = np.max(combined_array[:, 1])
            List.append(max_x)
            Line.append([max_x,min_y,max_x,max_y])
        List = list(set(sorted(List)))
        List = sorted(List, key = lambda x:x)
        sum_of_diffs = 0
        for i in range(1, len(List)):  
            # 计算当前元素和前一个元素的差值，并累加到sum_of_diffs  
            sum_of_diffs += List[i] - List[i-1]
        i = 0

        if List[i + 1] - List[i] < 7:
            List.pop(i)
        while i < len(List) - 1:
            if List[i + 1] - List[i] < 15:
                List.pop(i + 1)
            else:
                i += 1
        return List, Line

    horizontal = BinaryThreshold.copy()
    vertical = BinaryThreshold.copy()

    horizontalSize = 30
    # 构造横向卷积核
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    # 图像腐蚀
    horizontal = cv2.erode(horizontal, horizontalStructure, iterations=1)
    # 图像膨胀
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=1)
    yList, HorizontalLine = get_HorizontalLine_Coordinate(horizontal)

    verticalsize = 20
    # 构造纵向卷积核
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # 图像腐蚀
    vertical = cv2.erode(vertical, verticalStructure, iterations = 1)
    # 图像膨胀
    vertical = cv2.dilate(vertical, verticalStructure, iterations= 1)
    xList, VerticalLine= get_VerticalLine_Coordinate(vertical)
    HorizontalLine = sorted(HorizontalLine, key = lambda x:x[1])
    VerticalLine = sorted(VerticalLine, key = lambda x:x[0])

    # 剔除冗余框线
    xList, yList, HorizontalLine, VerticalLine = delete_redundanceLine_reset_net(xList, yList, HorizontalLine, VerticalLine)
    # 将线的端点与网格点相匹配
    for index in range(len(HorizontalLine)):
        HorizontalLine[index][0] = find_closest_number(xList,HorizontalLine[index][0]) 
        HorizontalLine[index][2] = find_closest_number(xList,HorizontalLine[index][2])
        HorizontalLine[index][1] = find_closest_number(yList,HorizontalLine[index][1])
        HorizontalLine[index][3] = find_closest_number(yList,HorizontalLine[index][3])
    for index in range(len(VerticalLine)):
        VerticalLine[index][1] = find_closest_number(yList,VerticalLine[index][1])
        VerticalLine[index][3] = find_closest_number(yList,VerticalLine[index][3])
        VerticalLine[index][0] = find_closest_number(xList,VerticalLine[index][0])
        VerticalLine[index][2] = find_closest_number(xList,VerticalLine[index][2])

    return xList, yList, HorizontalLine, VerticalLine


def get_Border(image_np, tableCoordinate):
    # 表格无底色时，进行二值化
    if image_np.shape.__len__() == 3:
        # 1. 仅保留框线中的内容
        image_np_crop = np.full_like(image_np, 255)
        image_np_crop[tableCoordinate[1]:tableCoordinate[3],tableCoordinate[0]:tableCoordinate[2]] = image_np[tableCoordinate[1]:tableCoordinate[3],tableCoordinate[0]:tableCoordinate[2]]
        # 2. 画出外框
        image_draw_line = draw_rectangle(image_np_crop)
        # cv_show_img(image_draw_line)
        # 3. 对图像进行二值化
        BinaryThreshold = get_threshold(image_draw_line)
    # 表格有底色时，输入的是处理后的二值化图片
    else:
        BinaryThreshold = np.full_like(image_np, 0)
        BinaryThreshold[tableCoordinate[1]:tableCoordinate[3],tableCoordinate[0]:tableCoordinate[2]] = image_np[tableCoordinate[1]:tableCoordinate[3],tableCoordinate[0]:tableCoordinate[2]]
    # 4. opencv得到表格框选中拐角处的坐标
    xList, yList, HorizontalLine, VerticalLine = findLines(BinaryThreshold)

    return xList, yList, HorizontalLine, VerticalLine

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

def Get_Ocr_TableImage(pdfPath, pageNumber,Coordinate):
    scale = 4
    with fitz.open(pdfPath) as pdfDoc:
        x1,y1,x2,y2 = Coordinate
        page = pdfDoc.load_page(pageNumber-1)
        clip_rect = fitz.Rect(x1,y1,x2,y2)
        mat = fitz.Matrix(scale, scale).prerotate(0)
        pix = page.get_pixmap(matrix=mat, alpha=False,clip = clip_rect)
        # pix.save('tmp.png')  # 将图片写入指定的文件夹内
        tableImage = np.array(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), dtype=np.uint8)
        # img = Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        # enhancer = ImageEnhance.Contrast(img)
        # enhance_image = enhancer.enhance(factor=3) # factor 是增强因子，小于1时为减弱因子

        # tableImage = np.array(enhance_image)
    return tableImage

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

def Is_Common_package(table):
    JudgeList = ['A','A1','A2','D','E','e','D1','E1']
    count = 0
    # # 判断第一列是否是符合标准的，符合就不旋转
    # for row in table:
    #     for tag in JudgeList:
    #         if tag in row[0]:
    #             count += 1
    #             break
    for row in table:
        if row[0] in JudgeList:# 完全匹配，而非子字符串匹配
            count += 1
            break
    if count > 3:
        return 0
    # 判断是否能找到符合条件的行
    else:
        for index in range(len(table)):
            count = 0
            row = table[index]
            for cell in row:
                if cell in JudgeList:
                    count += 1
            if count > 3:
                if index >= len(table)/2:
                   return 90
                # 转表不转图
                elif index == len(table)/2 - 1:
                    return -1
        # 没找到就直接逆时针旋转90
        return 270

def rotate(pdfPath,pageNumber, Coordinate):
    scale = 2
    direction = -1
# 页面转图片
    image = pdf2img_WithoutText(pdfPath, pageNumber, scale)
    tableCoordinate = [round(x*scale) for x in Coordinate]
    try:
        xList, yList, HorizontalLine, VerticalLine = get_Border(image, tableCoordinate)
    except IndexError:
        return 0
    # F4.4 找到图中除开文字的所有框线,并提取表格内容
    # 得到所有单元格的坐标
    cellsCoordinate = get_cells_coordinate(xList, yList, HorizontalLine, VerticalLine)
    TableImage = Get_Ocr_TableImage(pdfPath, pageNumber,Coordinate)
    table = get_texts_UsingOcr(TableImage, tableCoordinate, cellsCoordinate)
    rotate = Is_Common_package(table)
    return rotate

if __name__ == "__main__":
    pdfPath = r'D:\梁雪婷2024\new_package_part1.0\temp\C514430_DC-DC电源芯片_LT8645SEV#PBF_规格书_WJ109290.PDF'
    pageNumber = 28
    Coordinate = [357,359,549,513]  # 表格坐标
    from getCoordinate import *
    # Coordinate = getCoordinate(pdfPath, pageNumber)
    # print(Coordinate)
    rotate_ang = rotate(pdfPath, pageNumber, Coordinate)  # 调用函数进行旋转处理
    print(rotate_ang)