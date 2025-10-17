"""页数判断，少于10页不处理"""
"""文档目录关键字判断, 全文关键字判断, img类型添加"""
import fitz

 
def pre_judge(path):
    """
        预判断，纯图文档或者页数<=10页，不处理，返回1，否则返回0
    """
    flag = 0
    with fitz.open(path) as doc:
        if (doc.page_count <= 10):
            flag = 1
    return flag
 
def judge_toc_package(text):
    """
        判断输入目录文本是否含有package的关键字，若有，则返回True
    """
    text = text.lower()
    # if (('packag' in text) or ('dimension' in text) or ('外形尺寸' in text) or ('mechanical' in text) or ('封装' in text)
    #         or ('die' in text) or ('mark' in text)):
    if (('packag' in text) or ('dimension' in text) or ('尺寸' in text) or
            ('mechanical' in text) or ('封装' in text) or ('physical' in text) or ('qfp' in text)):
        # print(text)
        return True
    return False

def judge_package(text):
    """
        判断输入文本中是否含package_img的关键字，若有，则返回True
    """
    text = text.lower()
    if (('package' in text) or ('dimension' in text)
         or ('mechanical data' in text) or ('尺寸' in text) or ('ball' in text) or
            ('封装' in text) or ('physical' in text) or ('qfp' in text)):
        return True
    return False  # 不存在symbol_img的关键字


def toc_package_screen(path):
    """
        总目录索引定位封装图范围
    """
    page_list = []
    with fitz.open(path) as doc:
        page_count = doc.page_count
        toc_list = doc.get_toc()                 # 获取文档目录索引列表
    index = 0                 # 文档目录索引等级标记
    toc_list_count = len(toc_list)  # 文档目录索引个数
    # 获取文档最深目录索引级数
    for i in range(toc_list_count):
        if (index < toc_list[i][0]):
            index = toc_list[i][0]
    rank_max = index                     # 文档最深目录索引级数
 
    for i in range(toc_list_count):
        if (toc_list[i][0] <= index):
            index = rank_max           # index 标志初始化
            if (judge_toc_package(toc_list[i][1])):
                rank_flag = 0
                rank = toc_list[i][0]
                index = toc_list[i][0]
                start = toc_list[i][2] - 1          # 开始页码
                for j in range(i + 1, toc_list_count):
                    if (toc_list[j][0] <= rank):
                        rank_flag = 1
                        end = toc_list[j][2] - 1        # 结束页码
                        break
                if (not rank_flag):
                    for num in range(start, page_count):
                        page_list.append(num)
                else:
                    for num in range(start, end + 1):
                        page_list.append(num)
    if (len(page_list)):
                      # 目录方法成为结果返回方法
        start_page = toc_list[-1][2]
        with fitz.open(path) as doc:
            for i in range(start_page, page_count):
                if (len(doc[i].get_text()) == 0):
                    page_list.append(i)
                elif (len(doc[i].get_images()) != 0):
                    page_list.append(i)
    page_list = list(set(page_list))            # 去除重复元素
    page_list.sort()                  # 页码列表排序
    return page_list
 
def get_img_size_list(page):
    """
        获取指定页图像信息列表
    """
    img_list = page.get_images()
    img_size_list = []
    for img in img_list:
        img_size = (img[2], img[3])
        img_size_list.append(img_size)
    return img_size_list
 
def get_com_list(list1, list2, list3):
    """
        返回3个列表的公共元素
    """
    return list(set(list1) & set(list2) & set(list3))
 
def package_img_screen(path):
    """
        提取文档中以img类型存储的图片页码
    """
    page_list = []
    with fitz.open(path) as doc:
        page_count = doc.page_count
        page1 = doc[4]
        page2 = doc[-5]
        page3 = doc[(page_count - 1) // 2]
        if ((len(page1.get_images()) >= 80) or (len(page2.get_images()) >= 80)
                or (len(page3.get_images()) >= 80)):
            return []     # img无法提取，直接输出全页码
        log_img_size_list = get_com_list(get_img_size_list(page1), get_img_size_list(page2),
                                         get_img_size_list(page3))
 
        for i in range(page_count):
            if (len(doc[i].get_images()) >= 80):
                page_list.append(i)             # 单页img > 80， 直接作为结果
            else:
                cur_img_size_list = get_img_size_list(doc[i])
                if (len(set(cur_img_size_list) - set(log_img_size_list))):
                    #print(f"{i + 1}有图片")
                    page_list.append(i)
    return page_list

def keywords_screen(path):
    """
        全文关键字搜索
    """
    page_list = []
    with fitz.open(path) as doc:
        page_count = doc.page_count            # pdf所含页数
        for i in range(page_count):
            page = doc[i]                  # 获取页面对象
            text_block_list = page.get_text_blocks()     # 获取该页面上文本框信息列表
            text_block_count = len(text_block_list)     # 获取该页上文本块数量
            if (len(page.get_text()) == 0):
                page_list.append(i)
            else:
                for j in range(text_block_count):
                    if (judge_package(text_block_list[j][4].replace('\n', ''))):
                        page_list.append(i)
                        break
        for i in range(page_count - 5, page_count):
            page_list.append(i)
    page_list = list(set(page_list))
    return page_list

def page_size_screen(path):
    """
        遍历pdf页的大小
    """
    all_page_list = []
    with fitz.open(path) as doc:
        num = doc.page_count
        for i in range(num):
            size = doc[i].rect
            if (size[2] >= size[3]):
                all_page_list.append(i)

    return all_page_list

def extract_package_page_list(path):
    """
    提取文档的含symbol页
    :param path:
    :return 页码列表:
    """
    # 文档预判断
    with fitz.open(path) as doc:
        page_count = doc.page_count
    if (pre_judge(path)):
        return [i for i in range(page_count)]                 # 文档页数少于10页，不处理
    # 调用全页pdf匹配
    all_page_list = page_size_screen(path)
    toc_page_list = toc_package_screen(path)  # 调用package图的目录方法
    if (len(toc_page_list)):  # 存在package关键字目录
        # 取并集
        toc_page_list_new = list(set(all_page_list) | set(toc_page_list))
        # 排序
        toc_page_list_new.sort()
        return toc_page_list_new
    else:
        # 调用全文关键字搜索，关键字搜索若有结果就输出，没有结果输出空列表
        page_list1 = keywords_screen(path)
        #page_list2 = package_img_screen(path)
        #page_list = list(set(page_list1) | set(page_list2))
        page_list_new = list(set(all_page_list) | set(page_list1))
        page_list = page_list_new
        # 排序
        page_list.sort()
        return page_list
 
 
if __name__ == '__main__':
    path = r"" + input("请输入文件地址:\n")
    page_list = extract_package_page_list(path)
    print(page_list)
    print(len(page_list))