import fitz
import re
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
import os

# # 处理每个关键字的搜索结果
def rect_overlap_ratio(box1, box2):
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    inter_x0 = max(x0_1, x0_2)
    inter_y0 = max(y0_1, y0_2)
    inter_x1 = min(x1_1, x1_2)
    inter_y1 = min(y1_1, y1_2)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    min_area = min(area1, area2)
    if min_area == 0:
        return 0
    return inter_area / min_area


def is_page_editable(pdf_path, page_num):
    """
    判断 PDF 指定页面是否可编辑
    """
    try:
        doc = fitz.open(pdf_path)
        # 获取指定页面
        if 1 <= page_num <= len(doc):
            page = doc[page_num - 1]  # 索引从 0 开始
            # 提取文本内容
            text = page.get_text()
            # 检查页面是否包含文本
            if text.strip():
                return True
            else:
                # 检查页面是否主要由图像组成
                image_list = page.get_images(full=True)
                if len(image_list) > 0:
                    return False
                else:
                    # 无文本和图像（可能是空白页或特殊格式）
                    return False
        else:
            print(f"页面编号 {page_num} 超出范围")
            return False

    except Exception as e:
        print(f"处理 PDF 时出错: {e}")
        return False

def find_line_boundaries(page_dict, y0, y1):
    """确定行的上下边界"""
    line_height = y1 - y0
    # 扩大范围以包含整行
    line_top = y0 - line_height * 0.5
    line_bottom = y1 + line_height * 0.5
    return line_top, line_bottom

def get_full_line_text(page, x0, y0, x1, y1, line_top, line_bottom):
    """获取整行文本"""
    # 获取整页宽度
    page_width = page.rect.width
    # 获取整行文本
    full_line = page.get_text("text", clip=(0, line_top, page_width, line_bottom))
    return full_line.strip()

def clean_excel_text(text):
    # 移除所有非法的控制字符（除常用换行、回车、tab外）
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

def search_keywords_in_editable_page(pdf_path, page_num, keywords):
    """在可编辑的页面中搜索关键字并获取其标签和坐标"""
    try:
        doc = fitz.open(pdf_path)
        if not 1 <= page_num <= len(doc):
            print(f"页面编号 {page_num} 超出范围")
            return []

        page = doc[page_num - 1]

        if not is_page_editable(pdf_path, page_num):
            print(f"第 {page_num} 页不可编辑")
            return []

        keyword_results = {}
        page_dict = page.get_text("dict")

        # 获取每个关键字的坐标（原有代码不变）
        for keyword in keywords:
            instances = page.search_for(keyword)
            for inst in instances:
                x0, y0, x1, y1 = map(int, inst)
                # 1. 首先确定当前行的y坐标范围
                line_top, line_bottom = find_line_boundaries(page_dict, y0, y1)
                # 2. 获取整行的文本
                line_text = get_full_line_text(page, x0, y0, x1, y1, line_top, line_bottom)
                line_text = clean_excel_text(line_text)

                font_size = 0
                for span in page.get_text("dict", clip=(x0, y0, x1, y1))["blocks"]:
                    if "lines" in span:
                        for line in span["lines"]:
                            if "spans" in line:
                                for span_info in line["spans"]:
                                    if keyword.lower() in span_info["text"].lower():
                                        font_size = span_info["size"]
                                        break

                result = {
                    'page': page_num - 1,
                    'keyword': keyword,
                    'coordinates': (x0, y0, x1, y1),
                    'type': 'text',
                    'content': line_text,
                    'font_size': font_size,
                    'area': (x1 - x0) * (y1 - y0)
                }

                if keyword not in keyword_results:
                    keyword_results[keyword] = []
                keyword_results[keyword].append(result)

        overlap_threshold = 0.5
        intermediate_results = []

        # 排序+重叠检测
        for keyword, results in keyword_results.items():
            # 先按字体大小降序排序
            results.sort(key=lambda x: (-x['font_size'], x['area']))

            filtered_results = []
            for res in results:
                overlap = False
                for kept in filtered_results:
                    ratio = rect_overlap_ratio(res['coordinates'], kept['coordinates'])
                    if ratio > overlap_threshold:
                        # 重叠时保留面积较小的结果
                        if res['area'] < kept['area']:
                            filtered_results.remove(kept)
                            filtered_results.append(res)
                        overlap = True
                        break
                if not overlap:
                    filtered_results.append(res)

            intermediate_results.extend(filtered_results)

        # 对keyword为的数据进行font_size过滤
        filtered_by_fontsize = []
        temp_group = defaultdict(list)
        for res in intermediate_results:
            temp_group[res['keyword']].append(res)
        keywords_to_filter = { "DFN"}
        for k, group in temp_group.items():
            if k in keywords_to_filter and len(group) > 1:
                font_sizes = [r.get('font_size', 0) for r in group]
                max_font = max(font_sizes)
                if len(set(font_sizes)) > 1:
                    group = [r for r in group if r.get('font_size', 0) == max_font]
            filtered_by_fontsize.extend(group)

        # 处理最终结果
        for result in filtered_by_fontsize :
            result.pop('font_size', None)
            result.pop('area', None)
            if result['keyword'].lower() == 'side view':
                result['keyword'] = 'SIDEVIEW'
            elif result['keyword'].lower() == 'top view':
                result['keyword'] = 'TOPVIEW'

        return filtered_by_fontsize

    except Exception as e:
        print(f"处理页面时出错: {e}")
        return []
    finally:
        if 'doc' in locals():
            doc.close()

def perform_ocr(image):
    """
    对图像进行OCR识别
    """
    from packagefiles.UI.det_table import Run_onnx1
    # 使用Run_onnx1进行OCR识别
    boxes, texts = Run_onnx1(image, "temp_ocr")
    return boxes, texts


def process_package_image(package_img, keywords, page_num):
    """
    对package图像进行OCR识别，并检查关键字
    """
    try:
        # 进行OCR识别
        boxes, texts = perform_ocr(package_img)
        if not boxes or not texts:
            print("OCR识别失败")
            return []

        results = []
        # 处理每个识别到的文本区域
        for box, text in zip(boxes, texts):
            if text:
                t = text.strip().lower()
                if t == "side view":
                    text = "SIDEVIEW"
                elif t == "top view":
                    text = "TOPVIEW"
                elif t == "top viey":
                    text = "TOPVIEW"
                elif t == "top vew":
                    text = "TOPVIEW"
                elif t == "side vew":
                    text = "SIDEVIEW"

            # 检查文本是否包含任何关键字
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    # 计算文本框的边界坐标
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]
                    x0, y0 = min(x_coords), min(y_coords)
                    x1, y1 = max(x_coords), max(y_coords)

                    # 添加识别结果
                    results.append({
                        'page': page_num,
                        'keyword': keyword,
                        'coordinates': (x0, y0, x1, y1),
                        'type': 'ocr',  # 标记为OCR识别结果
                        'content': text.strip() if text else ""
                    })
                    break  # 找到一个匹配的关键字就跳出内层循环

        return results

    except Exception as e:
        print(f"处理package图像时出错: {e}")
        return []


def process_non_editable_page(pdf_path, page_num, keywords):
    """
    对不可编辑的PDF页面，直接渲染为图片并进行OCR识别
    """
    try:
        doc = fitz.open(pdf_path)
        if not 1 <= page_num <= len(doc):
            print(f"页面编号 {page_num} 超出范围")
            return []

        page = doc[page_num - 1]

        # 直接渲染页面为图片
        zoom = 3  # 增加缩放比例
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 对图片进行OCR识别，传入页码
        results = process_package_image(img_cv, keywords, page_num)

        # 将坐标转换回原始PDF页面比例
        scaled_results = []
        for result in results:
            x0, y0, x1, y1 = result['coordinates']
            keyword = result['keyword']
            if keyword.lower() == 'side view':
                keyword = 'sideview'
            elif keyword.lower() == 'top view':
                keyword = 'topview'
            scaled_results.append({
                'page': page_num - 1,
                'keyword': keyword,
                'coordinates': (
                    int(x0 / zoom),
                    int(y0 / zoom),
                    int(x1 / zoom),
                    int(y1 / zoom)
                ),
                'type': 'ocr',
                'content': result['content']
            })

        return scaled_results

    except Exception as e:
        print(f"处理不可编辑页面时出错: {e}")
        return []
    finally:
        if 'doc' in locals():
            doc.close()

def process_image(package_img, keywords,page_num):
    """
    对图像进行OCR识别，并检查关键字
    """
    try:
        # 进行OCR识别
        boxes, texts = perform_ocr(package_img)
        if not boxes or not texts:
            print("OCR识别失败")
            return []

        results = []
        # 处理每个识别到的文本区域
        for box, text in zip(boxes, texts):
            if text:
                t = text.strip().lower()
                if t == "side view":
                    text = "SIDEVIEW"
                elif t == "top view":
                    text = "TOPVIEW"
                elif t == "top viey":
                    text = "TOPVIEW"

            # 检查文本是否包含任何关键字
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    # 计算文本框的边界坐标
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]
                    x0, y0 = min(x_coords), min(y_coords)
                    x1, y1 = max(x_coords), max(y_coords)
                    zoom = 3
                    # 添加识别结果（坐标缩放回原始比例）
                    results.append({
                        'page': page_num,
                        'keyword': keyword,
                        'coordinates': (
                            int(x0 / zoom),
                            int(y0 / zoom),
                            int(x1 / zoom),
                            int(y1 / zoom)
                        ),
                        'type': 'ocr',
                        'content': text.strip() if text else ""
                    })
                    break  # 找到一个匹配的关键字就跳出内层循环

        return results

    except Exception as e:
        print(f"处理package图像时出错: {e}")
        return []

def visualize_results(pdf_path, page_num, results, is_editable):
    """
    将PDF页面转换为图像并绘制识别结果
    """
    try:
        # 打开PDF并转换为图像（不缩放）
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]

        # 根据页面是否可编辑选择不同的缩放比例
        zoom = 3 if not is_editable else 1  # 不可编辑页面使用高分辨率
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 在图像上绘制结果
        for result in results:
            x0, y0, x1, y1 = result['coordinates']
            # 调整坐标比例
            x0, y0, x1, y1 = int(x0 * zoom), int(y0 * zoom), int(x1 * zoom), int(y1 * zoom)

            # 根据类型选择不同颜色
            color = (0, 0, 255) if result['type'] == 'text' else (0, 255, 0)  # 文本红色，OCR绿色

            # 绘制边界框
            cv2.rectangle(img_cv, (x0, y0), (x1, y1), color, 2)
            # 添加文本标签
            label = f"{result['keyword']}: {result['content'][:30]}"  # 限制显示长度
            cv2.putText(img_cv, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 保存结果
        output_dir = "visualization_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"page_{page_num}_results.png")
        cv2.imwrite(output_path, img_cv)

        # 显示结果
        cv2.imshow(f"Page {page_num} Results ({'Editable' if is_editable else 'OCR'})", img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"可视化结果已保存到: {output_path}")

    except Exception as e:
        print(f"可视化结果时出错: {e}")
    finally:
        if 'doc' in locals():
            doc.close()


if __name__ == "__main__":
    # 测试代码
    pdf_path = "BGADFNSONQFP/一页双封装的/AD9164.pdf"  # 替换为实际的PDF文件路径
    page_num = 136 # 要处理的页码
    keywords = ["BGA", "DFN", "SON", "QFP", "QFN", "SOP","SOT", "Plastic Quad Flat Package","TOPVIEW","TOP VIEW", "SIDEVIEW","SIDE VIEW","TOP","SIDE","VIEW","DETAIL"]

    is_editable = is_page_editable(pdf_path, page_num)
    if is_editable:
        # 测试可编辑页面处理
        results = search_keywords_in_editable_page(pdf_path, page_num, keywords)
        print("可编辑页面结果:", results)
        print(len(results))
        # 可视化结果
        # visualize_results(pdf_path, page_num, results, True)
    else:
        # 测试不可编辑页面处理
        results = process_non_editable_page(pdf_path, page_num, keywords)
        print("不可编辑页面结果:", results)  # 可视化结果
        # visualize_results(pdf_path, page_num, results, False)
