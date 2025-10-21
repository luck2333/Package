"""yolox检测以及一些辅助函数"""
import os
import shutil
import json
import math

# 第三方库
import onnxruntime
from shapely.geometry import Polygon
from pathlib import Path

try:
    from packagefiles.model_paths import yolo_model_path
except ModuleNotFoundError:  # pragma: no cover - 兼容脚本直接运行
    def yolo_model_path(*parts):
        return str(Path(__file__).resolve().parents[2] / 'model' / 'yolo_model' / Path(*parts))

from packagefiles.UI.watermark_remove import watermark_remove
from packagefiles.UI.extract_package_page_list_2 import extract_package_page_list
from packagefiles.UI.page_num import *
from packagefiles.UI.test6 import *
from packagefiles.UI.test1 import *
from packagefiles.UI.rotate_angle import rotate

# 常量
IMAGE_PATH = r"Result/PDF_extract/yolo_img"      # 存放待yolox检测的图片文件地址
SAVE_IMG_PATH = r"Result/PDF_extract/yolo_result"    # 存放yolox检测的结果图片
ZOOM = (3, 3)       # pdf转图片时的放大倍数
# VOC_CLASSES = ('package', 'BGA', 'DFN_SON', 'Top', 'Side', 'Detail', 'Form', 'Note')  # BGA标签类型
# VOC_CLASSES = ('package', 'BGA', 'DFN','SON', 'QFP','QFN','SOP', 'Top', 'Side', 'Detail', 'Form', 'Note')  # 标签类型
VOC_CLASSES = ('package', 'BGA', 'DFN_SON','QFP','QFN','SOP', 'Top', 'Side', 'Detail', 'Form', 'Note')  # 3类标签类型
INPUT_SHAPE = (640, 640)    # 图片输入大小
MODEL = yolo_model_path('package_model', 'best_ckpt_0805.onnx')
SCORE_THR = 0.8          # 置信度
DETECT_JSON_PATH = r"detect.json"
TEMP_DIRECTORY = r"Result/temp"         # 临时存放pdf文件夹
PDF_NAME = TEMP_DIRECTORY + '\\detect.pdf'
PDF_NAME_MINI = TEMP_DIRECTORY + '\\detect_mini.pdf'

PACKAGE_COLOR = (1, 0, 0)
KEYVIEW_COLOR = (0, 1, 1)
TOP_COLOR = (0.5, 1, 0)
SIDE_COLOR = (1, 0.5, 0)
DETAIL_COLOR = (0.5, 0, 1)
FORM_COLOR = (0, 0, 1)
NOTE_COLOR = (1, 0, 0.5)

def remove_dir(dir_path):
    """
        删除dir_path文件夹（包括其所有子文件夹及文件）
    """
    shutil.rmtree(dir_path)

def create_dir(dir_path):
    """
        创建dir_path空文件夹（若存在该文件夹则清空该文件夹）
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

def pdf2img(pdf_path, pages):
    """
        将pages列表中的页转为图片
    :param pdf_path:
    :param pages:
    :return:
    """
    create_dir(IMAGE_PATH)
    with fitz.open(pdf_path) as doc:
        for i in range(len(pages)):
            page = doc[pages[i]]
            mat = fitz.Matrix(ZOOM[0], ZOOM[1])
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(os.path.join(IMAGE_PATH, f"{pages[i] + 1}.png"))

def expand_and_crop_image(box, image, expansion_factor_x, expansion_factor_y):
    height, width = image.shape[:2]

    # 框的坐标
    x1, y1, x2, y2 = box

    # 计算框的中心点和宽度/高度
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width_box = x2 - x1
    height_box = y2 - y1

    # 计算扩大后的框的尺寸
    expanded_width = int(width_box * expansion_factor_x)
    expanded_height = int(height_box * expansion_factor_y)

    # 计算扩大后的框的左上角和右下角坐标
    expanded_x1 = max(0, center_x - expanded_width / 2)
    expanded_y1 = max(0, center_y - expanded_height / 2)
    expanded_x2 = min(width, center_x + expanded_width / 2)
    expanded_y2 = min(height, center_y + expanded_height / 2)

    # 确保框的坐标是整数
    expanded_x1, expanded_y1, expanded_x2, expanded_y2 = map(int, (expanded_x1, expanded_y1, expanded_x2, expanded_y2))

    # 裁剪图片
    cropped_img = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]

    rectangle = expanded_x1, expanded_y1, expanded_x2, expanded_y2

    return rectangle, cropped_img

def straight_line(img, horizontalSize, verticalSize):
    src_img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src_img2 = cv2.bitwise_not(src_img0)

    thresh, AdaptiveThreshold = cv2.threshold(src_img2, 5, 255, 0)

    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    mask = horizontal + vertical
    return mask


def calculate_iou(box_a, box_b):
    # 确定矩形的 (x1, y1, x2, y2)
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # 计算交集面积
    interArea = max(xB - xA + 1, 0) * max(yB - yA + 1, 0)

    # 计算并集面积
    boxAArea = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    boxBArea = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    unionArea = boxAArea + boxBArea - interArea

    # 避免除数为0的情况
    if unionArea == 0:
        return 0

    # 计算IoU
    iou = interArea / unionArea
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def table_rectangle(image, rect):
    typ = [None, None, None, None, None]
    rectangle, img = expand_and_crop_image(rect, image, 2, 2)
    (rx1, ry1, rx2, ry2) = rectangle
    (bx1, by1, bx2, by2) = rect
    new_bx1 = bx1 - rx1
    new_by1 = by1 - ry1
    new_bx2 = bx2 - rx1
    new_by2 = by2 - ry1
    box = (new_bx1, new_by1, new_bx2, new_by2)
    src_img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src_img1 = cv2.GaussianBlur(src_img0, (3, 3), 0)
    src_img2 = cv2.bitwise_not(src_img1)
    thresh, AdaptiveThreshold = cv2.threshold(src_img2, 5, 255, 0)
    x1, y1, x2, y2 = box
    y1 = int(y1)
    y2 = int(y2)
    x1 = int(x1)
    x2 = int(x2)
    box_area = AdaptiveThreshold[y1:y2, x1:x2]
    white_area = np.count_nonzero(box_area)
    box_area = (y2 - y1) * (x2 - x1)
    if white_area / box_area > 0.75:
        typ[0] = "Background"
        typ[1] = "no frame"
        typ[2] = "no outer frame"
        typ[3] = "no Horizontal lines"
        typ[4] = "no vertical lines"
        mask = straight_line(img, int(image.shape[1] / 20), int(image.shape[0] / 30))
        kernel = np.ones((6, 6), np.uint8)
        binary = cv2.dilate(mask, kernel)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        screened_rectangles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            screened_rectangles.append([x, y, x + w, y + h])

        best_match = None
        max_iou = 0
        for screen_rect in screened_rectangles:
            iou = calculate_iou(box, screen_rect)
            if iou > max_iou:
                max_iou = iou
                best_match = screen_rect
    else:
        typ[0] = "no Background"
        mask = straight_line(img, int(image.shape[1] / 20), int(image.shape[0] / 30))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        screened_rectangles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            cnt_area = cv2.contourArea(cnt)
            ratio = cnt_area / area
            screened_rectangles.append([x, y, x + w, y + h, ratio])

        match = None
        max_iou = 0
        rect_ratio = 0
        for screen_rect_with_ratio in screened_rectangles:
            screen_rect = screen_rect_with_ratio[:4]
            iou = calculate_iou(box, screen_rect)
            if iou > max_iou:
                max_iou = iou
                match = screen_rect
                rect_ratio = screen_rect_with_ratio[-1]

        # match_area = (match[2] - match[0]) * (match[3] - match[1])
        # box_area = (box[2] - box[0]) * (box[3] - box[1])

        if max_iou > 0.5:
            typ[1] = "frame"
            best_match = match
            if rect_ratio >= 0.9:
                typ[2] = "outer frame"
            else:
                typ[2] = "no outer frame"

            need1, expanded_rectangle1 = expand_and_crop_image(best_match, img, 1.05, 1.05)
            # 设定长度阈值，即行中白色像素的最小数量
            length_threshold = expanded_rectangle1.shape[1] / 2

            src_img1 = cv2.cvtColor(expanded_rectangle1, cv2.COLOR_BGR2GRAY)
            src_img2 = cv2.bitwise_not(src_img1)
            thresh, AdaptiveThreshold = cv2.threshold(src_img2, 5, 255, 0)
            horizontal = AdaptiveThreshold.copy()
            horizontalSize = int(horizontal.shape[1] / 10)
            horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
            horizontal = cv2.erode(horizontal, horizontalStructure)
            horizontal = cv2.dilate(horizontal, horizontalStructure)

            sss = []
            for y in range(horizontal.shape[0]):
                row = horizontal[y, :]
                non_zero_count = np.count_nonzero(row)  # 计算非零像素的数量
                # 如果当前行的长度超过阈值
                if non_zero_count > length_threshold:
                    sss.append(y)
            if sss:
                count = 1
            else:
                count = 0
            for i in range(1, len(sss)):
                if sss[i] - sss[i - 1] > 1:  # 如果当前元素与前一个元素的差值大于1
                    count += 1  # 增加组的数量
            if count <= 5:
                typ[3] = "no Horizontal lines"
            else:
                typ[3] = "Horizontal lines"

            typ[4] = "vertical lines"

        else:
            typ[1] = "no frame"
            typ[2] = "no outer frame"
            typ[3] = "Horizontal lines"
            typ[4] = "no vertical lines"
            need, expanded_rectangle = expand_and_crop_image(box, mask, 1.2, 1.2)

            # 设定长度阈值，即行中白色像素的最小数量
            length_threshold = expanded_rectangle.shape[1] / 2

            # 初始化变量来存储结果
            long_lines_y_max = 0
            long_lines_y_min = expanded_rectangle.shape[0]

            # 遍历每一行
            for y in range(expanded_rectangle.shape[0]):
                row = expanded_rectangle[y, :]
                non_zero_count = np.count_nonzero(row)  # 计算非零像素的数量

                # 如果当前行的长度超过阈值
                if non_zero_count > length_threshold:
                    # 更新纵坐标的最大值和最小值
                    long_lines_y_max = max(long_lines_y_max, y)
                    long_lines_y_min = min(long_lines_y_min, y)

            cv2.line(expanded_rectangle, (expanded_rectangle.shape[1] // 2, long_lines_y_max),
                     (expanded_rectangle.shape[1] // 2, long_lines_y_min), (255), 1)
            contours2, hierarchy2 = cv2.findContours(expanded_rectangle, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            fff = []
            for cnt in contours2:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                fff.append([x, y, x + w, y + h, area])
            sorted_lst = sorted(fff, key=lambda x: x[4], reverse=True)
            match = sorted_lst[0][:4]

            best_match = [match[0] + need[0], match[1] + need[1], match[2] + need[0], match[3] + need[1]]

    (cx1, cy1, cx2, cy2) = best_match
    new_cx1 = cx1 + rx1
    new_cy1 = cy1 + ry1
    new_cx2 = cx2 + rx1
    new_cy2 = cy2 + ry1
    best_match_box = [new_cx1, new_cy1, new_cx2, new_cy2]
    return best_match_box


def correction_selected_coordinates(img, rect):
    """返回表格矫正后的坐标"""
    new_rect = table_rectangle(img, rect)
    return new_rect

def get_rects_d(rect1_coords, rect2_coords):
    """
        输入两个矩形坐标，返回这两个矩形之间的最短距离
    """
    rect1 = Polygon([(rect1_coords[0], rect1_coords[1]),
                     (rect1_coords[0], rect1_coords[3]),
                     (rect1_coords[2], rect1_coords[3]),
                     (rect1_coords[2], rect1_coords[1])])

    rect2 = Polygon([(rect2_coords[0], rect2_coords[1]),
                     (rect2_coords[0], rect2_coords[3]),
                     (rect2_coords[2], rect2_coords[3]),
                     (rect2_coords[2], rect2_coords[1])])

    return rect1.distance(rect2)

def get_rects(rect1_coords, rect2_coords):
    """
    输入两个矩形坐标，返回这两个矩形中心点之间的距离
    """
    # 计算第一个矩形的中心点
    rect1_center_x = (rect1_coords[0] + rect1_coords[2]) / 2
    rect1_center_y = (rect1_coords[1] + rect1_coords[3]) / 2

    # 计算第二个矩形的中心点
    rect2_center_x = (rect2_coords[0] + rect2_coords[2]) / 2
    rect2_center_y = (rect2_coords[1] + rect2_coords[3]) / 2

    # 计算两个中心点之间的欧几里得距离
    distance = ((rect1_center_x - rect2_center_x) ** 2 +
                (rect1_center_y - rect2_center_y) ** 2) ** 0.5

    return distance

def natural_sort_key(s):
    """
    使用正则匹配文件名中的数字
    假设文件名格式为'1.png'
    """
    int_part = re.search(r'\d+', s).group()
    return int(int_part)

def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)

def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    _COLORS = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.314, 0.717, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32).reshape(-1, 3)
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


class PreProcess(object):
    """
        自动搜索前处理流程：去水印 + 筛选封装图页面
        得到pages_list
    """
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.page_list = None     # 可能含封装图页面列表 程序页码 含0
        self.preprocess()

    def preprocess(self):
        """去水印 + 筛选页面"""
        watermark_remove(self.pdf_path)
        with fitz.open(self.pdf_path) as doc:
            page_count = doc.page_count
        self.page_list= extract_package_page_list(self.pdf_path)
        # 为方便封装图与Form的匹配，扩大封装图对象的搜索范围
        page_list = []
        for i in range(len(self.page_list)):
            page_list.append(self.page_list[i] - 1)
            page_list.append(self.page_list[i])
            page_list.append(self.page_list[i] + 1)

        page_list = set(page_list)
        self.page_list = list(filter(lambda num: num >= 0 and num < page_count, page_list))


class Detect(object):
    """yolox检测"""
    def __init__(self, pdf_path, pages):
        self.pdf_path = pdf_path
        self.pages = pages

        self.current_page = 0     # 当前正在处理第几页
        self.source_data = []         # 存放yolox直接检测结果数据，未匹配前
        self.data = []       # 存放匹配后的封装图对象数据
        self.data2 = []    # data数据的其他格式
        self.have_page = []       # 存放存在封装对象的页码 程序页码
            # {'page', 'pos', 'package_type', 'conf', 'keyview'{'page', 'pos', 'conf'}, 'Top', 'Side',
        # 'Detail', 'Note', 'Form'[{上}, {当前}, {下}]}
        self.source_package_data = []
        self.source_keyview_data = []    # 存放yolox直接检测keyview数据，未匹配前
        self.source_Top_data = []
        self.source_Side_data = []
        self.source_Detail_data = []
        self.source_Note_data = []
        self.source_Form_data = []      # 存放yolox直接检测Form数据，未匹配前

    def pre_process(self):
        """前处理：转图片"""
        pdf2img(self.pdf_path, self.pages)

    def process(self):
        """调用yolox检测"""
        create_dir(SAVE_IMG_PATH)
        img_item_list = os.listdir(IMAGE_PATH)
        img_item_list = sorted(img_item_list, key=natural_sort_key)     # 图片文件排序

        for i in range(len(img_item_list)):
            flag = 0            # 默认不画框
            img_item = img_item_list[i]
            img_path = os.path.join(IMAGE_PATH, img_item)
            # 使用Image读图片 避免opencv中文路径问题
            img_image = Image.open(img_path)
            col_img = cv2.cvtColor(np.asarray(img_image), cv2.COLOR_RGB2BGR)
            origin_img = cv2.cvtColor(np.asarray(img_image), cv2.COLOR_RGB2BGR)
            img, ratio = preprocess(origin_img, INPUT_SHAPE)
            session = onnxruntime.InferenceSession(MODEL)

            ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
            output = session.run(None, ort_inputs)
            predictions = demo_postprocess(output[0], INPUT_SHAPE)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

            if dets is not None:
                for item in dets:
                    if item[4] >= SCORE_THR:
                        flag = 1         # 需要画框
                        break
                if (flag):
                    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                    # 可视化显示
                    origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                                     SCORE_THR, class_names=VOC_CLASSES)  # 返回绘制好的图片
                    # 使用Image进行图片保存，避免opencv中文路径问题
                    img_rgb = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    img_pil.save(os.path.join(SAVE_IMG_PATH, img_item))

                for item in dets:
                    if item[4] >= SCORE_THR:
                        # 对表格类型添加自动矫正
                        if (VOC_CLASSES[int(item[5])] == 'Form'):
                            item[0:4] = correction_selected_coordinates(col_img, [int(item[0]), int(item[1]),
                                                                                  int(item[2]), int(item[3])])

                        self.source_data.append({
                            'page': int(os.path.splitext(img_item)[0]),
                            'type': VOC_CLASSES[int(item[5])],
                            'pos': [int(item[0] / ZOOM[0]), int(item[1] / ZOOM[1]),
                                    int(item[2] / ZOOM[0]), int(item[3] / ZOOM[1])],
                            'conf': item[4]
                        })
                        if (VOC_CLASSES[int(item[5])] == 'package'):
                            self.source_package_data.append({
                                'page': int(os.path.splitext(img_item)[0]),
                                'pos': [int(item[0] / ZOOM[0]), int(item[1] / ZOOM[1]),
                                        int(item[2] / ZOOM[0]), int(item[3] / ZOOM[1])],
                                'conf': item[4],
                                'flag': -1
                            })
                        elif (VOC_CLASSES[int(item[5])] == 'Top'):
                            self.source_Top_data.append({
                                'page': int(os.path.splitext(img_item)[0]),
                                'pos': [int(item[0] / ZOOM[0]), int(item[1] / ZOOM[1]),
                                        int(item[2] / ZOOM[0]), int(item[3] / ZOOM[1])],
                                'conf': item[4],
                                'flag': -1
                            })
                        elif (VOC_CLASSES[int(item[5])] == 'Side'):
                            self.source_Side_data.append({
                                'page': int(os.path.splitext(img_item)[0]),
                                'pos': [int(item[0] / ZOOM[0]), int(item[1] / ZOOM[1]),
                                        int(item[2] / ZOOM[0]), int(item[3] / ZOOM[1])],
                                'conf': item[4],
                                'flag': -1
                            })
                        elif (VOC_CLASSES[int(item[5])] == 'Detail'):
                            self.source_Detail_data.append({
                                'page': int(os.path.splitext(img_item)[0]),
                                'pos': [int(item[0] / ZOOM[0]), int(item[1] / ZOOM[1]),
                                        int(item[2] / ZOOM[0]), int(item[3] / ZOOM[1])],
                                'conf': item[4],
                                'flag': -1
                            })
                        elif (VOC_CLASSES[int(item[5])] == 'Note'):
                            self.source_Note_data.append({
                                'page': int(os.path.splitext(img_item)[0]),
                                'pos': [int(item[0] / ZOOM[0]), int(item[1] / ZOOM[1]),
                                        int(item[2] / ZOOM[0]), int(item[3] / ZOOM[1])],
                                'conf': item[4],
                                'flag': -1
                            })
                        elif (VOC_CLASSES[int(item[5])] == 'Form'):
                            self.source_Form_data.append({
                                'page': int(os.path.splitext(img_item)[0]),
                                'pos': [int(item[0] / ZOOM[0]), int(item[1] / ZOOM[1]),
                                        int(item[2] / ZOOM[0]), int(item[3] / ZOOM[1])],
                                'conf': item[4],
                                'flag': -1
                            })
                        else:
                            self.source_keyview_data.append({
                                'page': int(os.path.splitext(img_item)[0]),
                                'pos': [int(item[0] / ZOOM[0]), int(item[1] / ZOOM[1]),
                                        int(item[2] / ZOOM[0]), int(item[3] / ZOOM[1])],
                                'package_type': VOC_CLASSES[int(item[5])],
                                'conf': item[4],
                                'flag': -1
                            })

            self.current_page += 1       # 处理下一页

    def judge_other_package(self, page):
        """判断该页上是否有多个封装图，若有2个以上封装图，则返回True，否则返回False"""
        count = 0
        for i in range(len(self.source_package_data)):
            if (page == self.source_package_data[i]['page']):
                count += 1
        return True if count > 1 else False

    def judge_package(self, package):
        """判断该封装图信息是否满足封装图对象"""
        flag1 = 1 if package['package_type'] != None else 0
        flag2 = 1 if package['Top'] != None else 0
        flag3 = 1 if len(package['Side']) else 0
        flag4 = 1 if len(package['Detail']) else 0
        return True if ((flag1 + flag2 + flag3 + flag4) >= 2) else False

    def post_process(self):
        """后处理：从封装图，部分视图，表构建封装图对象 + 图表匹配 + 生成json"""
        self.package_match_view()
        self.get_new_pos()
        self.data2json()
        self.transfer_data()
        self.process_pages()
        self.generate_detect_pdf()
        self.clean_file()

    def package_match_view(self):
        """封装图匹配视图"""
        self.source_package_data = sorted(self.source_package_data, key=lambda x: (x['page'], x['pos'][1]))
        package_data = []     # 存放封装图匹配信息的中间变量
                # {'page', 'pos', 'conf', 'package_type', 'keyview'{'page', 'pos', 'conf'}, 'Top', 'Side'
        for i in range(len(self.source_package_data)):
            package_data.append({
                'page': self.source_package_data[i]['page'],
                'pos': self.source_package_data[i]['pos'],
                'conf': self.source_package_data[i]['conf'],
                'package_type': None,
                'keyview': None,
                'Top': None,
                'Side': [],
                'Detail': [],
                'Note': [],
                'Form': [{}, {}, {}]
            })
        # 封装图与关键特征视图信息匹配
        for i in range(len(self.source_package_data)):
            # 由于关键特征视图和Top视图只需要匹配一项，故需要从符合要求的项中寻找最佳
            keyview_data = []   # {'package_type', 'pos', 'conf', 'index'}
            Top_data = []     # {'pos', 'conf', 'index'}
            current_page_num = self.source_package_data[i]['page']   # 当前待匹配封装图所在页码
            package_pos = self.source_package_data[i]['pos']     # 当前待匹配封装图位置坐标
            # 开始匹配关键特征视图
            for j in range(len(self.source_keyview_data)):
                flag = self.source_keyview_data[j]['flag']
                keyview_page = self.source_keyview_data[j]['page']
                keyview_pos = self.source_keyview_data[j]['pos']
                if ((flag < 0)
                        and (keyview_page == current_page_num)
                        and (int(get_rects_d(package_pos, keyview_pos)) == 0)):
                    keyview_data.append({
                        'package_type': self.source_keyview_data[j]['package_type'],
                        'pos': keyview_pos,
                        'conf': self.source_keyview_data[j]['conf'],
                        'index': j
                    })
                if (keyview_page > current_page_num):
                    break
            # 从该封装图的关键特征视图数据集中挑选最佳选项
            if (len(keyview_data) == 0):
                pass
            elif (len(keyview_data) == 1):
                package_data[i]['package_type'] = keyview_data[0]['package_type']
                package_data[i]['keyview'] = {'pos': keyview_data[0]['pos'],
                    'conf': keyview_data[0]['conf']}
                # 修改源数据
                self.source_keyview_data[keyview_data[0]['index']]['flag'] = 1  # 该关键特征视图已匹配
            else:
                max_conf = keyview_data[0]['conf']
                max_index = 0
                for k in range(1, len(keyview_data)):
                    if (keyview_data[k]['conf'] > max_conf):
                        max_conf = keyview_data[k]['conf']
                        max_index = k
                package_data[i]['package_type'] = keyview_data[max_index]['package_type']
                package_data[i]['keyview'] = {'pos': keyview_data[max_index]['pos'],
                                              'conf': keyview_data[max_index]['conf']}
                # 修改源数据
                self.source_keyview_data[keyview_data[max_index]['index']]['flag'] = 1  # 该关键特征视图已匹配
            # 开始匹配Top视图
            for j in range(len(self.source_Top_data)):
                flag = self.source_Top_data[j]['flag']
                Top_page = self.source_Top_data[j]['page']
                Top_pos = self.source_Top_data[j]['pos']
                if ((flag < 0)
                        and (Top_page == current_page_num)
                        and (int(get_rects_d(Top_pos, package_pos)) == 0)):
                    Top_data.append({
                        'pos': Top_pos,
                        'conf': self.source_Top_data[j]['conf'],
                        'index': j
                    })
                if (Top_page > current_page_num):
                    break
            # 从该封装图的Top视图数据集中挑选最佳选项
            if (len(Top_data) == 0):
                pass
            elif (len(Top_data) == 1):
                package_data[i]['Top'] = {'pos': Top_data[0]['pos'],
                                              'conf': Top_data[0]['conf']}
                # 修改源数据
                self.source_Top_data[Top_data[0]['index']]['flag'] = 1  # 该关键特征视图已匹配
            else:
                max_conf = Top_data[0]['conf']
                max_index = 0
                for k in range(1, len(Top_data)):
                    if (Top_data[k]['conf'] > max_conf):
                        max_conf = Top_data[k]['conf']
                        max_index = k
                package_data[i]['Top'] = {'pos': Top_data[max_index]['pos'],
                                              'conf': Top_data[max_index]['conf']}
                # 修改源数据
                self.source_Top_data[Top_data[max_index]['index']]['flag'] = 1  # 该关键特征视图已匹配
            # 开始匹配Side视图
            for j in range(len(self.source_Side_data)):
                flag = self.source_Side_data[j]['flag']
                Side_page = self.source_Side_data[j]['page']
                Side_pos = self.source_Side_data[j]['pos']
                if ((flag < 0)
                        and (Side_page == current_page_num)
                        and (int(get_rects_d(Side_pos, package_pos)) == 0)):
                    package_data[i]['Side'].append({
                        'pos': self.source_Side_data[j]['pos'],
                        'conf': self.source_Side_data[j]['conf']
                    })
                    # 修改源数据
                    self.source_Side_data[j]['flag'] = 1
                if (Side_page > current_page_num):
                    break
            # 开始匹配Detail视图
            for j in range(len(self.source_Detail_data)):
                flag = self.source_Detail_data[j]['flag']
                Detail_page = self.source_Detail_data[j]['page']
                Detail_pos = self.source_Detail_data[j]['pos']
                if ((flag < 0)
                        and (Detail_page == current_page_num)
                        and (int(get_rects_d(Detail_pos, package_pos)) == 0)):
                    package_data[i]['Detail'].append({
                        'pos': self.source_Detail_data[j]['pos'],
                        'conf': self.source_Detail_data[j]['conf']
                    })
                    # 修改源数据
                    self.source_Detail_data[j]['flag'] = 1
                if (Detail_page > current_page_num):
                    break
            # 开始匹配Note视图
            for j in range(len(self.source_Note_data)):
                flag = self.source_Note_data[j]['flag']
                Note_page = self.source_Note_data[j]['page']
                Note_pos = self.source_Note_data[j]['pos']
                if ((flag < 0)
                        and (Note_page == current_page_num)
                        and (int(get_rects_d(Note_pos, package_pos)) == 0)):
                    package_data[i]['Note'].append({
                        'pos': self.source_Note_data[j]['pos'],
                        'conf': self.source_Note_data[j]['conf']
                    })
                    # 修改源数据
                    self.source_Note_data[j]['flag'] = 1
                if (Note_page > current_page_num):
                    break
        # 封装图数据预筛选
        # 从package_data中获取待删除信息，对source_package_data进行修改
        del_index = []
        for i in range(len(package_data)):
            if not self.judge_package(package_data[i]):
                del_index.append(i)
        package_data = [item for index, item in enumerate(package_data) if index not in del_index]
        self.source_package_data = [item for index, item in enumerate(self.source_package_data)
                                    if index not in del_index]
        for i in range(len(self.source_package_data)):
            current_page_num = self.source_package_data[i]['page']  # 当前待匹配封装图所在页码
            package_pos = self.source_package_data[i]['pos']  # 当前待匹配封装图位置坐标
            # 开始匹配Form
            # 判断是否是单页多封装图 -> 当页封装表匹配 -> 上页封装表匹配 -> 下页封装表匹配
            if (not self.judge_other_package(current_page_num)):
                # 上页封装表匹配
                pre_Form_data = []
                pre_package_page = current_page_num - 1
                for j in range(len(self.source_Form_data)):
                    flag = self.source_Form_data[j]['flag']
                    Form_page = self.source_Form_data[j]['page']
                    if ((flag < 0)
                            and (Form_page == pre_package_page)):
                        pre_Form_data.append({
                        'pos': self.source_Form_data[j]['pos'],
                        'conf': self.source_Form_data[j]['conf'],
                        'index': j
                    })
                    if (Form_page > pre_package_page):
                        break
                if (len(pre_Form_data) == 0):
                    pass
                elif (len(pre_Form_data) == 1):
                    package_data[i]['Form'][0] = {'pos': pre_Form_data[0]['pos'], 'page': pre_package_page,
                                                  'conf': pre_Form_data[0]['conf']}
                    # 修改源数据
                    self.source_Form_data[pre_Form_data[0]['index']]['flag'] = -1  # 该表格已匹配
                else:
                    bottom_y = pre_Form_data[0]['pos'][3]
                    min_index = 0
                    for k in range(len(pre_Form_data)):
                        if (pre_Form_data[k]['pos'][3] > bottom_y):
                            bottom_y = pre_Form_data[k]['pos'][3]
                            min_index = k
                    package_data[i]['Form'][0] = {'pos': pre_Form_data[min_index]['pos'], 'page': pre_package_page,
                                                  'conf': pre_Form_data[min_index]['conf']}

                    # 修改源数据
                    self.source_Form_data[pre_Form_data[min_index]['index']]['flag'] = -1  # 该表格已匹配

                # 下页封装表匹配
                next_Form_data = []
                next_package_page = current_page_num + 1
                for j in range(len(self.source_Form_data)):
                    flag = self.source_Form_data[j]['flag']
                    Form_page = self.source_Form_data[j]['page']
                    if ((flag < 0)
                            and (Form_page == next_package_page)):
                        next_Form_data.append({
                            'pos': self.source_Form_data[j]['pos'],
                            'conf': self.source_Form_data[j]['conf'],
                            'index': j
                        })
                    if (Form_page > next_package_page):
                        break
                if (len(next_Form_data) == 0):
                    pass
                elif (len(next_Form_data) == 1):
                    package_data[i]['Form'][2] = {'pos': next_Form_data[0]['pos'], 'page': next_package_page,
                                                  'conf': next_Form_data[0]['conf']}
                    # 修改源数据
                    self.source_Form_data[next_Form_data[0]['index']]['flag'] = -1  # 该表格已匹配
                else:
                    top_y = next_Form_data[0]['pos'][1]
                    min_index = 0
                    for k in range(len(next_Form_data)):
                        if (next_Form_data[k]['pos'][1] < top_y):
                            top_y = next_Form_data[k]['pos'][1]
                            min_index = k
                    package_data[i]['Form'][2] = {'pos': next_Form_data[min_index]['pos'], 'page': next_package_page,
                                                  'conf': next_Form_data[min_index]['conf']}

                    # 修改源数据
                    self.source_Form_data[next_Form_data[min_index]['index']]['flag'] = -1  # 该表格已匹配
            # 当页封装表匹配
            cur_Form_data = []
            for j in range(len(self.source_Form_data)):
                flag = self.source_Form_data[j]['flag']
                Form_page = self.source_Form_data[j]['page']
                if ((flag < 0)
                        and (Form_page == current_page_num)):
                    cur_Form_data.append({
                        'pos': self.source_Form_data[j]['pos'],
                        'conf': self.source_Form_data[j]['conf'],
                        'index': j
                    })
                if (Form_page > current_page_num):
                    break
            if (len(cur_Form_data) == 0):
                pass
            elif (len(cur_Form_data) == 1):
                package_data[i]['Form'][1] = {'pos': cur_Form_data[0]['pos'], 'page': current_page_num,
                                              'conf': cur_Form_data[0]['conf']}
                # 修改源数据
                self.source_Form_data[cur_Form_data[0]['index']]['flag'] = -1  # 该表格已匹配
            else:
                min_d = get_rects_d(cur_Form_data[0]['pos'], package_pos)
                min_index = 0
                for k in range(1, len(cur_Form_data)):
                    if (get_rects_d(cur_Form_data[k]['pos'], package_pos) < min_d):
                        min_d = get_rects_d(cur_Form_data[k]['pos'], package_pos)
                        min_index = k
                package_data[i]['Form'][1] = {'pos': cur_Form_data[min_index]['pos'], 'page': current_page_num,
                                              'conf': cur_Form_data[min_index]['conf']}

                # 修改源数据
                self.source_Form_data[cur_Form_data[min_index]['index']]['flag'] = -1  # 该表格已匹配

        # 数据过滤
        self.data = package_data

    def get_new_pos(self):
        """生成pdf坐标"""
        with fitz.open(self.pdf_path) as doc:
            for i in range(len(self.data)):
                cur_page_num = self.data[i]['page']
                rotation = doc[cur_page_num - 1].rotation
                cur_page_width = doc[cur_page_num - 1].rect[2]    # 页宽
                # 添加package_new_pos
                package_new_pos = [0, 0, 0, 0]
                if (rotation == 0):
                    package_new_pos = self.data[i]['pos']
                else:        # 转换坐标系
                    package_new_pos[0] = self.data[i]['pos'][1]
                    package_new_pos[1] = cur_page_width - self.data[i]['pos'][0]
                    package_new_pos[2] = self.data[i]['pos'][3]
                    package_new_pos[3] = cur_page_width - self.data[i]['pos'][2]
                self.data[i]['new_pos'] = package_new_pos
                # 添加keyview_new_pos
                if self.data[i]['keyview'] is not None:
                    keyview_new_pos = [0, 0, 0, 0]
                    if (rotation == 0):
                        keyview_new_pos = self.data[i]['keyview']['pos']
                    else:
                        keyview_new_pos[0] = self.data[i]['keyview']['pos'][1]
                        keyview_new_pos[1] = cur_page_width - self.data[i]['keyview']['pos'][0]
                        keyview_new_pos[2] = self.data[i]['keyview']['pos'][3]
                        keyview_new_pos[3] = cur_page_width - self.data[i]['keyview']['pos'][2]
                    self.data[i]['keyview']['new_pos'] = keyview_new_pos
                # 添加Top_new_pos
                if self.data[i]['Top'] is not None:
                    Top_new_pos = [0, 0, 0, 0]
                    if (rotation == 0):
                        Top_new_pos = self.data[i]['Top']['pos']
                    else:
                        Top_new_pos[0] = self.data[i]['Top']['pos'][1]
                        Top_new_pos[1] = cur_page_width - self.data[i]['Top']['pos'][0]
                        Top_new_pos[2] = self.data[i]['Top']['pos'][3]
                        Top_new_pos[3] = cur_page_width - self.data[i]['Top']['pos'][2]
                    self.data[i]['Top']['new_pos'] = Top_new_pos
                # 添加Side_new_pos
                for j in range(len(self.data[i]['Side'])):
                    Side_new_pos = [0, 0, 0, 0]
                    if (rotation == 0):
                        Side_new_pos = self.data[i]['Side'][j]['pos']
                    else:
                        Side_new_pos[0] = self.data[i]['Side'][j]['pos'][1]
                        Side_new_pos[1] = cur_page_width - self.data[i]['Side'][j]['pos'][0]
                        Side_new_pos[2] = self.data[i]['Side'][j]['pos'][3]
                        Side_new_pos[3] = cur_page_width - self.data[i]['Side'][j]['pos'][2]
                    self.data[i]['Side'][j]['new_pos'] = Side_new_pos
                # 添加Detail_new_pos
                for j in range(len(self.data[i]['Detail'])):
                    Detail_new_pos = [0, 0, 0, 0]
                    if (rotation == 0):
                        Detail_new_pos = self.data[i]['Detail'][j]['pos']
                    else:
                        Detail_new_pos[0] = self.data[i]['Detail'][j]['pos'][1]
                        Detail_new_pos[1] = cur_page_width - self.data[i]['Detail'][j]['pos'][0]
                        Detail_new_pos[2] = self.data[i]['Detail'][j]['pos'][3]
                        Detail_new_pos[3] = cur_page_width - self.data[i]['Detail'][j]['pos'][2]
                    self.data[i]['Detail'][j]['new_pos'] = Detail_new_pos
                # 添加Note_new_pos
                for j in range(len(self.data[i]['Note'])):
                    Note_new_pos = [0, 0, 0, 0]
                    if (rotation == 0):
                        Note_new_pos = self.data[i]['Note'][j]['pos']
                    else:
                        Note_new_pos[0] = self.data[i]['Note'][j]['pos'][1]
                        Note_new_pos[1] = cur_page_width - self.data[i]['Note'][j]['pos'][0]
                        Note_new_pos[2] = self.data[i]['Note'][j]['pos'][3]
                        Note_new_pos[3] = cur_page_width - self.data[i]['Note'][j]['pos'][2]
                    self.data[i]['Note'][j]['new_pos'] = Note_new_pos
                # 添加Form_new_pos
                # 添加pre_new_pos
                if (len(self.data[i]['Form'][0])):
                    pre_page_num = self.data[i]['Form'][0]['page']
                    rotation = doc[pre_page_num - 1].rotation
                    pre_page_width = doc[pre_page_num - 1].rect[2]
                    pre_Form_new_pos = [0, 0, 0, 0]
                    if (rotation == 0):
                        pre_Form_new_pos = self.data[i]['Form'][0]['pos']
                    else:
                        pre_Form_new_pos[0] = self.data[i]['Form'][0]['pos'][1]
                        pre_Form_new_pos[1] = pre_page_width - self.data[i]['Form'][0]['pos'][0]
                        pre_Form_new_pos[2] = self.data[i]['Form'][0]['pos'][3]
                        pre_Form_new_pos[3] = pre_page_width - self.data[i]['Form'][0]['pos'][2]
                    self.data[i]['Form'][0]['new_pos'] = pre_Form_new_pos
                # 添加cur_new_pos
                if (len(self.data[i]['Form'][1])):
                    cur_page_num = self.data[i]['Form'][1]['page']
                    rotation = doc[cur_page_num - 1].rotation
                    cur_page_width = doc[cur_page_num - 1].rect[2]
                    cur_Form_new_pos = [0, 0, 0, 0]
                    if (rotation == 0):
                        cur_Form_new_pos = self.data[i]['Form'][1]['pos']
                    else:
                        cur_Form_new_pos[0] = self.data[i]['Form'][1]['pos'][1]
                        cur_Form_new_pos[1] = cur_page_width - self.data[i]['Form'][1]['pos'][0]
                        cur_Form_new_pos[2] = self.data[i]['Form'][1]['pos'][3]
                        cur_Form_new_pos[3] = cur_page_width - self.data[i]['Form'][1]['pos'][2]
                    self.data[i]['Form'][1]['new_pos'] = cur_Form_new_pos
                # 添加next_new_pos
                if (len(self.data[i]['Form'][2])):
                    next_page_num = self.data[i]['Form'][2]['page']
                    rotation = doc[next_page_num - 1].rotation
                    next_page_width = doc[next_page_num - 1].rect[2]
                    next_Form_new_pos = [0, 0, 0, 0]
                    if (rotation == 0):
                        next_Form_new_pos = self.data[i]['Form'][2]['pos']
                    else:
                        next_Form_new_pos[0] = self.data[i]['Form'][2]['pos'][1]
                        next_Form_new_pos[1] = next_page_width - self.data[i]['Form'][2]['pos'][0]
                        next_Form_new_pos[2] = self.data[i]['Form'][2]['pos'][3]
                        next_Form_new_pos[3] = next_page_width - self.data[i]['Form'][2]['pos'][2]
                    self.data[i]['Form'][2]['new_pos'] = next_Form_new_pos

    def data2json(self):
        """将匹配好的信息写入json"""
        pass

    def transfer_data(self):
        """将封装图对象数据转为指定格式"""
        for i in range(len(self.data)):
            self.have_page.append(self.data[i]['page'] - 1)
            part_content = []
            # 添加关键特征视图
            if (self.data[i]['package_type']):
                part_content.append({
                    'part_name': self.data[i]['package_type'],
                    'yolo_part_name': self.data[i]['package_type'],
                    'page': self.data[i]['page'] - 1,
                    'rect': self.data[i]['keyview']['pos'],
                    'new_rect': self.data[i]['keyview']['new_pos']
                })
            # 添加Top视图
            if (self.data[i]['Top']):
                part_content.append({
                    'part_name': 'Top',
                    'yolo_part_name': 'Top',
                    'page': self.data[i]['page'] - 1,
                    'rect': self.data[i]['Top']['pos'],
                    'new_rect': self.data[i]['Top']['new_pos']
                })
            # 添加Side视图
            for j in range(len(self.data[i]['Side'])):
                part_content.append({
                    'part_name': 'Side',
                    'yolo_part_name': 'Side',
                    'page': self.data[i]['page'] - 1,
                    'rect': self.data[i]['Side'][j]['pos'],
                    'new_rect': self.data[i]['Side'][j]['new_pos']
                })
            # 添加Detail视图
            for j in range(len(self.data[i]['Detail'])):
                part_content.append({
                    'part_name': 'Detail',
                    'yolo_part_name': 'Detail',
                    'page': self.data[i]['page'] - 1,
                    'rect': self.data[i]['Detail'][j]['pos'],
                    'new_rect': self.data[i]['Detail'][j]['new_pos']
                })
            # 添加Note
            for j in range(len(self.data[i]['Note'])):
                part_content.append({
                    'part_name': 'Note',
                    'yolo_part_name': 'Note',
                    'page': self.data[i]['page'] - 1,
                    'rect': self.data[i]['Note'][j]['pos'],
                    'new_rect': self.data[i]['Note'][j]['new_pos']
                })
            # 添加Form
            # 添加上Form
            if (len(self.data[i]['Form'][0])):
                self.have_page.append(self.data[i]['page'] - 2)
                part_content.append({
                    'part_name': 'Form',
                    'yolo_part_name': 'Form',
                    'page': self.data[i]['page'] - 2,
                    'rect': self.data[i]['Form'][0]['pos'],
                    'new_rect': self.data[i]['Form'][0]['new_pos']
                })
            # 添加当前Form
            if (len(self.data[i]['Form'][1])):
                part_content.append({
                    'part_name': 'Form',
                    'yolo_part_name': 'Form',
                    'page': self.data[i]['page'] - 1,
                    'rect': self.data[i]['Form'][1]['pos'],
                    'new_rect': self.data[i]['Form'][1]['new_pos']
                })
            # 添加下Form
            if (len(self.data[i]['Form'][2])):
                self.have_page.append(self.data[i]['page'])
                part_content.append({
                    'part_name': 'Form',
                    'yolo_part_name': 'Form',
                    'page': self.data[i]['page'],
                    'rect': self.data[i]['Form'][2]['pos'],
                    'new_rect': self.data[i]['Form'][2]['new_pos']
                })
            data_dict = {
                'page': self.data[i]['page'] - 1,
                'type': 'img',
                'rect': self.data[i]['pos'],
                'new_rect': self.data[i]['new_pos'],
                'package_type': self.data[i]['package_type'],
                'source': 'manual',
                'part_content': part_content if len(part_content) else None,
                'reco_content': None
            }
            self.data2.append(data_dict)
        self.have_page = list(set(self.have_page))
        self.have_page.sort()
    def sort_package_by_position(self, package):
        """
        根据package的位置进行排序（从上到下）
        """
        x0, y0, x1, y1 = package['rect']
        return y0  # 只按y坐标（从上到下）排序

    def sort_data2(self):
        """
        对data2中的数据进行排序
        """
        # 按页码排序
        self.data2.sort(key=lambda x: x['page'])

        # 对同一页的package进行排序
        current_page = None
        page_start_idx = 0

        for i, package in enumerate(self.data2):
            if current_page != package['page']:
                # 对上一页的package进行排序
                if current_page is not None:
                    self.data2[page_start_idx:i] = sorted(
                        self.data2[page_start_idx:i],
                        key=self.sort_package_by_position
                    )
                current_page = package['page']
                page_start_idx = i

        # 对最后一页的package进行排序
        if current_page is not None:
            self.data2[page_start_idx:] = sorted(
                self.data2[page_start_idx:],
                key=self.sort_package_by_position
            )

    def process_page_keywords_and_match(self, pdf_path, page_num, keywords):
        """
        判断页面是否可编辑，获取所有关键字，去重，合并视图关键字，并进行类型和视图关键字匹配
        """
        is_editable = is_page_editable(pdf_path, page_num + 1)
        all_results = []
        if is_editable:
            print(f"第 {page_num} 页是可编辑的，使用文本搜索方式")
            results = search_keywords_in_editable_page(pdf_path, page_num + 1, keywords)
            found_keywords = set(result['keyword'] for result in results)
            if not any(kw in found_keywords for kw in ["TOP", "SIDE", "DETAIL", "VIEW"]):
                # print(f"在文本中未找到视图关键字，开始对页面中的所有 package 进行 OCR 识别")
                page_packages = [item for item in self.data2 if item['page'] == page_num]

                doc = fitz.open(pdf_path)
                if page_num >= 0 and page_num < len(doc):
                    page = doc[page_num]
                    for package_idx, package in enumerate(page_packages):
                        rect = package['rect']
                        zoom = 3
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False, clip=rect)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        # img.save(f"1.png")
                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        package_results = process_package_image(img_cv, keywords, page_num)
                        scaled_results = []
                        for result_idx, result in enumerate(package_results):
                            x0, y0, x1, y1 = result['coordinates']
                            scaled_x0 = int(x0 / zoom) + rect[0]
                            scaled_y0 = int(y0 / zoom) + rect[1]
                            scaled_x1 = int(x1 / zoom) + rect[0]
                            scaled_y1 = int(y1 / zoom) + rect[1]
                            scaled_results.append({
                                'page': page_num,
                                'keyword': result['keyword'],
                                'coordinates': (scaled_x0, scaled_y0, scaled_x1, scaled_y1),
                                'type': result['type'],
                                'content': result['content']
                            })
                        results.extend(scaled_results)

                doc.close()
        else:
            # 然后再进行OCR识别
            results = process_non_editable_page(pdf_path, page_num + 1, keywords)
        all_results.extend(results)
        all_results = self.find_adjacent_view_keywords(all_results)

        # 坐标完全相同的数据去重（只保留一个）
        unique_results = []
        seen_coords = set()
        for item in all_results:
            coords_tuple = tuple(item['coordinates'])
            if coords_tuple not in seen_coords:
                unique_results.append(item)
                seen_coords.add(coords_tuple)
        all_results = unique_results

        # 匹配
        print(f"开始处理页面 {page_num} 的package与类型关键字匹配")
        self.match_package_with_type(page_num, all_results)
        print(f"开始处理页面 {page_num} 的package与视图关键字匹配")
        page_packages = [item for item in self.data2 if item['page'] == page_num]
        for package in page_packages:
            self.match_package_with_view(package, all_results)

    def transform_coordinates(self, rect, angle, original_width, original_height):
        x0, y0, x1, y1 = rect

        if angle == 0:
            return rect
        elif angle == 90:
            # 顺时针90度：
            return [
                original_height - y1,  # 新x0
                x0,  # 新y0
                original_height - y0,  # 新x1
                x1  # 新y1
            ]
        elif angle == 180:
            # 顺时针180度：x' = width - x, y' = height - y
            return [
                original_width - x1,  # 新x0
                original_height - y1,  # 新y0
                original_width - x0,  # 新x1
                original_height - y0  # 新y1
            ]
        elif angle == 270:
            # 顺时针270度：x' = y, y' = width - x
            return [
                y0,  # 新x0
                original_width - x1,  # 新y0
                y1,  # 新x1
                original_width - x0  # 新y1
            ]
        else:
            return rect

    def process_pages(self):
        output_path = 'output.json'

        # 清空文件内容
        with open(output_path, 'w', encoding='utf-8') as f:
            pass  # 打开即关闭，文件内容被清空

        all_pages = sorted(set(item['page'] for item in self.data2))
        keywords = ["BGA", "DFN", "SON", "QFP", "QFN", "SOP", "SOT", "SOIC",
                    "PLASTIC BALL GRID ARRAY", "Plastic Quad Flat Package","Quad Flatpack","TOPVIEW", "SIDEVIEW",
                    "TOP VIEW", "SIDE VIEW", "TOP", "SIDE", "VIEW", "DETAIL"]

        for page_num in all_pages:
            pdf_path = self.pdf_path
            page_number = page_num

            # 查找该页所有Form表格
            form_rects = []
            for item in self.data2:
                for part in item['part_content']:
                    if part['page'] == page_num and part['part_name'] == 'Form':
                        form_rects.append(part['new_rect'])
            # 1. 无表格
            if len(form_rects) == 0:
                self.process_page_keywords_and_match(pdf_path, page_num, keywords)
                continue

            # 2. 多个表格
            elif len(form_rects) > 1:
                self.process_page_keywords_and_match(pdf_path, page_num, keywords)
                continue

            elif len(form_rects) == 1:
                Coordinate = form_rects[0]
                rotate_angle = rotate(pdf_path, page_number + 1, Coordinate)
                print(f"旋转角度：{rotate_angle}")
                if rotate_angle != 90:
                    self.process_page_keywords_and_match(pdf_path, page_num, keywords)
                else:
                    with fitz.open(pdf_path) as doc:
                        original_width, original_height = doc[page_num].rect.width, doc[page_num].rect.height
                    # rotated_angle = 90
                    output_path = f"output/rotated_{page_num}.pdf"
                    rotated_width, rotated_height = rotate_pdf_page(pdf_path, page_num, rotate_angle, output_path)
                    # 坐标批量转换
                    for item in self.data2:
                        if item['page'] == page_number:
                            item['rect'] = self.transform_coordinates(item['rect'], rotate_angle, original_width,
                                                                      original_height)
                            item['new_rect'] = self.transform_coordinates(item['new_rect'], rotate_angle,
                                                                          original_width, original_height)
                            if item.get('part_content'):
                                for part in item['part_content']:
                                    part['rect'] = self.transform_coordinates(part['rect'], rotate_angle,
                                                                              original_width, original_height)
                                    part['new_rect'] = self.transform_coordinates(part['new_rect'], rotate_angle,
                                                                                  original_width, original_height)

                    self.process_page_keywords_and_match(output_path, page_num, keywords)
                    inverse_angle = (360 - rotate_angle) % 360
                    for item in self.data2:
                        if item['page'] == page_number:
                            item['rect'] = self.transform_coordinates(item['rect'], inverse_angle, rotated_width,
                                                                      rotated_height)
                            item['new_rect'] = self.transform_coordinates(item['new_rect'], inverse_angle,
                                                                          rotated_width, rotated_height)
                            if item.get('part_content'):
                                for part in item['part_content']:
                                    part['rect'] = self.transform_coordinates(part['rect'], inverse_angle,
                                                                              rotated_width, rotated_height)
                                    part['new_rect'] = self.transform_coordinates(part['new_rect'], inverse_angle,
                                                                                  rotated_width, rotated_height)

        # 4. 遍历data2中每个part_content，处理part_name和package_type
        yolo_keywords = ["BGA", "DFN_SON", "QFP", "QFN", "SOP", "DFN", "SON"]
        for pkg in self.data2:
            part_content = pkg.get('part_content', [])
            if part_content:
                part_names = [part['part_name'] for part in part_content]
                has_yolo_keyword = any(name in yolo_keywords for name in part_names)
                if not has_yolo_keyword:
                    for part in part_content:
                        if part['part_name'] == 'TOPVIEW':
                            part['part_name'] = pkg.get('package_type', part['part_name'])
            if pkg.get('package_type') == 'DFN_SON':
                pkg['package_type'] = 'SON'
                for part in part_content:
                    if part['part_name'] == 'DFN_SON':
                        part['part_name'] = 'SON'
            elif pkg.get('package_type') == 'PLASTIC BALL GRID ARRAY':
                pkg['package_type'] = 'BGA'
                for part in part_content:
                    if part['part_name'] == 'PLASTIC BALL GRID ARRAY':
                        part['part_name'] = 'BGA'

            elif pkg.get('package_type') == 'Plastic Quad Flat Package':
                pkg['package_type'] = 'QFP'
                for part in part_content:
                    if part['part_name'] == 'Plastic Quad Flat Package':
                        part['part_name'] = 'QFP'
            elif pkg.get('package_type') == 'Quad Flatpack':
                pkg['package_type'] = 'QFP'
                for part in part_content:
                    if part['part_name'] == 'Quad Flatpack':
                        part['part_name'] = 'QFP'
        for pkg in self.data2:
            part_content = pkg.get('part_content', [])
            if part_content:
                part_names = [part['part_name'] for part in part_content]
                has_side = any(name in ['Side', 'SIDEVIEW'] for name in part_names)
                has_QFP = any(name in ['QFP'] for name in part_names)
                if not has_side:
                    for part in part_content:
                        if part.get('yolo_part_name') == 'Side':
                            part['part_name'] = 'Side'
                if not has_QFP:
                    for part in part_content:
                        if part.get('yolo_part_name') == 'QFP':
                            part['part_name'] = 'QFP'
        # with open('output.txt', 'a', encoding='utf-8') as txt_file:
        #     # 直接将JSON字符串写入
        #     txt_file.write(json.dumps(self.data2, indent=4, ensure_ascii=False))
    def find_adjacent_view_keywords(self, keywords_list, vertical_distance_threshold=10):
        """
        查找同一行内的TOP/VIEW和SIDE/VIEW组合，基于水平投影和垂直投影
        支持任意顺序，遍历所有TOP/SIDE和VIEW的组合
        """
        # 按y坐标排序
        sorted_keywords = sorted(keywords_list, key=lambda x: x['coordinates'][1])
        merged_keywords = []
        used_indices = set()
        n = len(sorted_keywords)
        # 先合并所有TOP/SIDE和VIEW的组合
        for i in range(n):
            kw1 = sorted_keywords[i]
            kw1_text = kw1['keyword'].upper()
            if kw1_text in ['TOP', 'SIDE']:
                for j in range(n):
                    if i == j or j in used_indices:
                        continue
                    kw2 = sorted_keywords[j]
                    kw2_text = kw2['keyword'].upper()
                    if kw2_text == 'VIEW':
                        # 计算垂直重叠
                        y0_1, y1_1 = kw1['coordinates'][1], kw1['coordinates'][3]
                        y0_2, y1_2 = kw2['coordinates'][1], kw2['coordinates'][3]
                        vertical_overlap = min(y1_1, y1_2) - max(y0_1, y0_2)
                        height_1 = y1_1 - y0_1
                        vertical_overlap_ratio = vertical_overlap / height_1 if height_1 > 0 else 0
                        if vertical_overlap_ratio >= 0.8:
                            # 计算水平间距
                            x1_1, x0_2 = kw1['coordinates'][2], kw2['coordinates'][0]
                            x1_2, x0_1 = kw2['coordinates'][2], kw1['coordinates'][0]
                            horizontal_gap = min(abs(x1_1 - x0_2), abs(x1_2 - x0_1))
                            if horizontal_gap <= vertical_distance_threshold:
                                # 合并
                                merged_coords = (
                                    min(kw1['coordinates'][0], kw2['coordinates'][0]),
                                    min(kw1['coordinates'][1], kw2['coordinates'][1]),
                                    max(kw1['coordinates'][2], kw2['coordinates'][2]),
                                    max(kw1['coordinates'][3], kw2['coordinates'][3])
                                )
                                merged_keywords.append({
                                    'page': kw1['page'],
                                    'keyword': f"{kw1_text}VIEW",
                                    'coordinates': merged_coords,
                                    'type': kw1['type'],
                                    'content': f"{kw1['content']} {kw2['content']}".strip()
                                })
                                used_indices.add(i)
                                used_indices.add(j)
                                break
        # 添加未被合并的关键字
        for idx, kw in enumerate(sorted_keywords):
            if idx not in used_indices:
                merged_keywords.append(kw)
        return merged_keywords

    def match_package_with_type(self, page_num, all_results):
        """
        检查页面类型关键字并匹配package，将匹配结果存储在列表C中
        """
        # 定义类型关键字列表
        type_keywords = ["BGA", "DFN", "SON", "QFP", "QFN", "SOP", "SOIC", "PLASTIC BALL GRID ARRAY",
                         "Plastic Quad Flat Package","Quad Flatpack"]
        yolo_keywords = ["BGA", "DFN_SON", "QFP", "QFN", "SOP"]
        # 第一优先级：可编辑页面的关键字（type为'text'）
        editable_keywords = [item for item in all_results
                             if item['page'] == page_num
                             and item['keyword'] in type_keywords
                             and item.get('type') == 'text']

        if editable_keywords:
            # 如果找到可编辑页面的关键字，只使用这些
            type_keywords_list = editable_keywords
        else:
            # 只有当可编辑页面没有找到关键字时，才使用OCR识别的关键字
            ocr_keywords = [item for item in all_results
                            if item['page'] == page_num
                            and item['keyword'] in type_keywords
                            and item.get('type') == 'ocr']
            type_keywords_list = ocr_keywords

        # 获取当前页面的所有package（列表B）
        package_list = [item for item in self.data2 if item['page'] == page_num]
        # 如果存在类型关键字
        if type_keywords_list:
            for package in package_list:
                distances = []
                for type_kw in type_keywords_list:
                    # 计算package与类型关键字的距离
                    dist = get_rects(package['new_rect'], type_kw['coordinates'])
                    distances.append((dist, type_kw))

                distances.sort(key=lambda x: x[0])
                if distances:
                    closest_type = distances[0][1]
                    type_keywords_list.remove(closest_type)

                    if 'package_type' in package:
                        # 同时更新part_content中的关键特征视图名称
                        package['package_type'] = closest_type['keyword']
                        if package.get('part_content'):
                            for part in package['part_content']:
                                if part['part_name'] in yolo_keywords:
                                    part['part_name'] = package['package_type']
                        # package['package_type'] = closest_type['keyword']
                        print(f"Package 匹配到类型: {closest_type['keyword']}")
                    # 新增：获取content，调用test1.py的check_keywords_and_numbers
                    content = closest_type.get('content', '')
                    info = clean_result(check_keywords_and_numbers(content))
                    result_dict = {
                        'pdf': os.path.basename(self.pdf_path),
                        'page_num': page_num,
                        'content': content,
                        'package_type': closest_type.get('keyword', ''),
                        'pin': info.get('pin') if info.get('pin') is not None else None,
                        'length': info.get('length') if info.get('length') is not None else None,
                        'width': info.get('width') if info.get('width') is not None else None,
                        'height': info.get('height') if info.get('height') is not None else None,
                        'horizontal_pin': info.get('horizontal_pin') if info.get(
                            'horizontal_pin') is not None else None,
                        'vertical_pin': info.get('vertical_pin') if info.get('vertical_pin') is not None else None,
                    }

                    # 写入json文档，标准JSON数组格式
                    output_path = 'output.json'
                    # 先读取已有内容
                    if os.path.exists(output_path):
                        with open(output_path, 'r', encoding='utf-8') as f:
                            try:
                                data = json.load(f)
                            except json.JSONDecodeError:
                                data = []
                    else:
                        data = []


                    data.append(result_dict)
                    # 再写回
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)


    def match_package_with_view(self, package, all_results):
        """
        检查package中的视图与视图关键字匹配，将匹配结果存储在列表C中
        """
        # 定义视图关键字列表
        view_keywords = ["TOPVIEW", "SIDEVIEW", "DETAIL"]
        page_num = package['page']
        # 列表A：存储含有视图关键字的数据
        list_a = [item for item in all_results
                  if item['page'] == page_num
                  and any(keyword in item['keyword'] for keyword in view_keywords)]
        # 新增：筛选y1在package的new_rect的y0与y1之间的关键字
        pkg_y0 = package['new_rect'][1]
        pkg_y1 = package['new_rect'][3]
        h = pkg_y1 - pkg_y0
        list_a = [item for item in list_a if pkg_y0 - h * 0.1 <= item['coordinates'][3] <= pkg_y1 + h * 0.1]
        list_a.sort(key=lambda x: x['coordinates'][1])

        package_views = package.get('part_content', [])
        # 列表B：存储part_content中name为top、side、detail的部分
        list_b = [part for part in package_views if
                  part['part_name'].lower() in ['top', 'side', 'detail', 'qfp', 'sop']]
        list_b.sort(key=lambda x: x['new_rect'][1])
        if list_a:
            for part in list_b:
                ref_rect = part['new_rect']
                x0 = ref_rect[0]
                x1 = ref_rect[2]
                y0 = ref_rect[1]
                y1 = ref_rect[3]
                view_center_x = (x0 + x1) / 2
                view_center_y = (y0 + y1) / 2

                filtered_parts = [
                    part for part in list_b
                    if x0 <= ((part['new_rect'][0] + part['new_rect'][2]) / 2) <= x1
                       and ((part['new_rect'][1] + part['new_rect'][3]) / 2) > y1
                ]
                filtered_parts.sort(
                    key=lambda part: (part['new_rect'][1] + part['new_rect'][3]) / 2
                )
                filtered_views = []
                if filtered_parts:
                    first_fp_y0 = filtered_parts[0]['rect'][1]
                    for view in list_a:
                        coords = view['coordinates']
                        center_x = (coords[0] + coords[2]) / 2
                        center_y = (coords[1] + coords[3]) / 2
                        if x0 <= center_x <= x1 and center_y < first_fp_y0:
                            filtered_views.append(view)
                else:
                    for view in list_a:
                        coords = view['coordinates']
                        center_x = (coords[0] + coords[2]) / 2
                        if x0 <= center_x <= x1:
                            filtered_views.append(view)
                # 存储上方和下方的视图关键字
                top_keywords = []
                bottom_keywords = []

                for view in filtered_views:
                    # 计算视图关键字的中心点
                    coords = view['coordinates']
                    kw_center_x = (coords[0] + coords[2]) / 2
                    kw_center_y = (coords[1] + coords[3]) / 2

                    # 计算两个中心点之间的距离
                    distance = ((view_center_x - kw_center_x) ** 2 + (view_center_y - kw_center_y) ** 2) ** 0.5
                    # 计算相对于y轴的角度（以视图中心点为原点）
                    dx = kw_center_x - view_center_x  # x方向差值
                    dy = kw_center_y - view_center_y  # y方向差值
                    angle = math.atan2(dy, dx)
                    angle = math.degrees(angle)
                    if angle < 0:
                        angle += 360  # 将负角度转换为0-360度范围

                    # 根据角度范围分类
                    if 45 <= angle <= 135 and x0 <= kw_center_x <= x1:  # 下方区域
                        bottom_keywords.append((distance, view, angle))
                    elif 225 <= angle <= 315 and x0 <= kw_center_x <= x1:  # 上方区域
                        top_keywords.append((distance, view, angle))

                # 优先选择上方的视图关键字
                if top_keywords:
                    # 按距离排序
                    top_keywords.sort(key=lambda x: x[0])
                    closest_view = top_keywords[0][1]
                elif bottom_keywords:
                    # 按距离排序
                    bottom_keywords.sort(key=lambda x: x[0])
                    closest_view = bottom_keywords[0][1]
                else:
                    continue
                print(f"part_name:{part['part_name']}匹配到{closest_view['keyword']}")
                # 不直接更新part_name，增加yolo_part_name字段
                part['part_name'] = closest_view['keyword']
                list_a.remove(closest_view)
    def generate_detect_pdf(self):
        """根据新转换后的数据生成pdf"""
        with fitz.open(self.pdf_path) as doc:
            for i in range(len(self.data2)):
                page = doc[self.data2[i]['page']]   # 封装图所在页
                rect = self.data2[i]['new_rect']
                # 绘制框选区域
                p1 = (rect[0], rect[1])
                p2 = (rect[0], rect[3])
                p3 = (rect[2], rect[1])
                p4 = (rect[2], rect[3])
                page.draw_line(p1, p2, PACKAGE_COLOR, width=2)
                page.draw_line(p1, p3, PACKAGE_COLOR, width=2)
                page.draw_line(p3, p4, PACKAGE_COLOR, width=2)
                page.draw_line(p2, p4, PACKAGE_COLOR, width=2)
                # 框标签
                text = 'package'
                page.insert_text(fitz.Point(rect[0], rect[1]), text, fontsize=12, color=PACKAGE_COLOR)
                if (self.data2[i]['part_content']):  # 绘制部分视图
                    part_content = self.data2[i]['part_content']
                    for j in range(len(part_content)):
                        rect = part_content[j]['new_rect']
                        page = doc[part_content[j]['page']]
                        if (part_content[j]['part_name'] == 'Note'):
                            color = NOTE_COLOR
                            text = 'Note'
                        elif (part_content[j]['part_name'] == 'Top'):
                            color = TOP_COLOR
                            text = 'Top'
                        elif (part_content[j]['part_name'] == 'Side'):
                            color = SIDE_COLOR
                            text = 'Side'
                        elif (part_content[j]['part_name'] == 'Detail'):
                            color = DETAIL_COLOR
                            text = 'Detail'
                        elif (part_content[j]['part_name'] == 'Form'):
                            color = FORM_COLOR
                            text = 'Form'
                        else:
                            color = KEYVIEW_COLOR
                            text = part_content[j]['part_name']
                        # 绘制框选区域
                        p1 = (rect[0], rect[1])
                        p2 = (rect[0], rect[3])
                        p3 = (rect[2], rect[1])
                        p4 = (rect[2], rect[3])
                        page.draw_line(p1, p2, color=color, width=2)
                        page.draw_line(p1, p3, color=color, width=2)
                        page.draw_line(p3, p4, color=color, width=2)
                        page.draw_line(p2, p4, color=color, width=2)
                        # 框标签
                        page.insert_text(fitz.Point(rect[0], rect[1]), text, fontsize=12, color=color)
                doc.save(PDF_NAME, garbage=1)

    def clean_file(self):
        # remove_dir(IMAGE_PATH)
        # remove_dir(SAVE_IMG_PATH)
        pass


