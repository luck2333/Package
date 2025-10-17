import os
import sys
import cv2
import math
import copy
import onnxruntime
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from wand.exceptions import ImageError
from PIL import Image, ImageDraw,ImageEnhance
import warnings
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
onnx_file_path = "model/ocr_model/onnx_orientation/resnet_orientation.onnx"
session = onnxruntime.InferenceSession(onnx_file_path)

matplotlib.use('TkAgg')

#图像等级增强：
def adjust_contrast(image, factor):
    '调整图片对比度'
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def sharpen_image(image):
    '图片锐化'
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(2.0)

def var_calculate(data):
    if (data > 4658.1001): return 1
    elif (data > 4230.0475): return 2
    else: return 3

def ent_calculate(data):
    if (data < 6.2285): return 1
    elif (data < 7.3436): return 2
    else: return 3

def rank_calculate(var_value, ent_value):
    f1 = var_calculate(var_value)
    f2 = ent_calculate(ent_value)
    if (f1 + f2 == 4 or f1 + f2 == 5): return 'L2'
    elif (f1 == f2): return 'L'+ str(f1)
    elif (f1 + f2 == 6): return 'L3'
    elif (f1 + f2 == 3): return 'L1'

def get_variance(image_path):
    img = cv2.imread(image_path,0)
    height, width = img.shape
    size = img.size
    p = [0]*256
    for i in range(height):
        for j in range(width):
            p[img[i][j]] += 1
    m = 0
    for i in range(256):
        p[i] /= size
        m += i*p[i]
    var_value = 0
    for i in range(256):
        var_value += (i-m)*(i-m)*p[i]
    return var_value
    # xlsx_operation1(image_filename,var_value)


def calculate_2d_entropy(image_path, window_size=3):
    # 打开图像并转换为灰度图像
    img = Image.open(image_path).convert("L")
    # 将图像转换为数组
    img_array = np.array(img)
    # 获取图像的高度和宽度
    height, width = img_array.shape
    # 初始化联合概率矩阵
    joint_prob_matrix = np.zeros((256, 256))
    # 计算联合概率分布
    for y in range(height - window_size + 1):
        for x in range(width - window_size + 1):
            window = img_array[y:y + window_size, x:x + window_size]
            pixel_values = window.flatten()
            for i in range(len(pixel_values) - 1):
                for j in range(i + 1, len(pixel_values)):
                    joint_prob_matrix[pixel_values[i], pixel_values[j]] += 1
    # 归一化联合概率矩阵
    num_windows = (height - window_size + 1) * (width - window_size + 1)
    num_pairs_per_window = window_size ** 2 * (window_size ** 2 - 1) // 2
    joint_prob_matrix /= (num_windows * num_pairs_per_window)
    # 计算二维信息熵
    entropy_2d = -np.sum(joint_prob_matrix * np.log2(joint_prob_matrix + 1e-10))  # 加上小常数以避免log(0)
    return entropy_2d
    # xlsx_operation2(image_name,entropy_2d)



# PalldeOCR 检测模块 需要用到的图片预处理类
class NormalizeImage(object):
    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (
                                img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        self.limit_side_len = kwargs['limit_side_len']
        self.limit_type = kwargs.get('limit_type', 'min')

    def __call__(self, data):
        img = data['image']
        src_h, src_w,_= img.shape
        img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type0(self, img):

        limit_side_len = self.limit_side_len
        h, w,_= img.shape

        # limit the max side
        if max(h, w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        else:
            ratio = 1.
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = int(round(resize_h / 32) * 32)
        resize_w = int(round(resize_w / 32) * 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        # return img, np.array([h, w])
        return img, [ratio_h, ratio_w]


### 检测结果后处理过程（得到检测框）
class DBPostProcess(object):

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                   src_w, src_h)

            boxes_batch.append({'points': boxes})
        return boxes_batch


## 根据推理结果解码识别结果
class process_pred(object):
    def __init__(self, character_dict_path=None, character_type='ch', use_space_char=False):
        self.character_str = ''
        with open(character_dict_path, 'rb') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip('\n').strip('\r\n')
                self.character_str += line
        if use_space_char:
            self.character_str += ' '
        dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        result_list = []
        ignored_tokens = [0]
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                # char_list.append(self.character[int(text_index[batch_idx][idx])])
                try:
                    char_list.append(self.character[int(text_index[batch_idx][idx])])
                except IndexError:
                    char_list.append(self.character[int(text_index[batch_idx][idx]) - 1])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def __call__(self, preds, label=None):
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label


class det_rec_functions(object):
    def __init__(self, image, use_large=False):
        self.img = image.copy()
        self.det_file = 'model/ocr_model/onnx_det/0529det_model.onnx'
        self.small_rec_file = 'model/ocr_model/onnx_rec/package_rec_model.onnx'
        self.large_rec_file = 'model/ocr_model/onnx_rec/package_rec_model.onnx'
        self.onet_det_session = onnxruntime.InferenceSession(self.det_file)
        if use_large:
            self.onet_rec_session = onnxruntime.InferenceSession(self.large_rec_file)
        else:
            self.onet_rec_session = onnxruntime.InferenceSession(self.small_rec_file)
        self.infer_before_process_op, self.det_re_process_op = self.get_process()
        self.postprocess_op = process_pred('model/ocr_model/rec_dict.txt', 'en', True)

    def preprocess_image_cv2(self, img):
        # 1. 读取图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        # 1. 调整图像大小
        img_resized = cv2.resize(img, (64, 64))

        # 2. 归一化到 [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0

        # 3. 转换为 (C, H, W)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))

        # 4. 标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_standardized = (img_transposed - mean) / std

        return img_standardized
    ## 图片预处理过程
    def transform(self, data, ops=None):
        """ transform """
        if ops is None:
            ops = []
        for op in ops:
            data = op(data)
            if data is None:
                return None
        return data

    def create_operators(self, op_param_list, global_config=None):
        assert isinstance(op_param_list, list), ('operator config should be a list')
        ops = []
        for operator in op_param_list:
            assert isinstance(operator,
                              dict) and len(operator) == 1, "yaml format error"
            op_name = list(operator)[0]
            param = {} if operator[op_name] is None else operator[op_name]
            if global_config is not None:
                param.update(global_config)
            op = eval(op_name)(**param)
            ops.append(op)
        return ops

    ### 检测框的后处理
    def order_points_clockwise(self, pts):
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    ### 定义图片前处理过程，和检测结果后处理过程
    def get_process(self):
        det_db_thresh = 0.3
        det_db_box_thresh = 0.5
        max_candidates = 2000
        unclip_ratio = 1.6
        use_dilation = True

        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 2500,
                'limit_type': 'max'
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]

        infer_before_process_op = self.create_operators(pre_process_list)
        det_re_process_op = DBPostProcess(det_db_thresh, det_db_box_thresh, max_candidates, unclip_ratio, use_dilation)
        return infer_before_process_op, det_re_process_op

    def sorted_boxes(self, dt_boxes):

        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                    (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes

    ### 图像输入预处理
    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = [int(v) for v in "3, 48, 100".split(",")]
        assert imgC == img.shape[2]
        imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    ## 推理检测图片中的部分
    def get_boxes(self, img_path, name, show, image1):
        img_ori = self.img
        img_part = img_ori.copy()
        data_part = {'image': img_part}
        data_part = self.transform(data_part, self.infer_before_process_op)
        img_part, shape_part_list = data_part
        img_part = np.expand_dims(img_part, axis=0)
        shape_part_list = np.expand_dims(shape_part_list, axis=0)
        inputs_part = {self.onet_det_session.get_inputs()[0].name: img_part}
        outs_part = self.onet_det_session.run(None, inputs_part)
        post_res_part = self.det_re_process_op(outs_part[0], shape_part_list)
        dt_boxes_part = post_res_part[0]['points']
        dt_boxes_part = self.filter_tag_det_res(dt_boxes_part, img_ori.shape)
        dt_boxes_part = self.sorted_boxes(dt_boxes_part)


        # 对检测框进行延申：
        det_boxs = []
        arr_List = np.array(dt_boxes_part)
        for det_box in arr_List:
            det_box[1:3, 0] += 2
            det_boxs.append(det_box)

        det_lists = []
        for det in det_boxs:
            a_list = []
            for d in det:
                a_list.append(list(d))
            det_lists.append(a_list)
        # 显示检测结果
        if show:
            a = image1
            image = Image.fromarray(a)
            # image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            for det_list in det_lists:
                x1 = det_list[0][0]
                y1 = det_list[0][1]
                x2 = det_list[2][0]
                y2 = det_list[2][1]
                try:
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255))
                except ValueError:
                    draw.rectangle([x1, y1, x2, y2 + 10], outline=(255, 0, 0))
            # image.show()
            if not os.path.exists('det_sign'):
                os.makedirs('det_sign')
            if os.path.exists(f'det_sign/{name}.png'):
                image.save(f'det_sign/{name}_1.png')
            else:
                image.save(f'det_sign/{name}.png')
        return det_boxs

    ### 根据bounding box得到单元格图片
    def get_rotate_crop_image(self, img, points):
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        return dst_img

    ### 单张图片推理
    def get_img_res(self, onnx_model, img, process_op):
        h, w = img.shape[:2]
        img = self.resize_norm_img(img, w * 1.0 / h)
        img = img[np.newaxis, :]
        inputs = {onnx_model.get_inputs()[0].name: img}
        outs = onnx_model.run(None, inputs)
        result = process_op(outs[0])
        return result

    def recognition_img(self, dt_boxes, name, image_path,Is_crop):
        img_ori = self.img
        img = img_ori.copy()
        image = Image.open(image_path)
        image = adjust_contrast(image, factor=1.5)
        enhance_img = sharpen_image(image)
        enhance_img = np.array(enhance_img)

        # 识别过程
        # 根据dt_box得到小图片
        img_list = []
        i = 0
        for box in dt_boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = self.get_rotate_crop_image(enhance_img, tmp_box)

            if Is_crop:
                crop = Image.fromarray(img_crop)
                if not os.path.exists(f'dataset_crop/{name}'):
                    os.makedirs(f'dataset_crop/{name}')
                crop.save(f'dataset_crop/{name}/{name}_{i}.jpg')
                i += 1
            img_list.append(img_crop)

        ## 识别小图片
        results = []
        results_info = []
        for pic in img_list:
            # 图像方向矫正
            # 使用 Pillow 的 fromarray 转换
            input_tensor = np.expand_dims(self.preprocess_image_cv2(pic), axis=0)
            # print(img.shape)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            outputs = session.run([output_name], {input_name: input_tensor})

            classes = ['0', '180', '270', '90']
            probabilities = outputs[0][0]  # 取出第一个样本的结果
            predicted_class = np.argmax(probabilities)  # 获取预测类别索引
            if classes[predicted_class.item()] == '90':
                if isinstance(pic, Image.Image):  # 如果是 PIL.Image
                    pic = np.array(pic)  # 转换为 NumPy 数组
                elif pic is None:  # 如果为空
                    raise ValueError("Image is None. Check the image path or loading process.")
                # 调用 OpenCV 转换
                pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
                pic = cv2.rotate(pic, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif classes[predicted_class.item()] == '180':
                if isinstance(pic, Image.Image):  # 如果是 PIL.Image
                    pic = np.array(pic)  # 转换为 NumPy 数组
                elif pic is None:  # 如果为空
                    raise ValueError("Image is None. Check the image path or loading process.")

                # 调用 OpenCV 转换
                pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
            elif classes[predicted_class.item()] == '270':
                if isinstance(pic, Image.Image):  # 如果是 PIL.Image
                    pic = np.array(pic)  # 转换为 NumPy 数组
                elif pic is None:  # 如果为空
                    raise ValueError("Image is None. Check the image path or loading process.")

                # 调用 OpenCV 转换
                pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
                pic = cv2.rotate(pic, cv2.ROTATE_90_CLOCKWISE)

            else:
                if isinstance(pic, Image.Image):  # 如果是 PIL.Image
                    pic = np.array(pic)
            # 图像方向矫正结束
            res = self.get_img_res(self.onet_rec_session, pic, self.postprocess_op)
            results.append(res[0])
            results_info.append(res)

        return results, results_info

    def recognition_singleimg(self, image):
        results = []
        results_info = []

        res = self.get_img_res(self.onet_rec_session, image, self.postprocess_op)
        results.append(res[0])
        results_info.append(res)

        return results, results_info

# def ONNX_Use(image, image_path, name):
#     ocr_sys = det_rec_functions(image)
#     dt_boxes = ocr_sys.get_boxes(image_path, name, show=False, image1=image)
#     results, results_info = ocr_sys.recognition_img(dt_boxes, name, image_path,Is_crop=False)
#     # results, results_info = ocr_sys.recognition_singleimg(image)
#
#     text_list = []
#     for result in results_info:
#         text_list.append(result[0][0])
#     print(text_list)
def ONNX_Use(image, image_path,name):
    ocr_sys = det_rec_functions(image)
    dt_boxes = ocr_sys.get_boxes(image_path, name, show=False, image1=image)
    #results, results_info = ocr_sys.recognition_img(dt_boxes,name,Is_crop=True)
    # text_list = []
    # for result in results_info:
    #     text_list.append(result[0][0])
    rows_to_extract = [0, 2]
    value_box = [list(np.concatenate([arr[row] for row in rows_to_extract])) for arr in dt_boxes]
    return value_box
def ONNX_ocr(image, image_path,name,boxes):
    ocr_sys = det_rec_functions(image)
    #dt_boxes = ocr_sys.get_boxes(image_path, name, show=False, image1=image)
    results, results_info = ocr_sys.recognition_img(boxes, name, image_path,Is_crop=False)
    # results, results_info = ocr_sys.recognition_singleimg(image)

    text_list = []
    for result in results_info:
        text_list.append(result[0][0])
    print(text_list)
    return text_list
def Run_onnx(image_path, name):
    image = cv2.imread(image_path)
    # ONNX运行代码
    ONNX_Use(image, image_path, name)
def Run_onnx_ocr(image_path,boxes,name):
    image = cv2.imread(image_path)
    text_list = ONNX_ocr(image, image_path, name,boxes)
    return text_list


if __name__ == '__main__':
    # OCR整个文件夹下的所有图片，注意sort排序只能排序图片名称为数字的图片，如果文件夹中图片命名全英文，建议注释sort排序代码
    # 如果想检查检测结果，可将ONNX_Use函数中ocr_sys.get_boxes所含变量show改为True，之后检测结果自动保存在新建文件夹det_sign中
    # DataSet_path = 'qxy'
    # image_names = os.listdir(DataSet_path)
    # image_names.sort(key=lambda x: int(x.split('.')[0]))
    # # image_names.sort(key=lambda x: x.split('.')[0])
    # for name in image_names:
    #     image_path = os.path.join(DataSet_path, name)
    #     name1 = name.split('.')[0]
    #     Run_onnx(image_path, name1)


    # #单张图片检测
    image_path = 'SON_test/bottom1.jpg'
    name = image_path.split('/')[1]
    name = name.split('.')[0]

    Run_onnx(image_path,name)

