import os
import sys
import cv2
import time
import math
import copy
import re
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
import warnings
import onnxruntime as ort
import onnxruntime
from pathlib import Path

try:
    from packagefiles.model_paths import ocr_model_path
except ModuleNotFoundError:  # pragma: no cover - 兼容脚本直接运行
    def ocr_model_path(*parts):
        return str(Path(__file__).resolve().parents[2] / 'model' / 'ocr_model' / Path(*parts))
from packagefiles.UI.utils.upline import uplineCoordinate,isExistUpline
OCR_ONNX = r"packagefiles/TableProcessing/ocr_onnx/"




def calculate_overlap_percentage(box1, box2):
    """
    计算两个矩形框的重叠百分比。

    :param box1: 第一个矩形框，格式为 (x1, y1, x2, y2)
    :param box2: 第二个矩形框，格式为 (x1, y1, x2, y2)
    :return: 重合区域占第一个矩形框面积的百分比
    """
    # 计算重叠区域的坐标
    x_left = max(box1[0][0], box2[0][0])
    y_top = max(box1[0][1], box2[0][1])
    x_right = min(box1[1][0], box2[1][0])
    y_bottom = min(box1[1][1], box2[1][1])

    # 如果没有重叠区域
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    # 重叠区域面积
    overlap_area = (x_right - x_left) * (y_bottom - y_top)

    # 第一个矩形框的面积
    box1_area = (box1[1][1] - box1[0][1]) * (box1[1][0] - box1[0][0])

    # 计算重叠百分比
    overlap_percentage = (overlap_area / box1_area) * 100
    return overlap_percentage

warnings.filterwarnings("ignore")




# PalldeOCR 检测模块 需要用到的图片预处理类
class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

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
        src_h, src_w, _ = img.shape
        img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, _ = img.shape

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
    """
    The post process for Differentiable Binarization (DB).
    """

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
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

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
                    # print(text_index[batch_idx][idx])
                    # print(len(self.character))
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
        self.det_file = ocr_model_path('onnx_det', '0722det.onnx')
        self.small_rec_file = ocr_model_path('onnx_rec', 'package_rec_model.onnx')
        self.large_rec_file = ocr_model_path('onnx_rec', 'package_rec_model.onnx')
        self.onet_det_session = onnxruntime.InferenceSession(self.det_file)
        if use_large:
            self.onet_rec_session = onnxruntime.InferenceSession(self.large_rec_file)
        else:
            self.onet_rec_session = onnxruntime.InferenceSession(self.small_rec_file)
        self.infer_before_process_op, self.det_re_process_op = self.get_process()
        # self.postprocess_op = process_pred('en_dict.txt', 'en', True)
        self.postprocess_op = process_pred(ocr_model_path('rec_dict.txt'), 'en', True)
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
        """
        create operators based on the config

        Args:
            params(list): a dict list, used to create some operators
        """
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
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
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
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
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
    def get_boxes(self, name, show, image1):
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
            # print(det_box)
        # 显示检测结果
        if show:
            det_lists = []
            for det in det_boxs:
                a_list = []
                for d in det:
                    a_list.append(list(d))
                det_lists.append(a_list)
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
                    if y1 > y2 + 10:
                        y1, y2 = y2 + 10, y2
                    draw.rectangle([x1, y1, x2, y2+10], outline=(255, 0, 0))
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
        dst_img_height, dst_img_width = dst_img.shape[0:2]
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

    def preprocess_image_cv2(self,img):
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

    def recognition_img(self, dt_boxes, name, Is_crop):
        img_ori = self.img
        img = img_ori.copy()
        # img = self.crop_noisy(img)
        ### 识别过程
        ## 根据bndbox得到小图片
        img_list = []
        i = 0
        for box in dt_boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = self.get_rotate_crop_image(img, tmp_box)
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
            #---------------------------------------------------------------------------------------
            # cv2.imshow("tt",pic)
            # cv2.waitKey()
            # rgb_image = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式




            #-------------------------------------------------------------------------------------
            res = self.get_img_res(self.onet_rec_session, pic, self.postprocess_op)
            # print(res)
            res[0] = list(res[0])
            #存在上划线
            # if '!' in res[0][0] and isExistUpline(pic) and uplineCoordinate(pic):



            if isExistUpline(pic) and uplineCoordinate(pic):
                # cv2.imshow("upline",pic)
                # cv2.waitKey()

                # print(upline_coordinate)
                upline_coordinate, cropheight = uplineCoordinate(pic)

                if cropheight < 0.5*pic.shape[0]:
                    if len(upline_coordinate) == 1:

                        # print(cropheight)
                        # print(pic.shape)
                        if upline_coordinate[0][0]-3 > 0:
                            upline_coordinate_left = upline_coordinate[0][0]-3
                        else:
                            upline_coordinate_left = 0
                        if upline_coordinate[0][2]+3 < pic.shape[1]:
                            upline_coordinate_right = upline_coordinate[0][2] + 3
                        else:
                            upline_coordinate_right = pic.shape[1]

                        upline_crop = pic[cropheight:, upline_coordinate_left:upline_coordinate_right]
                        # cv2.imshow("upline_crop",upline_crop)
                        # cv2.waitKey()
                        if upline_crop.shape[0]/upline_crop.shape[1] > 6:
                            res_upline = None
                            # upline_crop = pic[:cropheight, upline_coordinate_left:upline_coordinate_right]

                            # res_upline = self.get_img_res(self.onet_rec_session, upline_crop, self.postprocess_op)
                        else:
                            res_upline = self.get_img_res(self.onet_rec_session, upline_crop, self.postprocess_op)
                        crop = pic[cropheight:, :]
                        res_ori_upline = self.get_img_res(self.onet_rec_session, crop, self.postprocess_op)

                        if res_upline != None and res_upline[0][0] == res_ori_upline[0][0]:
                            res[0][0] = ''.join(['!' + char for char in res_upline[0][0]])
                            res[0][0] = re.sub(r'!+', '!', res[0][0])
                        # else:
                        elif res_upline != None:
                            extend_crop_left = pic[cropheight:, :upline_coordinate[0][0]]
                            extend_crop_right = pic[cropheight:, upline_coordinate[0][2]:]
                            if extend_crop_left.shape[1]== 0 or extend_crop_left.shape[0]/extend_crop_left.shape[1] > 6:
                                res_left = ""
                            else:
                                res_left = self.get_img_res(self.onet_rec_session, extend_crop_left, self.postprocess_op)
                                res_left = res_left[0][0]
                            if extend_crop_right.shape[1] == 0 or extend_crop_right.shape[0]/extend_crop_right.shape[1] > 6:
                                res_right = ""
                            else:
                                res_right = self.get_img_res(self.onet_rec_session, extend_crop_right, self.postprocess_op)
                                res_right = res_right[0][0]
                            res[0][0] = ''.join(['!' + char for char in res_upline[0][0]])
                            # print(res[0][0])
                            # print(res_left)
                            # print(res_right)
                            res[0][0] = res_left + res[0][0] + res_right
                            res[0][0] = re.sub(r'!+', '!', res[0][0])
                        # print("upline:",res[0][0])

            results.append(res[0])
            # print(res)
            results_info.append(res)
        return results, results_info

def list_to_tuple(data):
    if isinstance(data, list):  # 如果当前数据是列表
        return tuple(list_to_tuple(item) for item in data)  # 递归转换
    else:
        return data  # 如果不是列表，直接返回


def ONNX_Use(image, name):
    ocr_sys = det_rec_functions(image)
    dt_boxes = ocr_sys.get_boxes(name, show=False, image1=image)
    b = []
    for det_list in dt_boxes:
        a = []
        for dets in det_list:

            arr_to_list = list(dets)
            a.append([int(arr_to_list[0]),int(arr_to_list[1])])
        b.append(a)
    results, results_info = ocr_sys.recognition_img(dt_boxes,name,Is_crop=False)
    text_list = []
    for result in results_info:
        text_list.append(result[0][0])

    # 保存文本和文本坐标入字典：
    b = list_to_tuple(b)
    # print(image_path)
    DBboxes = []

    for item in b:
        DBboxes.append(item)

    # 保存文本和文本坐标入字典：
    text_save = dict(zip(b, text_list))
    write_txt = []
    # print(image_path)
    DBboxes = []
    DBtext = []
    for key, values in text_save.items():
        txt1 = {"transcription": '', "points": '', "difficult": 'false'}
        txt1["transcription"] = values
        DBtext.append(values)
        txt1["points"] = key
        DBboxes.append(key)
        write_txt.append(txt1)
    # print(write_txt)

    return write_txt

def resize_image(image):
    width,height,c = image.shape
    if width > 2000 and height > 2000:
        new_height = int(height / 2.5)
        new_width = int(width / 2.5)
        resized_image = cv2.resize(image, (new_height, new_width), interpolation=cv2.INTER_LINEAR)

        return resized_image
    else:
        return image

def Run_onnx1(image,name):
    write = ONNX_Use(image, name)
    texts = []
    boxes = []

    for item in write:
        texts.append(item['transcription'])
        box = [list(point) for point in item['points']]
        boxes.append(box)
    return boxes, texts

if __name__ == '__main__':
    dir_path = 'output'
    img_name = 'page_1_ocr.png'
    img_path = os.path.join(dir_path, img_name)


    # DataSet_path = 'img'
    # image_names = os.listdir(DataSet_path)
    # png_files = [f for f in os.listdir(DataSet_path) if f.endswith(".png")]
    # png_files.sort(key=lambda x: int(x.split('.')[0]))
    #
    #
    #
    # for name in png_files:
    #     image_path = os.path.join(DataSet_path,name)
    #     name1 = name.split('.')[0]
    #     write = Run_onnx(image_path,name1)
    #     print(write)
    #     text_to_write = f'train/{name}\t{write}'
    #     with open('det_sign/det_sign.txt', 'a') as f:
    #         f.write(text_to_write + '\n')


    # # # #单张图片进行测试
    # from PIL import Image, ImageDraw, ImageFont
    # dir_path = 'testImg'
    # img_name = 'img_68.png'
    # img_path = os.path.join(dir_path,img_name)
    # img = cv2.imread(img_path)
    # result = Run_onnx(img_path,img_name)
    # print(result)
    # # print(result[0]['transcription'])
    # # print(img.shape)
    # w,h = img.shape[1],img.shape[0]
    # background_color = (255, 255, 255)  # 白色背景
    # canvas = Image.new("RGB", (w, h), background_color)
    # # 创建绘图对象
    # draw = ImageDraw.Draw(canvas)
    # # 设置字体（可选）
    # try:
    #     font = ImageFont.truetype("arial.ttf", 18)  # 使用系统字体
    # except IOError:
    #     font = ImageFont.load_default()  # 如果字体不可用，使用默认字体
    #
    #
    # text_list = []
    # for item in result:
    #     # print(item['points'][0])
    #     # print(item['points'][2])
    #     text_list.append({'text':item['transcription'],'coordinates':item['points'][0]})
    #
    # # 在画布上绘制文本
    # flag = 0
    # for item in text_list:
    #     text = item["text"]
    #     coordinates = item["coordinates"]
    #     # print(coordinates)
    #     x,y = int(coordinates[0]),int(coordinates[1])
    #     # if flag == 0:
    #     #     y = y+8
    #     #     flag = 1
    #     # else:
    #     #     y = y
    #     #     flag = 0
    #     draw.text((x,y), text, fill=(0, 0, 0), font=font)  # 黑色文本
    #
    # # 显示画布
    # canvas.show()  # 显示画布

    # from PIL import Image, ImageDraw, ImageFont
    # import cv2
    # import os
    #
    # # 读取图片
    # dir_path = 'testImg'
    # img_name = 'img_67.png'
    # img_path = os.path.join(dir_path, img_name)
    # img = cv2.imread(img_path)
    # result = Run_onnx(img_path, img_name)
    #
    # # 创建画布
    # w, h = img.shape[1], img.shape[0]
    # canvas = Image.new("RGB", (w, h), (255, 255, 255))  # 白色背景
    # draw = ImageDraw.Draw(canvas)
    #
    # # 关键修改：使用支持中文的字体
    # try:
    #     # 方法1：指定系统中文字体（如微软雅黑）
    #     font = ImageFont.truetype("msyh.ttc", 18)  # Windows 常见中文字体
    #     # 方法2：使用绝对路径（如果字体不在系统目录）
    #     # font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 18)  # 黑体
    # except:
    #     print("警告：未找到中文字体，使用默认字体（可能无法显示中文）")
    #     font = ImageFont.load_default()
    #
    # # 绘制文本
    # for item in result:
    #     text = item['transcription']
    #     x, y = int(item['points'][0][0]), int(item['points'][0][1])
    #     draw.text((x, y), text, fill=(0, 0, 0), font=font)  # 黑色文本
    #
    # # 保存或显示
    # canvas.show()  # 显示画布
    # # canvas.save("output.png")  # 可选：保存结果

