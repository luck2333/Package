import argparse
import os
import cv2
import numpy as np
import onnxruntime
from math import sqrt


def make_parser(img_path, weight):
    output_path = r'Result/Package_extract/onnx_output/'

    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=weight,
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default=img_path,
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=output_path,
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    return parser


# yolox_onnx 需要的一些函数(从yolox中提取)
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_img_info(img_path):
    image = cv2.imread(img_path)
    size = image.shape
    w = size[1]  # 宽度
    h = size[0]  # 高度
    return w, h


def get_rotate_crop_image(img, points):  # 图片分割，在ultil中的原有函数,from utils import get_rotate_crop_image
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
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
    # if dst_img_height * 1.0 / dst_img_width >= 1:
    #     dst_img = np.rot90(dst_img)
    return dst_img


def find_the_only_body(img_path):
    global location
    global YOLOX_body
    print(location)
    if len(location) == 0:
        YOLOX_body = np.array([[1, 1, 1, 1]])
    if len(location) == 1:
        YOLOX_body = np.zeros((1, 4))
        YOLOX_body[0] = location[0]
    if len(location) > 1:
        new_location = location.copy()
        while len(new_location) > 1:
            # 删除离图片边界过近的框、超过图片边界的框
            w, h = get_img_info(img_path)
            print("w, h", w, h)
            mark_location = np.zeros((len(location)))
            new_new_location = np.zeros((0, 4))
            for i in range(len(location)):
                if location[i][0] < 0 or location[i][1] < 0 or location[i][2] > w or location[i][3] > h:
                    mark_location[i] = 1
            for i in range(len(mark_location)):
                if mark_location[i] == 0:
                    new_new_location = np.r_[new_new_location, [location[i]]]

            # 如果删除了距离边界过近的框还是数量大于1，找到最大的框作为body

            mark_location = np.zeros((len(new_new_location)))
            print("new_new_location", new_new_location)
            if len(new_new_location) > 1:
                for i in range(len(new_new_location)):
                    mark_location[i] = sqrt((new_new_location[i][2] - new_new_location[i][0]) ** 2 + (
                            new_new_location[i][3] - new_new_location[i][1]) ** 2)
                max_no = np.argmax(mark_location)

                new_location = np.zeros((0, 4))

                new_location = np.r_[new_location, [new_new_location[max_no]]]
                new_new_location = new_location
            new_location = new_new_location
            print("new_location", new_location)
        location = new_location
    if len(location) == 0:
        YOLOX_body = np.array([[1, 1, 1, 1]])
    if len(location) == 1:
        YOLOX_body = location
        box = np.array([[YOLOX_body[0][0], YOLOX_body[0][1]], [YOLOX_body[0][2], YOLOX_body[0][1]],
                        [YOLOX_body[0][2], YOLOX_body[0][3]], [YOLOX_body[0][0], YOLOX_body[0][3]]], np.float32)
        with open(img_path, 'rb') as f:
            np_arr = np.frombuffer(f.read(), dtype=np.uint8)
            # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    #以彩图读取
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取
        box_img = get_rotate_crop_image(img, box)
        cv2.namedWindow('origin', 0)
        cv2.imshow('origin', box_img)  # 显示当前ocr的识别区域
        cv2.waitKey(0)
    return YOLOX_body


def onnx_inference(img_path, package_classes, weight):
    # VOC_CLASSES = ('Border', 'Pad', 'Pin', 'angle', 'multi_value_1', 'multi_value_2', 'multi_value_3', 'multi_value_4',
    #                'multi_value_angle', 'multi_value_thickness', 'multi_value_triangle', 'other', 'pairs_inSide_col',
    #                'pairs_inSide_row', 'pairs_inSide_thickness', 'pairs_outSide_col', 'pairs_outSide_row', 'plane',
    #                'serial_number')
    if package_classes == 'BGA':
        VOC_CLASSES = ('multi_value_1',
                       'multi_value_2',
                       'multi_value_3',
                       'multi_value_triangle',
                       'pairs_inside_col',
                       'pairs_inside_row',
                       'pairs_outside_col',
                       'pairs_outside_row',
                       'other',
                       'BGA_serial_number',
                       'BGA_serial_letter',
                       'BGA_Border',
                       'BGA_PIN',)
    else:
        VOC_CLASSES = ('multi_value_1',
                       'multi_value_2',
                       'multi_value_3',
                       'multi_value_4',
                       'multi_value_angle',
                       'multi_value_triangle',
                       'angle',
                       'pairs_inside_col',
                       'pairs_inside_row',
                       'pairs_outside_col',
                       'pairs_outside_row',
                       'other',
                       'serial_number',
                       'QFN_Border',
                       'QFN_multi_value',
                       'QFN_pad',
                       'QFN_pairs_arrow',
                       'QFN_pairs_inside_oblique',
                       'QFP_Border',
                       'QFP_Pad',
                       'QFP_Pin',
                       'SOP_Border',
                       'SOP_Pin',
                       'multi_value_thickness',
                       'plane',
                       'pairs_inSide_thickness',
                       'BGA_serial_number',
                       'BGA_serial_letter',
                       'BGA_Border',)

    args = make_parser(img_path, weight).parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    origin_img = cv2.imread(args.image_path)
    img, ratio = preprocess(origin_img, input_shape)

    session = onnxruntime.InferenceSession(args.model)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape)[0]

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
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=VOC_CLASSES)
    else:
        final_boxes = np.zeros((0, 4))
        final_scores = np.zeros(0)
        final_cls_inds = np.zeros(0)

    print("final_boxes", final_boxes)

    output_pairs_data_location(np.array(final_cls_inds), np.array(final_boxes),
                               package_classes)  # 将yolox检测的pairs和data进行匹配输入到txt文本中
    '''
    final_boxes:记录yolox检测的坐标位置np(, 4)[x1,y1,x2,y2]
    final_cls_inds:记录每个yolox检测的种类np(, )[1,2,3,]
    final_scores:记录yolox每个检测的分数np(, )[80.9,90.1,50.2,]
    '''

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)


def output_pairs_data_location(cls, bboxes, package_classes):
    #########################################输出识别的类别和对角线坐标#############################################################

    # print("cls",cls)#tensor([1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.])
    # print("bboxes",bboxes)#(x1,y1,x2,y2)左上角与右下角坐标，yolox坐标原点是左上角
    # tensor([[ 781.2277,  311.5244,  820.7728,  350.0395],
    # [1039.5348,  101.0938, 1071.0819,  149.9549],
    # [ 532.6776,   97.2061,  604.5829,  143.0282],
    # [ 764.6501,  447.9761,  804.3810,  484.4607],
    # [ 317.2012,  520.6475,  372.4589,  589.8723],
    # [ 721.6006,  253.5669,  795.9207,  293.0987],
    # [ 754.2359,  209.0722,  765.7953,  417.7099],
    # [ 432.3812,  123.5709,  689.7897,  133.8431],
    # [1065.0292,  153.1357, 1110.1744,  163.3379],
    # [ 831.5218,  337.4080,  841.0558,  387.1069],
    # [ 389.1720,  567.0306,  400.0041,  606.2711],
    # [ 814.5815,  403.3475,  823.8375,  460.4219]])

    other_num = 0
    numbers_num = 0
    serial_num_num = 0
    pin_num = 0
    pad_num = 0
    border_num = 0
    pairs_num = 0
    angle_pairs_num = 0
    BGA_serial_letter_num = 0
    BGA_serial_num_num = 0
    '''
    0  'Border', 
    1  'Pad', 
    2  'Pin', 
    3  'angle', 
    4  'multi_value_1', 
    5  'multi_value_2', 
    6  'multi_value_3', 
    7  'multi_value_4', 
    8  'multi_value_angle', 
    9  'multi_value_thickness', 
    10 'multi_value_triangle', 
    11 'other', 
    12 'pairs_inSide_col', 
    13 'pairs_inSide_row', 
    14 'pairs_inSide_thickness', 
    15 'pairs_outSide_col', 
    16 'pairs_outSide_row', 
    17 'plane', 
    18 'serial_number'
    '''
    '''
    0'multi_value_1',
    1'multi_value_2',
    2'multi_value_3',
    3'multi_value_4',
    4'multi_value_angle',
    5'multi_value_triangle',
    6'angle',
    7'pairs_inside_col',
    8'pairs_inside_row',
    9'pairs_outside_col',
    10'pairs_outside_row',
    11'other',
    12'serial_number'
    13'QFN_Border',
    14'QFN_multi_value',
    15'QFN_pad',
    16'QFN_pairs_arrow',
    17'QFN_pairs_inside_oblique',
    18'QFP_Border',
    19'QFP_Pad',
    20'QFP_Pin',
    21'SOP_Border',
    22'SOP_Pin',
    23'multi_value_thickness',
    24'plane',
    25'pairs_inSide_thickness',
    计划添加
    26'BGA_serial_number'
    27'BGA_serial_letter'
    28'BGA_Border'
    
    '''
    '''
    'multi_value_1', 
   'multi_value_2', 
   'multi_value_3', 
   'multi_value_triangle', 
   'pairs_inside_col',
   'pairs_inside_row', 
   'pairs_outside_col', 
   'pairs_outside_row', 
   'other', 
   'BGA_serial_number',
   'BGA_serial_letter', 
   'BGA_Border', 
   'BGA_PIN',
    '''
    # 获取两种类型的数量
    for i in range(len(cls)):
        if package_classes == 'BGA':
            if cls[i] == 8:
                other_num += 1
            if cls[i] == 0 or cls[i] == 1 or cls[i] == 2 or cls[i] == 3:
                numbers_num += 1
            if cls[i] == 9:
                BGA_serial_num_num += 1
            if cls[i] == 10:
                BGA_serial_letter_num += 1
            if cls[i] == 4 or cls[i] == 5 or cls[i] == 6 or cls[i] == 7:
                pairs_num += 1
            if cls[i] == 11:
                border_num += 1
            if cls[i] == 12:
                pin_num += 1

        else:
            if cls[i] == 11:
                other_num += 1
            if cls[i] == 0 or cls[i] == 1 or cls[i] == 2 or cls[i] == 3 or cls[i] == 4 or cls[i] == 5 or cls[i] == 23:
                numbers_num += 1
            if cls[i] == 12:
                serial_num_num += 1
            if cls[i] == 7 or cls[i] == 8 or cls[i] == 9 or cls[i] == 10:
                pairs_num += 1
            if cls[i] == 6:
                angle_pairs_num += 1

            if package_classes == 'QFP':
                if cls[i] == 20:
                    pin_num += 1
                if cls[i] == 19:
                    pad_num += 1
                if cls[i] == 18:
                    border_num += 1
            if package_classes == 'QFN':
                if cls[i] == 15:
                    pad_num += 1

            if package_classes == 'SOP':
                if cls[i] == 22:
                    pin_num += 1
                if cls[i] == 21:
                    border_num += 1
            if package_classes == 'BGA':
                if cls[i] == 26:
                    BGA_serial_num_num += 1
                if cls[i] == 27:
                    BGA_serial_letter_num += 1
                if cls[i] == 28:
                    border_num += 1
                if cls[i] == 29:
                    pin_num += 1


    yolox_num = np.empty((numbers_num, 4))  # [x1,x2,x3,x4]
    yolox_other = np.empty((other_num, 4))  # [x1,x2,x3,x4]
    yolox_serial_num = np.empty((serial_num_num, 4))

    yolox_pin = np.empty((pin_num, 4))
    yolox_pad = np.empty((pad_num, 4))
    yolox_border = np.empty((border_num, 4))
    yolox_pairs = np.empty((pairs_num, 5))
    yolox_angle_pairs = np.empty((angle_pairs_num, 4))
    yolox_BGA_serial_num = np.empty((BGA_serial_num_num, 4))
    yolox_BGA_serial_letter = np.empty((BGA_serial_letter_num, 4))

    j = 0
    k = 0
    l = 0
    m = 0
    n = 0
    o = 0
    p = 0
    q = 0
    r = 0
    s = 0
    for i in range(len(cls)):
        if package_classes == 'BGA':
            if cls[i] == 8:
                yolox_other[k][0] = bboxes[i][0]
                yolox_other[k][1] = bboxes[i][1]
                yolox_other[k][2] = bboxes[i][2]
                yolox_other[k][3] = bboxes[i][3]
                k = k + 1
            if cls[i] == 0 or cls[i] == 1 or cls[i] == 2 or cls[i] == 3:
                yolox_num[j][0] = bboxes[i][0]
                yolox_num[j][1] = bboxes[i][1]
                yolox_num[j][2] = bboxes[i][2]
                yolox_num[j][3] = bboxes[i][3]
                j = j + 1
            if cls[i] == 9:
                yolox_BGA_serial_num[s][0] = bboxes[i][0]
                yolox_BGA_serial_num[s][1] = bboxes[i][1]
                yolox_BGA_serial_num[s][2] = bboxes[i][2]
                yolox_BGA_serial_num[s][3] = bboxes[i][3]
                s = s + 1
            if cls[i] == 10:
                yolox_BGA_serial_letter[r][0] = bboxes[i][0]
                yolox_BGA_serial_letter[r][1] = bboxes[i][1]
                yolox_BGA_serial_letter[r][2] = bboxes[i][2]
                yolox_BGA_serial_letter[r][3] = bboxes[i][3]
                r = r + 1
            if cls[i] == 4 or cls[i] == 5 or cls[i] == 6 or cls[i] == 7:
                yolox_pairs[p][0] = bboxes[i][0]
                yolox_pairs[p][1] = bboxes[i][1]
                yolox_pairs[p][2] = bboxes[i][2]
                yolox_pairs[p][3] = bboxes[i][3]
                if cls[i] == 4 or cls[i] == 5:
                    yolox_pairs[p][4] = 1
                else:
                    yolox_pairs[p][4] = 0
                p = p + 1
            if cls[i] == 11:
                yolox_border[o][0] = bboxes[i][0]
                yolox_border[o][1] = bboxes[i][1]
                yolox_border[o][2] = bboxes[i][2]
                yolox_border[o][3] = bboxes[i][3]
                o = o + 1
            if cls[i] == 12:
                yolox_pin[m][0] = bboxes[i][0]
                yolox_pin[m][1] = bboxes[i][1]
                yolox_pin[m][2] = bboxes[i][2]
                yolox_pin[m][3] = bboxes[i][3]
                m = m + 1

        else:

            if cls[i] == 0 or cls[i] == 1 or cls[i] == 2 or cls[i] == 3 or cls[i] == 4 or cls[i] == 5 or cls[i] == 23:
                yolox_num[j][0] = bboxes[i][0]
                yolox_num[j][1] = bboxes[i][1]
                yolox_num[j][2] = bboxes[i][2]
                yolox_num[j][3] = bboxes[i][3]
                j = j + 1
            if cls[i] == 11:
                yolox_other[k][0] = bboxes[i][0]
                yolox_other[k][1] = bboxes[i][1]
                yolox_other[k][2] = bboxes[i][2]
                yolox_other[k][3] = bboxes[i][3]
                k = k + 1
            if cls[i] == 12:
                yolox_serial_num[l][0] = bboxes[i][0]
                yolox_serial_num[l][1] = bboxes[i][1]
                yolox_serial_num[l][2] = bboxes[i][2]
                yolox_serial_num[l][3] = bboxes[i][3]
                l = l + 1

            if cls[i] == 7 or cls[i] == 8 or cls[i] == 9 or cls[i] == 10:
                yolox_pairs[p][0] = bboxes[i][0]
                yolox_pairs[p][1] = bboxes[i][1]
                yolox_pairs[p][2] = bboxes[i][2]
                yolox_pairs[p][3] = bboxes[i][3]
                if cls[i] == 9 or cls[i] == 10:
                    yolox_pairs[p][4] = 0
                else:
                    yolox_pairs[p][4] = 1
                p = p + 1
            if cls[i] == 6:
                yolox_angle_pairs[q][0] = bboxes[i][0]
                yolox_angle_pairs[q][1] = bboxes[i][1]
                yolox_angle_pairs[q][2] = bboxes[i][2]
                yolox_angle_pairs[q][3] = bboxes[i][3]
                q = q + 1
            if package_classes == 'QFP':
                if cls[i] == 20:
                    yolox_pin[m][0] = bboxes[i][0]
                    yolox_pin[m][1] = bboxes[i][1]
                    yolox_pin[m][2] = bboxes[i][2]
                    yolox_pin[m][3] = bboxes[i][3]
                    m = m + 1
                if cls[i] == 19:
                    yolox_pad[n][0] = bboxes[i][0]
                    yolox_pad[n][1] = bboxes[i][1]
                    yolox_pad[n][2] = bboxes[i][2]
                    yolox_pad[n][3] = bboxes[i][3]
                    n = n + 1
                if cls[i] == 18:
                    yolox_border[o][0] = bboxes[i][0]
                    yolox_border[o][1] = bboxes[i][1]
                    yolox_border[o][2] = bboxes[i][2]
                    yolox_border[o][3] = bboxes[i][3]
                    o = o + 1
            if package_classes == 'QFN':
                if cls[i] == 15:
                    yolox_pad[n][0] = bboxes[i][0]
                    yolox_pad[n][1] = bboxes[i][1]
                    yolox_pad[n][2] = bboxes[i][2]
                    yolox_pad[n][3] = bboxes[i][3]
                    n = n + 1
            if package_classes == 'SOP':
                if cls[i] == 22:
                    yolox_pin[m][0] = bboxes[i][0]
                    yolox_pin[m][1] = bboxes[i][1]
                    yolox_pin[m][2] = bboxes[i][2]
                    yolox_pin[m][3] = bboxes[i][3]
                    m = m + 1
                if cls[i] == 21:
                    yolox_border[o][0] = bboxes[i][0]
                    yolox_border[o][1] = bboxes[i][1]
                    yolox_border[o][2] = bboxes[i][2]
                    yolox_border[o][3] = bboxes[i][3]
                    o = o + 1
            if package_classes == 'BGA':
                if cls[i] == 26:
                    yolox_BGA_serial_num[s][0] = bboxes[i][0]
                    yolox_BGA_serial_num[s][1] = bboxes[i][1]
                    yolox_BGA_serial_num[s][2] = bboxes[i][2]
                    yolox_BGA_serial_num[s][3] = bboxes[i][3]
                    s = s + 1
                if cls[i] == 27:
                    yolox_BGA_serial_letter[r][0] = bboxes[i][0]
                    yolox_BGA_serial_letter[r][1] = bboxes[i][1]
                    yolox_BGA_serial_letter[r][2] = bboxes[i][2]
                    yolox_BGA_serial_letter[r][3] = bboxes[i][3]
                    r = r + 1
                if cls[i] == 28:
                    yolox_border[o][0] = bboxes[i][0]
                    yolox_border[o][1] = bboxes[i][1]
                    yolox_border[o][2] = bboxes[i][2]
                    yolox_border[o][3] = bboxes[i][3]
                    o = o + 1
                if cls[i] == 29:
                    yolox_pin[m][0] = bboxes[i][0]
                    yolox_pin[m][1] = bboxes[i][1]
                    yolox_pin[m][2] = bboxes[i][2]
                    yolox_pin[m][3] = bboxes[i][3]
                    m = m + 1


    global YOLOX_num
    global YOLOX_other
    global YOLOX_pairs_single  # np.二维数组[x1,y1,x2,y2,0 = outside 1 = inside]
    global YOLOX_serial_num
    global YOLOX_pin
    global YOLOX_pad
    global YOLOX_border
    global YOLOX_angle_pairs
    global YOLOX_BGA_serial_num
    global YOLOX_BGA_serial_letter

    YOLOX_pairs_single = yolox_pairs
    YOLOX_num = yolox_num
    YOLOX_other = yolox_other
    YOLOX_serial_num = yolox_serial_num
    YOLOX_pin = yolox_pin
    YOLOX_pad = yolox_pad
    YOLOX_border = yolox_border
    YOLOX_angle_pairs = yolox_angle_pairs
    YOLOX_BGA_serial_num = yolox_BGA_serial_num
    YOLOX_BGA_serial_letter = yolox_BGA_serial_letter


def begain_output_pairs_data_location(img_path, package_classes):
    global YOLOX_num  # np.二维数组[x1,y1,x2,y2]
    global YOLOX_pairs_single  # np.二维数组[x1,y1,x2,y2,0 = outside 1 = inside]
    global YOLOX_other

    global YOLOX_serial_num  # np.二维数组[x1,y1,x2,y2]
    global YOLOX_pin  # np.二维数组[x1,y1,x2,y2]
    global YOLOX_pad  # np.二维数组[x1,y1,x2,y2]
    global YOLOX_border  # np.二维数组[x1,y1,x2,y2]
    global YOLOX_angle_pairs
    global YOLOX_BGA_serial_num
    global YOLOX_BGA_serial_letter

    global YOLOX_weight  # str

    YOLOX_pairs_single = np.empty((1, 5))  # np.二维数组[x1,y1,x2,y2,0 = outside 1 = inside]
    YOLOX_num = np.empty((1, 4))  # [x1,x2,x3,x4]
    YOLOX_other = np.empty((1, 4))  # [x1,x2,x3,x4]
    YOLOX_serial_num = np.empty((1, 4))
    YOLOX_pin = np.empty((1, 4))
    YOLOX_pad = np.empty((1, 4))
    YOLOX_border = np.empty((1, 4))
    YOLOX_angle_pairs = np.empty((1, 4))
    YOLOX_BGA_serial_num = np.empty((1, 4))
    YOLOX_BGA_serial_letter = np.empty((1, 4))

    # YOLOX_weight = "DFN_SON_0831.onnx"
    if package_classes == 'BGA':
        YOLOX_weight = 'BGA_0730.onnx'
    else:
        YOLOX_weight = "QFP_SOP_QFN.onnx"
    weight_path = 'model/yolo_model/package_model/' + YOLOX_weight
    # onnx_inference(img_path, package_classes, weight='model/yolo_model/package_model/QFP_SOP_QFN.onnx', )
    onnx_inference(img_path, package_classes, weight_path, )

    yolox_num = YOLOX_num
    other = YOLOX_other
    yolox_pairs = YOLOX_pairs_single
    yolox_serial_num = YOLOX_serial_num
    pin = YOLOX_pin
    pad = YOLOX_pad
    border = YOLOX_border
    angle_pairs = YOLOX_angle_pairs
    BGA_serial_num = YOLOX_BGA_serial_num
    BGA_serial_letter = YOLOX_BGA_serial_letter

    return yolox_pairs, yolox_num, yolox_serial_num, pin, other, pad, border, angle_pairs, BGA_serial_num, BGA_serial_letter


if __name__ == '__main__':
    img_path = r'E:\python\PackageWizard1.0\Result\Package_extract\data\bottom.jpg'
    begain_output_pairs_data_location(img_path, package_classes='BGA')
    
