import os
import os.path as osp
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from learning.colors import COLORS
import cv2
import json
from imageio import imsave


def maybe_mkdir(*directories):
    for d in directories:
        if not osp.isdir(d):
            os.mkdir(d)


def plot_learning_curve(exp_idx, step_losses, step_scores, eval_scores=None,
                        mode='max', img_dir='.'):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(np.arange(1, len(step_losses) + 1), step_losses, marker='')
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('Number of iterations')
    axes[1].plot(np.arange(1, len(step_scores) + 1),
                 step_scores, color='b', marker='')
    if eval_scores is not None:
        axes[1].plot(np.arange(1, len(eval_scores) + 1),
                     eval_scores, color='r', marker='')
    if mode == 'max':
        axes[1].set_ylim(0.5, 1.0)
    else:    # mode == 'min'
        axes[1].set_ylim(0.0, 0.5)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Number of epochs')

    # Save plot as image file
    plot_img_filename = 'learning_curve-result{}.svg'.format(exp_idx)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fig.savefig(os.path.join(img_dir, plot_img_filename))

    # Save details as pkl file
    pkl_filename = 'learning_curve-result{}.pkl'.format(exp_idx)
    with open(os.path.join(img_dir, pkl_filename), 'wb') as fo:
        pkl.dump([step_losses, step_scores, eval_scores], fo)


def nms(boxes, conf_thres=0.2, iou_thres=0.5):
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]
    areas = (x2 - x1) * (y2 - y1)
    scores = boxes[..., 4]

    keep = []
    order = scores.argsort()[::-1]

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    nms_box = []
    for idx in range(len(boxes)):
        if idx in keep and boxes[idx, 4] > conf_thres:
            nms_box.append(boxes[idx])
        else:
            nms_box.append(np.zeros(boxes.shape[-1]))
    boxes = np.array(nms_box)
    return boxes


def convert_boxes(input_y):
    is_batch = len(input_y.shape) == 5
    if not is_batch:
        input_y = np.expand_dims(input_y, 0)
    boxes = np.reshape(input_y, (input_y.shape[0], -1, input_y.shape[-1]))
    if is_batch:
        return np.array(boxes)
    else:
        return boxes[0]


def predict_nms_boxes(input_y, conf_thres=0.2, iou_thres=0.5):
    is_batch = len(input_y.shape) == 5
    if not is_batch:
        input_y = np.expand_dims(input_y, 0)
    boxes = np.reshape(input_y, (input_y.shape[0], -1, input_y.shape[-1]))
    nms_boxes = []
    for box in boxes:
        nms_box = nms(box, conf_thres, iou_thres)
        nms_boxes.append(nms_box)
    if is_batch:
        return np.array(nms_boxes)
    else:
        return nms_boxes[0]


def cal_recall(gt_bboxes, bboxes, iou_thres=0.5):
    p = 0
    tp = 0
    for idx, (gt, bbox) in enumerate(zip(gt_bboxes, bboxes)):
        gt = gt[np.nonzero(np.any(gt > 0, axis=1))]
        bbox = bbox[np.nonzero(np.any(bbox > 0, axis=1))]
        p += len(gt)
        if bbox.size == 0:
            continue
        iou = _cal_overlap(gt, bbox)
        predicted_class = np.argmax(bbox[...,5:], axis=-1)
        for g, area in zip(gt, iou):
            gt_c = np.argmax(g[5:])
            idx = np.argmax(area)
            if np.max(area) > iou_thres and predicted_class[idx] == gt_c:
                tp += 1
    return tp / p

def cal_F1(gt_bboxes, bboxes, iou_thres=0.5):
    p = 0
    tp = 0
    pp = 0
    for idx, (gt, bbox) in enumerate(zip(gt_bboxes, bboxes)):
        gt = gt[np.nonzero(np.any(gt > 0, axis=1))]
        bbox = bbox[np.nonzero(np.any(bbox > 0, axis=1))]
        p += len(gt)
        pp += len(bbox)
        if bbox.size == 0:
            continue
        iou = _cal_overlap(gt, bbox)
        predicted_class = np.argmax(bbox[...,5:], axis=-1)
        for g, area in zip(gt, iou):
            gt_c = np.argmax(g[5:])
            idx = np.argmax(area)
            if np.max(area) > iou_thres and predicted_class[idx] == gt_c:
                tp += 1
    recall = tp / p
    precision = tp / pp

    f1 = 2*precision*recall / (precision + recall)

    return f1

def _cal_overlap(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - \
        np.maximum(np.expand_dims(a[:, 0], axis=1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - \
        np.maximum(np.expand_dims(a[:, 1], axis=1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    intersection = iw * ih

    ua = np.expand_dims((a[:, 2] - a[:, 0]) *
                        (a[:, 3] - a[:, 1]), axis=1) + area - intersection

    ua = np.maximum(ua, np.finfo(float).eps)

    return intersection / ua

def draw_pred_boxes(image, pred_boxes, class_map, text=True, score=False):
    im_h, im_w = image.shape[:2]
    output = image.copy()
    for box in pred_boxes:
        overlay = output.copy()

        class_idx = np.argmax(box[5:])
        color = COLORS[class_idx]
        line_width, alpha = (2, 0.8)
        x_min, x_max = [int(x * im_w) for x in [box[0], box[2]]]
        y_min, y_max = [int(x * im_h) for x in [box[1], box[3]]]
        cv2.rectangle(overlay, (x_min, y_min),
                      (x_max, y_max), color, line_width)
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

        if text:
            p_text = str(round(np.max(box[5:]), 3)) if score else class_map[str(class_idx)]
            y_offset = -6
            text_size = 0.6
            text_line_width = 1
            output = cv2.putText(output, p_text, (x_min + 4, y_min + y_offset),
                                 cv2.FONT_HERSHEY_DUPLEX, text_size, color, text_line_width)

    return output


def iou(b1, b2):
    l1, l2 = b1[0] - int(0.5 * b1[2]), b2[0] - int(0.5 * b2[2])
    u1, u2 = l1 + b1[2] - 1, l2 + b2[2] - 1
    intersection_w = max(0, min(u1, u2) - max(l1, l2) + 1)
    if intersection_w == 0:
        return 0
    l1, l2 = b1[1] - int(0.5 * b1[3]), b2[1] - int(0.5 * b2[3])
    u1, u2 = l1 + b1[3] - 1, l2 + b2[3] - 1
    intersection_h = max(0, min(u1, u2) - max(l1, l2) + 1)
    intersection = intersection_w * intersection_h
    if intersection == 0:
        return 0

    union = b1[2] * b1[3] + b2[2] * b2[3] - intersection
    if union == 0:
        raise ValueError('Union value must not be a zero or negative number. (boxes: {}, {})'.format(b1, b2))

    return intersection / union


def kmeans_iou(boxes, k, n_iter=10):
    n_boxes = len(boxes)
    if k > n_boxes:
        raise ValueError('k({}) must be less than or equal to the number of boxes({}).'.format(k, n_boxes))

    # Update clusters and centroids alternately.
    centroids = boxes[np.random.choice(n_boxes, k, replace=False)]
    for _ in range(n_iter):
        cluster_indices = np.array([np.argmax([iou(b, c) for c in centroids]) for b in boxes])
        for i in range(k):
            if np.count_nonzero(cluster_indices == i) == 0:
                print(i)
        centroids = [np.mean(boxes[cluster_indices == i], axis=0) for i in range(k)]

    return np.array(centroids)


def calculate_anchor_boxes(im_paths, anno_dir, num_anchors):
    boxes = []

    for im_path in im_paths:
        im_h, im_w = scipy.misc.imread(im_path).shape[:2]

        anno_fname = '{}.anno'.format(osp.splitext(osp.basename(im_path))[0])
        anno_fpath = osp.join(anno_dir, anno_fname)
        if not osp.isfile(anno_fpath):
            print('ERROR | Corresponding annotation file is not found: {}'.format(anno_fpath))
            return

        with open(anno_fpath, 'r') as f:
            anno = json.load(f)
        for class_name in anno:
            for x_min, y_min, x_max, y_max in anno[class_name]:
                center_x, center_y = (x_max + x_min) * 0.5, (y_max + y_min) * 0.5
                width, height = x_max - x_min + 1, y_max - y_min + 1
                boxes.append([center_x, center_y, width, height])

    boxes = np.array(boxes, dtype=np.float32)
    anchors = kmeans_iou(boxes, num_anchors, 10)[:, 2:]

    return np.array(anchors)
