import os
import colorsys

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from PIL import Image, ImageDraw, ImageFont

from mayo.log import log
from mayo.util import memoize_property, pad_to_shape
from mayo.task.image.detect import util
from mayo.task.image.detect.base import ImageDetectTaskBase


class YOLOv2(ImageDetectTaskBase):
    """
    YOLOv2 image detection algorithm.

    references:
        https://arxiv.org/pdf/1612.08242.pdf
        https://github.com/allanzelener/YAD2K/
        https://github.com/experiencor/keras-yolo2/
    """
    _truth_keys = ['bbox/label', 'bbox', 'bbox/count']
    _default_scales = {
        'object': 1,
        'noobject': 0.5,
        'coordinate': 5,
        'class': 1,
    }

    def __init__(
            self, session, preprocess, num_classes, shape,
            anchors, scales, moment=None, num_cells=13,
            iou_threshold=0.6, score_threshold=0.6,
            nms_iou_threshold=0.4, nms_max_boxes=10):
        """
        anchors (tensor):
            a (num_anchors x 2) tensor of anchor boxes [(h, w), ...].
        scales (dict): the weights of the losses.
        num_cells (int): the number of cells to divide the image into a grid.
        iou_threshold (float):
            the threshold of IOU upper-bound to suppress the absense
            of object in training loss.
        score_threshold (float):
            the threshold used to filter less confident detections
            during validation.
        nms_iou_threshold (float):
            the IOU threshold used by non-max suppression during validation.
        """
        self.batch_size = session.batch_size
        self._anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        after_shape = preprocess.shape
        height, width = after_shape['height'], after_shape['width']
        if height != width:
            raise ValueError('We expect the image to be square for YOLOv2.')
        self.image_size = height
        if self.image_size % num_cells:
            raise ValueError(
                'The size of image must be divisible by the number of cells.')
        self.num_cells = num_cells
        self.cell_size = self.image_size / num_cells
        self._base_shape = [num_cells, num_cells, self.num_anchors]

        self.scales = dict(self._default_scales, **scales)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_max_boxes = nms_max_boxes

        super().__init__(session, preprocess, shape, moment)

    @memoize_property
    def anchors(self):
        return tf.constant(self._anchors)

    def _activate(self, box):
        """ Activation functions for bounding box coordinates.  """
        yx, hw = tf.split(box, [2, 2], axis=-1)
        yx = tf.sigmoid(yx)
        hw = tf.exp(hw)
        return [yx, hw]

    def _cell_to_global(self, yx, hw):
        """
        Transform (batch x num_cells x num_cells x num_anchors x 4)
        raw bounding box (after sigmoid and exponentiation) to the
        full-image bounding box.
        """
        # grid setup
        line = tf.range(0, self.num_cells)
        rows = tf.reshape(line, [self.num_cells, 1])
        rows = tf.tile(rows, [1, self.num_cells])
        cols = tf.reshape(line, [1, self.num_cells])
        cols = tf.tile(cols, [self.num_cells, 1])
        grid = tf.stack([rows, cols], axis=-1)
        grid = tf.reshape(grid, [1, self.num_cells, self.num_cells, 1, 2])
        grid = tf.cast(grid, tf.float32)
        # box transformation
        yx += grid
        hw *= tf.reshape(self.anchors, [1, 1, 1, self.num_anchors, 2])
        box = tf.concat([yx, hw], axis=-1) / self.num_cells
        return box

    def _truth_to_cell(self, each):
        """
        Allocates ground truth bounding boxes and labels into a cell grid of
        objectness, bounding boxes and labels.

        box:
            a (num_truths x 4) tensor where each row is a
            boudning box denoted by [y, x, h, w].
        label: a (num_truths) tensor of labels.
        count: num_truths.
        returns:
            - a (num_cells x num_cells x num_anchors)
              tensor of objectness.
            - a (num_cells x num_cells x num_anchors x 4)
              tensor of bounding boxes.
            - a (num_cells x num_cells x num_anchors) tensor of labels.
        """
        box, label, count = each
        with tf.control_dependencies([tf.assert_greater(count, 0)]):
            box = box[:count, :]
            label = label[:count]

        # normalize the scale of bounding boxes to the size of each cell
        # y, x, h, w are (num_truths)
        y, x, h, w = tf.unstack(box * self.num_cells, axis=-1)

        # indices of the box-containing cells (num_truths)
        # any position within [k * cell_size, (k + 1) * cell_size)
        # gets mapped to the index k.
        row_float, col_float = (tf.floor(v) for v in (y, x))
        row, col = (tf.cast(v, tf.int32) for v in (row_float, col_float))

        # ious: num_truths x num_boxes
        pair_box, pair_anchor = util.cartesian(
            tf.stack([h, w], axis=-1), self.anchors)
        ious = util.iou(pair_box, pair_anchor, anchors=True)
        # box indices: (num_truths)
        _, anchor_index = tf.nn.top_k(ious)
        anchor_index_squeezed = tf.squeeze(anchor_index, axis=-1)
        # coalesced indices (num_truths x 3), where 3 values denote
        # (#row, #column, #anchor)
        index = tf.stack([row, col, anchor_index_squeezed], axis=-1)

        # objectness tensor
        # for each (batch), indices (num_truths x 3) are used to scatter values
        # into a (num_cells x num_cells x num_anchors) tensor.
        # resulting in a (batch x num_cells x num_cells x num_anchors) tensor.
        objectness = tf.scatter_nd(index, tf.ones([count]), self._base_shape)

        # boxes
        # anchor_index: (num_truths)
        # best_anchor: (num_truths x 2)
        best_anchor = tf.gather_nd(self.anchors, anchor_index)
        best_anchor = best_anchor
        h_anchor, w_anchor = tf.unstack(best_anchor, axis=-1)
        # adjusted relative to cell row and col (num_truths x 4)
        box = [y - row_float, x - col_float, h / h_anchor, w / w_anchor]
        box = tf.stack(box, axis=-1)
        # boxes
        box = tf.scatter_nd(index, box, self._base_shape + [4])

        # labels
        label = tf.scatter_nd(index, label, self._base_shape)
        return objectness, box, label

    def _filter(self, prediction):
        object_mask, cls, outbox = prediction
        # filter objects with a low confidence score
        # cell x cell x anchors x classes
        confidence = object_mask * cls
        base_shape = [self.num_cells * self.num_cells * self.num_anchors]
        boxes = tf.reshape(outbox, base_shape + [4])
        confidence = tf.reshape(confidence, base_shape + [self.num_classes])
        # cell x cell x anchors
        classes = tf.argmax(confidence, axis=-1, output_type=tf.int32)
        scores = tf.reduce_max(confidence, axis=-1)
        mask = scores >= self.score_threshold
        # only confident objects are left
        boxes = tf.boolean_mask(boxes, mask)
        scores = tf.boolean_mask(scores, mask)
        classes = tf.boolean_mask(classes, mask)

        # non-max suppression
        indices = tf.image.non_max_suppression(
            boxes, scores, self.nms_max_boxes, self.nms_iou_threshold)
        indices = tf.expand_dims(indices, -1)
        boxes = tf.gather_nd(boxes, indices)
        scores = tf.gather_nd(scores, indices)
        classes = tf.gather_nd(classes, indices)
        count = tf.shape(boxes)[0]
        boxes = pad_to_shape(boxes, [self.nms_max_boxes, 4])
        scores = pad_to_shape(scores, [self.nms_max_boxes])
        classes = pad_to_shape(classes, [self.nms_max_boxes])
        return boxes, scores, classes, count

    def transform(self, net, data, prediction, truth):
        """
        Additional tensor decomposition in preprocessing.

        prediction:
            a (batch x num_cells x num_cells x num_anchors x (5 + num_classes))
            prediction, where each element of the last dimension (5 +
            num_classes) consists of a objectness probability (1), a bounding
            box (4), and a one-hot list (num_classes) of class probabilities.
        truth:
            a (batch x num_objects x 5) tensor, where each item of the
            last dimension (5) consists of a bounding box (4), and a label (1).
        """
        prediction = prediction['output']
        batch = util.shape(prediction)[0]
        prediction = tf.reshape(
            prediction,
            (batch, self.num_cells, self.num_cells,
             self.num_anchors, 4 + 1 + self.num_classes))
        box_predict, obj_predict, hot_predict = tf.split(
            prediction, [4, 1, self.num_classes], axis=-1)
        obj_predict = tf.nn.sigmoid(obj_predict)
        obj_predict_squeeze = tf.squeeze(obj_predict, -1)
        yx_predict, hw_predict = self._activate(box_predict)
        prediction = {
            'object': obj_predict_squeeze,
            'object_mask': obj_predict,
            'box': tf.concat([yx_predict, hw_predict], axis=-1),
            'outbox': self._cell_to_global(yx_predict, hw_predict),
            'class': hot_predict,
        }
        inputs = (
            prediction['object_mask'], prediction['class'],
            prediction['outbox'])
        box_test, score_test, class_test, count_test = tf.map_fn(
            self._filter, inputs,
            dtype=(tf.float32, tf.float32, tf.int32, tf.int32))
        prediction['test'] = {
            'box': box_test,
            'class': class_test,
            'score': score_test,
            'count': count_test,
        }
        # the original box and label values from the dataset
        if truth:
            rawlabel, truebox, count = truth
            obj, box, label = tf.map_fn(
                self._truth_to_cell, (truebox, rawlabel, count),
                dtype=(tf.float32, tf.float32, tf.int32))
            truth = {
                'object': obj,
                'object_mask': tf.expand_dims(obj, -1),
                'box': box,
                'class': slim.one_hot_encoding(label, self.num_classes),
                'count': count,
                'rawbox': truebox,
                'rawclass': rawlabel,
            }
        return data['input'], prediction, truth

    @memoize_property
    def colors(self):
        for i in range(self.num_classes):
            rgb = colorsys.hsv_to_rgb(i / self.num_classes, 0.5, 1)
            yield tuple(int(c * 255) for c in rgb)

    def _test(self, name, image, boxes, scores, classes, count):
        height, width, channels = image.shape
        image = Image.fromarray(np.uint8(255.0 * image))
        image = image.convert('RGBA')
        thickness = int((height + width) / 300)
        log.info('{}: {} boxes.'.format(name.decode(), count))
        font = os.path.join(os.path.split(__file__)[0], 'opensans.ttf')
        font = ImageFont.truetype(font, 12)
        max_score = max(scores)
        boxes = boxes[:count]
        for box, score, cls in zip(boxes, scores, classes):
            layer = Image.new('RGBA', image.size, (255, 255, 255, 0))
            draw = ImageDraw.ImageDraw(layer)
            y, x, h, w = box
            y, x, h, w = y * height, x * width, h * height, w * width
            top = max(0, round(y - h / 2))
            bottom = min(height, round(y + h / 2))
            left = max(0, round(x - w / 2))
            right = min(width, round(x + w / 2))
            transparency = 127 + int(128 * score / max_score)
            color = self.colors[cls] + (transparency, )
            for i in range(thickness):
                draw.rectangle(
                    (left + i, top + i, right - i, bottom - i), outline=color)
            # draw label
            cls_name = self.class_names[cls]
            label = ' {} {:.2f} '.format(cls_name, score)
            label_width, label_height = draw.textsize(label, font=font)
            label_pos = [left, top]
            label_rect = [left + label_width, top + label_height]
            draw.rectangle(label_pos + label_rect, fill=color)
            draw.text(label_pos, label, fill=(0, 0, 0, 127), font=font)
            image = Image.alpha_composite(image, layer)
            log.info(
                '  Confidence: {:.2f}, class: {}, box: {}'
                .format(score, cls_name, ((top, left), (bottom, right))))
        path = self.session.config.system.search_path.run.outputs[0]
        path = os.path.join(path, 'detect')
        os.makedirs(path, exist_ok=True)
        name = os.path.split(str(name))[1]
        name, ext = os.path.splitext(name)
        path = os.path.join(path, '{}.png'.format(name))
        image.save(path, quality=90)

    def test(self, names, images, predictions):
        test = predictions['test']
        boxes, scores, classes, count = \
            test['box'], test['score'], test['class'], test['count']
        for args in zip(names, images, boxes, scores, classes, count):
            self._test(*args)

    def _iou_score(self, pred_box, truth_box, num_objects):
        shape = [
            self.batch_size, self.num_cells, self.num_cells,
            self.num_anchors, 1, 4]
        outbox_p = tf.reshape(pred_box, shape)
        shape = [self.batch_size, 1, 1, 1, num_objects, 4]
        outbox = tf.reshape(truth_box, shape)
        iou_score = util.iou(outbox_p, outbox)
        return tf.reduce_max(iou_score, axis=4, keepdims=True)

    def train(self, net, prediction, truth):
        """Training loss.  """
        num_objects = tf.shape(truth['rawbox'])[1]

        # coordinate loss
        coord_loss = tf.reduce_sum(
            (prediction['box'] - truth['box']) ** 2, axis=-1)
        coord_loss = tf.reduce_sum(truth['object'] * coord_loss)
        coord_loss *= self.scales['coordinate']

        # objectness loss
        obj_loss = truth['object'] * (1 - prediction['object']) ** 2
        obj_loss = self.scales['object'] * tf.reduce_sum(obj_loss)

        # class loss
        class_loss = (truth['class'] - prediction['class']) ** 2
        class_loss = tf.reduce_sum(truth['object_mask'] * class_loss)
        class_loss *= self.scales['class']

        # no-object loss
        # match shape
        # (batch x num_cells x num_cells x num_anchors x num_objects x 4)
        iou_score = self._iou_score(
            prediction['outbox'], truth['rawbox'], num_objects)
        is_obj_absent = tf.cast(iou_score <= self.iou_threshold, tf.float32)
        noobj_loss = (1 - truth['object_mask']) * is_obj_absent
        noobj_loss *= prediction['object_mask'] ** 2
        noobj_loss = self.scales['noobject'] * tf.reduce_sum(noobj_loss)

        return obj_loss + noobj_loss + coord_loss + class_loss
