import os
import colorsys

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from PIL import Image, ImageDraw

from mayo.util import memoize_property
from mayo.task.image.base import ImageTaskBase
from mayo.task.image.detect import util


class YOLOv2(ImageTaskBase):
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
            nms_iou_threshold=0.5, nms_max_boxes=10):
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
        grid = tf.cast(grid, tf.float32) * self.cell_size
        # box transformation
        yx += grid
        hw *= tf.reshape(self.anchors, [1, 1, 1, self.num_anchors, 2])
        box = tf.concat([yx, hw], axis=-1)
        # normalize box position to 0-1
        return box / self.image_size

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
        count = 3
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
        # filter objects with a low confidence score
        # batch x cell x cell x anchors x classes
        confidence = prediction['object_mask'] * prediction['class']
        # batch x cell x cell x anchors
        classes = tf.argmax(confidence, axis=-1)
        scores = tf.reduce_max(confidence, axis=-1)
        mask = scores >= self.score_threshold
        # only confident objects are left
        boxes = tf.boolean_mask(prediction['outbox'], mask)
        scores = tf.boolean_mask(scores, mask)
        classes = tf.boolean_mask(classes, mask)

        # non-max suppression
        fn = lambda each: tf.image.non_max_suppression(
            each[0], each[1], self.nms_max_boxes, self.nms_iou_threshold)
        indices = tf.map_fn(fn, (boxes, scores), dtype=tf.int32)
        boxes = tf.gather_nd(boxes, indices)
        scores = tf.gather_nd(scores, indices)
        classes = tf.gather_nd(classes, indices)
        return boxes, scores, classes

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
        batch, height, width, channels = util.shape(prediction)
        prediction = tf.reshape(
            prediction,
            (batch, height, width, self.num_anchors, self.num_classes + 5))
        box_predict, obj_predict, hot_predict = tf.split(
            prediction, [4, 1, self.num_classes], axis=-1)
        obj_predict_squeeze = tf.squeeze(obj_predict, -1)
        yx_predict, hw_predict = self._activate(box_predict)
        prediction = {
            'object': obj_predict_squeeze,
            'object_mask': obj_predict,
            'box': tf.concat([yx_predict, hw_predict], axis=-1),
            'outbox': self._cell_to_global(yx_predict, hw_predict),
            'class': hot_predict,
        }
        box_test, score_test, class_test = self._filter(prediction)
        prediction['test'] = {
            'box': box_test,
            'class': class_test,
            'score': score_test,
        }
        # the original box and label values from the dataset
        if truth:
            label, truebox, count = truth
            obj, box, label = tf.map_fn(
                self._truth_to_cell, (truebox, label, count),
                dtype=(tf.float32, tf.float32, tf.int32))
            truth = {
                'object': obj,
                'object_mask': tf.expand_dims(obj, -1),
                'box': box,
                'rawbox': truebox,
                'class': slim.one_hot_encoding(label, self.num_classes),
                'count': count,
            }
        return data, prediction, truth

    @memoize_property
    def colors(self):
        for i in range(self.num_classes):
            rgb = colorsys.hsv_to_rgb((i / self.num_classes, 1, 1))
            yield [int(c * 255) for c in rgb]

    def test(self, name, image, prediction):
        height, width, _ = image.size
        image = Image.fromarray(np.uint8(image))
        boxes, scores, classes = prediction['test']
        thickness = int((height + width) / 300)
        for box, score, cls in zip(boxes, scores, classes):
            draw = ImageDraw(image)
            y, x, h, w = box
            top = max(0, round(y - h / 2))
            bottom = min(height, round(y + h / 2))
            left = max(0, round(x - w / 2))
            right = min(width, round(x + w / 2))
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[cls])
            # draw label
            label = '{} {:.2f}'.format(self.class_names[cls], score)
            label_width, label_height = draw.textsize(label)
            label_pos = [
                left, max(0, top - label_height),
                left + label_width, top + label_height]
            draw.rectangle(label_pos, fill=self.colors[cls])
            draw.text(label_pos[:2], label, fill=(0, 0, 0))
        path = self.session.config.system.search_path.run
        path = os.path.join(path, 'classify')
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, name)
        image.save(path, quality=90)

    def eval(self, net, prediction, truth):
        ...

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
        shape = [
            self.batch_size, self.num_cells, self.num_cells,
            self.num_anchors, 1, 4]
        outbox_p = tf.reshape(prediction['outbox'], shape)
        shape = [self.batch_size, 1, 1, 1, num_objects, 4]
        outbox = tf.reshape(truth['rawbox'], shape)
        iou_score = util.iou(outbox_p, outbox)
        iou_score = tf.reduce_max(iou_score, axis=4, keepdims=True)
        is_obj_absent = tf.cast(iou_score <= self.iou_threshold, tf.float32)
        noobj_loss = (1 - truth['object_mask']) * is_obj_absent
        noobj_loss *= prediction['object_mask'] ** 2
        noobj_loss = self.scales['noobject'] * tf.reduce_sum(noobj_loss)

        return obj_loss + noobj_loss + coord_loss + class_loss
