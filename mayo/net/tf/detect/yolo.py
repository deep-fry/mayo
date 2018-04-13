import tensorflow as tf
from tensorflow.contrib import slim

from mayo.net.tf.detect.util import cartesian, iou


class YOLOv2(object):
    def __init__(
            self, config, anchors,
            object_scale=1, noobject_scale=0.5,
            coordinate_scale=5, class_scale=1,
            num_cells=13, iou_threshold=0.6):
        """
        anchors: a (num_anchors x 2) tensor of anchor boxes [h, w].
        num_cells: the number of cells to divide the image into a grid.
        iou_threshold:
            the threshold of IOU upper-bound to suppress the absense
            of object in loss.
        """
        self.anchors = anchors
        self.num_anchors = anchors.shape[0]
        height, width = config.image_shape()
        if height != width:
            raise ValueError('We expect the image to be square for YOLOv2.')
        self.image_size = height
        if not self.image_size % num_cells:
            raise ValueError(
                'The size of image must be divisible by the number of cells.')
        self.num_cells = num_cells
        self.cell_size = self.image_size / num_cells
        self.iou_threshold = iou_threshold
        self._base_shape = [num_cells, num_cells, self.num_anchors]

    def transform_truth(self, box, label):
        """
        Allocates ground truth bounding boxes and labels into a cell grid of
        objectness, bounding boxes and labels.

        references:
            https://arxiv.org/pdf/1612.08242.pdf
            https://github.com/allanzelener/YAD2K/blob/master/yad2k/models/keras_yolo.py

        truths:
            a (batch x num_truths x 4) tensor where each row is a
            boudning box denoted by [y, x, h, w].
        labels:
            a (batch x num_truths) tensor of labels.
        returns:
            - a (batch x num_cells x num_cells x num_anchors)
              tensor of objectness.
            - a (batch x num_cells x num_cells x num_anchors x 4)
              tensor of bounding boxes.
            - a (batch x num_cells x num_cells x num_anchors) tensor of labels.
        """
        # normalize the scale of bounding boxes to the size of each cell
        # y, x, h, w are (batch x num_truths)
        y, x, h, w = tf.unstack(box * self.cell_size, axis=-1)

        # indices of the box-containing cells (batch x num_truths)
        # any position within [k * cell_size, (k + 1) * cell_size)
        # gets mapped to the index k.
        row, col = (tf.cast(tf.floor(v), tf.int32) for v in (y, x))

        # ious: batch x num_truths x num_boxes
        pair_box, pair_anchor = cartesian(
            tf.stack([h, w], axis=-1), self.anchors)
        ious = iou(pair_box, pair_anchor)
        # box indices: (batch x num_truths)
        _, anchor_index = tf.nn.top_k(ious)
        # coalesced indices (batch x num_truths x 3), where 3 values denote
        # (#row, #column, #anchor)
        index = tf.stack([row, col, anchor_index], axis=-1)

        # objectness tensor, batch-wise allocation
        # for each (batch), indices (num_truths x 3) are used to scatter values
        # into a (num_cells x num_cells x num_anchors) tensor.
        # resulting in a (batch x num_cells x num_cells x num_anchors) tensor.
        objectness = tf.map_fn(tf.scatter_nd, (index, 1, self._base_shape))

        # boxes
        # best anchors
        best_anchor = tf.gather_nd(self.anchors, anchor_index)
        # (batch x num_truths)
        h_anchor, w_anchor = tf.unstack(best_anchor, axis=-1)
        # adjusted relative to cell row and col (batch x num_truths x 4)
        # FIXME shouldn't we adjust to the center of each grid cell,
        # rather than its top-left?
        box = [y - row, x - col, tf.log(h / h_anchor), tf.log(w / w_anchor)]
        box = tf.stack(box, axis=-1)
        # boxes, batch-wise allocation
        box = tf.scatter_nd(index, box, self._base_shape + [4])

        # labels, batch-wise allocation
        label = tf.scatter_nd(index, label, self._base_shape)
        return objectness, box, label

    def loss(self, prediction, truth):
        obj_p, box_p, class_p = prediction
        obj, box, label = self.transform_truth(truth)
        # one-hot encoding for labels
        onehot = slim.one_hot_encoding(label, self.num_classes)
        # coordinate loss
        xy_p, wh_p = tf.split(box_p, [2, 2], axis=-1)
        xy, wh = tf.split(box, [2, 2], axis=-1)
        coord_loss = (xy - xy_p) ** 2 + (tf.sqrt(wh) - tf.sqrt(wh_p)) ** 2
        coord_loss = self.coordinate_scale * tf.reduce_sum(obj * coord_loss)
        # objectness loss
        obj_loss = self.object_scale * tf.reduce_sum(obj * (1 - obj_p) ** 2)
        # class loss
        class_loss = tf.reduce_sum(obj * (onehot - class_p) ** 2)
        class_loss *= self.class_scale
        # no-object loss
        iou_score = iou(box_p, box)
        is_obj_absent = tf.cast(iou_score <= self.iou_threshold, tf.int32)
        noobj_loss = tf.reduce_sum((1 - obj) * is_obj_absent * obj_p ** 2)
        noobj_loss *= self.noobject_scale
        return obj_loss + coord_loss + class_loss + noobj_loss
