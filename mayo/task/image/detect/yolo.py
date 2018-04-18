import tensorflow as tf
from tensorflow.contrib import slim

from mayo.task.image.base import ImageTaskBase
from mayo.task.image.detect.util import cartesian, iou


class YOLOv2(ImageTaskBase):
    """
    YOLOv2 image detection algorithm.

    references:
        https://arxiv.org/pdf/1612.08242.pdf
        https://github.com/allanzelener/YAD2K/blob/master/yad2k/models/keras_yolo.py
    """
    def __init__(
            self, session, anchors,
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
        super().__init__(session)
        self.anchors = anchors
        self.num_anchors = anchors.shape[0]
        height, width = config.image_shape()
        if height != width:
            raise ValueError('We expect the image to be square for YOLOv2.')
        self.image_size = height
        self.batch_size = config.dataset.batch_size
        if not self.image_size % num_cells:
            raise ValueError(
                'The size of image must be divisible by the number of cells.')
        self.num_cells = num_cells
        self.cell_size = self.image_size / num_cells
        self.iou_threshold = iou_threshold
        self._base_shape = [num_cells, num_cells, self.num_anchors]
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.coordinate_scale = coordinate_scale
        self.class_scale = class_scale

    def transform_truth(self, box, label):
        """
        Allocates ground truth bounding boxes and labels into a cell grid of
        objectness, bounding boxes and labels.

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

    def _cell_to_global(self, box):
        """
        Transform (batch x num_cells x num_cells x num_anchors x 4)
        raw bounding box (before sigmoid and exponentiation) to the
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
        grid *= self.cell_size
        # box transformation
        yx, hw = tf.split(box, [2, 2], axis=-1)
        yx = grid + tf.sigmoid(yx)
        hw = tf.exp(hw)
        hw *= tf.reshape(self.anchors, [1, 1, 1, self.num_anchors, 2])
        box = tf.concat([yx, hw], axis=-1)
        # normalize box position to 0-1
        return box / self.image_size

    def loss(self, prediction, truth):
        """
        tensor:
            a (batch x num_cells x num_cells x num_anchors x (5 + num_classes))
            prediction, where each element of the last dimension (5 +
            num_classes) consists of a objectness probability (1), a bounding
            box (4), and a one-hot list (num_classes) of class probabilities.
        truth:
            a (batch x num_objects x 5) tensor, where each item of the
            last dimension (5) consists of a bounding box (4), and a label (1).
        """
        cell_obj_p, cell_box_p, cell_onehot_p = tf.split(
            prediction, [1, 4, self.num_classes], axis=-1)
        # the original box and label values from the dataset
        global_box, raw_label = tf.split(truth, [4, 1])
        cell_obj, cell_box, cell_label = self.transform_truth(
            global_box, raw_label)
        num_objects = global_box.shape[1]

        # coordinate loss
        xy_p, wh_p = tf.split(cell_box_p, [2, 2], axis=-1)
        xy, wh = tf.split(cell_box, [2, 2], axis=-1)
        coord_loss = (xy - xy_p) ** 2 + (tf.sqrt(wh) - tf.sqrt(wh_p)) ** 2
        coord_loss = tf.reduce_sum(cell_obj * coord_loss)
        coord_loss *= self.coordinate_scale

        # objectness loss
        obj_loss = tf.reduce_sum(cell_obj * (1 - cell_obj_p) ** 2)
        obj_loss *= self.object_scale

        # class loss
        cell_onehot = slim.one_hot_encoding(cell_label, self.num_classes)
        class_loss = cell_obj * (cell_onehot - cell_onehot_p) ** 2
        class_loss = self.class_scale * tf.reduce_sum(class_loss)

        # no-object loss
        # match shape
        # (batch x num_cells x num_cells x num_anchors x num_objects x 4)
        global_box_p = self._cell_to_global(cell_box_p)
        global_box_p = tf.reshape(
            global_box_p,
            [self.batch_size, self.num_cells, self.num_cells,
             self.num_anchors, 1, 4])
        global_box = tf.reshape(
            global_box, [self.batch_size, 1, 1, 1, num_objects, 4])
        iou_score = iou(global_box_p, global_box)
        iou_score = tf.reduce_max(iou_score, axis=4, keepdims=True)
        is_obj_absent = tf.cast(iou_score <= self.iou_threshold, tf.int32)
        noobj_loss = (1 - cell_obj) * is_obj_absent * cell_obj_p ** 2
        noobj_loss = self.noobject_scale * tf.reduce_sum(noobj_loss)

        return obj_loss + coord_loss + class_loss + noobj_loss
