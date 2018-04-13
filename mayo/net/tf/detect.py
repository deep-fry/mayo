import tensorflow as tf

from mayo.error import ShapeError


def box_to_corners(box, unstack=True, stack=True):
    if unstack:
        box = tf.unstack(box, axis=-1)
    y, x, h, w = box
    h_half, w_half = h / 2, w / 2
    corners = [y - h_half, x - w_half, y + h_half, x + w_half]
    if stack:
        return tf.stack(corners, axis=-1)
    return corners


def corners_to_box(corners, unstack=True, stack=True):
    if unstack:
        corners = tf.unstack(corners, axis=-1)
    y_min, x_min, y_max, x_max = corners
    y = (y_min + y_max) / 2
    x = (x_min + x_max) / 2
    h = y_max - y_min
    w = x_max - x_min
    box = [y, x, h, w]
    if stack:
        return tf.stack(box, axis=-1)
    return box


def area(y_max, x_max, y_min, x_min):
    return (y_max - y_min) * (x_max - x_min)


def iou(boxes1, boxes2, anchors=False):
    """
    Compute IOU values from two tensors of bounding boxes.

    boxes1, boxes2:
        a (... x 4) tensor, where the last dimension contains the
        coordinates (x, y, w, h), respectively denote the center of the
        box and its width and height.
    anchors:
        boxes1 and 2 to be (... x 2) tensors containing only
        widths and heights, treating the origin as the centers.

    returns: a (boxes1.shape + boxes2.shape) tensor of IOU values.
    """
    expected_size = 2 if anchors else 4
    if boxes1.shape[1] != expected_size or boxes2.shape[1] != expected_size:
        raise ShapeError(
            'The number of values representing the bounding box should be {}.'
            .format(expected_size))
    shape1 = boxes1.get_shape()[:-1]
    shape2 = boxes2.get_shape()[:-1]
    ndims1 = shape1.ndims
    ndims2 = shape2.ndims
    # num_rows1 x num_row2 x expected_size tensors to form
    # all possible bbox pairs
    reshaped1 = tf.reshape(boxes1, shape1 + [1] * ndims2 + [expected_size])
    tiled1 = tf.tile(reshaped1, tf.concat([[1] * ndims1, shape2, [1]]))
    reshaped2 = tf.reshape(boxes2, [1] * ndims1 + shape2 + [expected_size])
    tiled2 = tf.tile(reshaped2, tf.concat([shape1, [1] * ndims2, [1]]))
    if anchors:
        h1, w1 = tf.unstack(tiled1, axis=-1)
        h2, w2 = tf.unstack(tiled2, axis=-1)
        y1_max, x1_max = h1 / 2, w1 / 2
        y2_max, x2_max = h2 / 2, w2 / 2
        y1_min, x1_min, y2_min, y2_min = -y1_max, -x1_max, -y2_max, -x2_max
    else:
        y1_max, x1_max, y1_min, x1_min = box_to_corners(tiled1, stack=False)
        y2_max, x2_max, y2_min, x2_min = box_to_corners(tiled2, stack=False)
    # intersect corners
    yi_max = tf.minimum(y1_max, y2_max)
    xi_max = tf.minimum(x1_max, x2_max)
    yi_min = tf.maximum(y1_min, y2_min)
    xi_min = tf.maximum(x1_min, x2_min)
    # areas
    area_intersect = area(yi_max, xi_max, yi_min, xi_min)
    area1 = area(y1_max, x1_max, y1_min, x1_min)
    area2 = area(y2_max, x2_max, y2_min, x2_min)
    return area_intersect / (area1 + area2 - area_intersect)


class YOLOv2(object):
    def __init__(self, anchors, image_size, num_cells):
        """
        anchors: a (num_anchors x 2) tensor of anchor boxes [h, w].
        image_size: the height and width of the image.
        num_cells: the number of cells to divide the image into a grid.
        """
        if not image_size % num_cells:
            raise ValueError(
                'The size of image must be divisible by the number of cells.')
        self.anchors = anchors
        self.num_anchors = anchors.shape[0]
        self.image_size = image_size
        self.num_cells = num_cells
        self.cell_size = image_size / num_cells
        self._base_shape = [num_cells, num_cells, self.num_anchors]

    def _transform_truths(self, bboxes, labels):
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
        bboxes *= self.cell_size
        # y, x, h, w are (batch x num_truths)
        y, x, h, w = tf.unstack(bboxes, axis=-1)
        # indices of the bbox-containing cells (batch x num_truths)
        # any position within [k * cell_size, (k + 1) * cell_size)
        # gets mapped to the index k.
        row, col = (tf.cast(tf.floor(v), tf.int32) for v in (y, x))
        # ious: batch x num_truths x num_bboxes
        ious = iou(tf.stack([h, w], axis=-1), self.anchors, anchors=True)
        # bbox indices: (batch x num_truths)
        _, bbox_indices = tf.nn.top_k(ious)
        # coalesced indices (batch x num_truths x 3), where 3 values denote
        # (#row, #column, #anchor)
        indices = tf.stack([row, col, bbox_indices], axis=-1)
        # objectness tensor, batch-wise allocation
        # for each (batch), indices (num_truths x 3) are used to scatter values
        # into a (num_cells x num_cells x num_anchors) tensor.
        # resulting in a (batch x num_cells x num_cells x num_anchors) tensor.
        objectness = tf.map_fn(tf.scatter_nd, (indices, 1, self._base_shape))
        # bboxes
        # adjusted bbox, relative to cell
        # best anchors (batch x num_truths x 2)
        best_anchors = tf.gather_nd(self.anchors, bbox_indices)
        # (batch x num_truths)
        h_anchor, w_anchor = tf.unstack(best_anchors, axis=-1)
        # bboxes (batch x num_truths x 4)
        bboxes = [y - row, x - col, tf.log(h / h_anchor), tf.log(w / w_anchor)]
        bboxes = tf.stack(bboxes, axis=-1)
        # bboxes, batch-wise allocation
        bboxes = tf.scatter_nd(indices, bboxes, self._base_shape + [4])
        # labels, batch-wise allocation
        labels = tf.scatter_nd(indices, labels, self._base_shape)
        return objectness, bboxes, labels

    def loss(predictions, truths):
        pass
