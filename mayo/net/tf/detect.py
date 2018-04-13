import tensorflow as tf

from mayo.error import ShapeError


def box_to_corners(box, y_first=True):
    y, x, h, w = tf.unstack(box, axis=-1)
    h_half, w_half = h / 2, w / 2
    return tf.stack([y - h_half, x - w_half, y + h_half, x + w_half], axis=-1)


def corners_to_box(corners):
    y_min, x_min, y_max, x_max = tf.unstack(corners, axis=-1)
    y = (y_min + y_max) / 2
    x = (x_min + x_max) / 2
    h = y_max - y_min
    w = x_max - x_min
    return tf.stack([y, x, h, w], axis=-1)


def area(y_max, x_max, y_min, x_min):
    return (y_max - y_min) * (x_max - x_min)


def iou(boxes1, boxes2, anchors=False):
    """
    Compute IOU values from two tensors of bounding boxes.

    boxes1, boxes2:
        a (num_rows x 4) tensor, where the last dimension contains the
        coordinates (x, y, w, h), respectively denote the center of the
        box and its width and height.
    anchors:
        boxes1 and 2 to be (num_rows x 2) tensors containing only
        widths and heights, treating the origin as the centers.

    returns: a (num_rows[boxes1], num_rows[boxes2]) tensor of IOU values.
    """
    if boxes1.shape.ndims != 2 and boxes2.shape.ndim != 2:
        raise ShapeError(
            'The number of dimensions of bounding boxes should be 2.')
    expected_size = 2 if anchors else 4
    if boxes1.shape[1] != expected_size or boxes2.shape[1] != expected_size:
        raise ShapeError(
            'The number of values representing the bounding box should be {}.'
            .format(expected_size))
    # num_rows1 x num_row2 x 4 tensors to form all possible bbox pairs
    num_rows1, num_rows2 = boxes1.shape[0], boxes2.shape[0]
    tile1 = tf.tile(tf.expand_dims(boxes1, 1), [1, num_rows2, 1])
    tile2 = tf.tile(tf.expand_dims(boxes2, 0), [num_rows1, 1, 1])
    if anchors:
        h1, w1 = tf.unstack(tile1)
        h2, w2 = tf.unstack(tile2)
        y1_max, x1_max = h1 / 2, w1 / 2
        y2_max, x2_max = h2 / 2, w2 / 2
        y1_min, x1_min, y2_min, y2_min = -y1_max, -x1_max, -y2_max, -x2_max
    else:
        y1_max, x1_max, y1_min, x1_min = tf.unstack(box_to_corners(tile1))
        y2_max, x2_max, y2_min, x2_min = tf.unstack(box_to_corners(tile2))
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


def truths_to_label(bboxes, labels, anchors, image_size, num_cells):
    """
    Allocates ground truth bounding boxes and labels into a cell grid of
    objectness, bounding boxes and labels.  This algorithm follows YOLOv2.

    references:
        https://arxiv.org/pdf/1612.08242.pdf
        https://github.com/allanzelener/YAD2K/blob/master/yad2k/models/keras_yolo.py

    truths:
        a (num_truths x 4) tensor where each row is a
        boudning box denoted by [y, x, h, w].
    labels:
        a (num_truths) tensor of labels.
    anchors:
        a (num_anchors x 4) tensor of anchor boxes [y, x, h, w].
    returns:
        a (num_cells x num_cells x num_anchors) tensor of objectness.
        a (num_cells x num_cells x num_anchors x 4) tensor of bounding boxes.
        a (num_cells x num_cells x num_anchors) tensor of labels.
    """
    if not image_size % num_cells:
        raise ValueError(
            'The size of image must be divisible by the number of cells.')
    cell_size = image_size / num_cells
    # normalize the scale of bounding boxes to the size of each cell
    bboxes *= cell_size
    # y, x, h, w are (num_truths)
    y, x, h, w = tf.unstack(bboxes, axis=1)
    # indices of the bbox-containing cells (num_truths)
    # any position within [k * cell_size, (k + 1) * cell_size)
    # gets mapped to the index k.
    row, col = (tf.cast(tf.floor(v), tf.int32) for v in (y, x))
    # ious: num_truths x num_bboxes
    ious = iou(tf.stack([h, w], axis=-1), anchors, anchors=True)
    # bbox indices: (num_truths)
    _, bbox_indices = tf.nn.top_k(ious)
    indices = tf.stack([row, col, bbox_indices], axis=-1)
    # objectness tensor
    num_anchors = anchors.shape[0]
    output_shape = [num_cells, num_cells, num_anchors]
    objectness = tf.scatter_nd(indices, 1, output_shape)
    # bboxes
    # adjusted bbox, relative to cell
    best_anchors = tf.gather_nd(anchors, bbox_indices)
    h_anchor, w_anchor = tf.unstack(best_anchors, axis=-1)
    bboxes = [y - row, x - col, tf.log(h / h_anchor), tf.log(w / w_anchor)]
    bboxes = tf.stack(bboxes, axis=-1)
    bboxes = tf.scatter_nd(indices, bboxes, output_shape + [4])
    # labels
    labels = tf.scatter_nd(indices, labels, output_shape)
    return objectness, bboxes, labels
