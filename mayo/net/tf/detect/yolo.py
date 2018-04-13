import tensorflow as tf

from mayo.net.tf.detect.util import iou


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
