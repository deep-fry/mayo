import tensorflow as tf


class Preprocess(object):
    images_per_shard = 1024
    queue_memory_factor = 16
    num_readers = 4

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _filename_queue(self, mode):
        """
        Queue for file names to read from
        """
        files = self.config.datafiles(mode)
        if mode == 'train':
            shuffle = True
            capacity = 16
        else:
            shuffle = False
            capacity = 1
        return tf.train.string_input_producer(
            files, shuffle=shuffle, capacity=capacity)

    def _queue(self, mode):
        """
        Queue for serialized image data
        """
        min_images_in_queue = self.images_per_shard * self.queue_memory_factor
        batch_size = self.config.train.batch_size
        if mode == 'train':
            # shuffling
            return tf.RandomShuffleQueue(
                min_images_in_queue + 3 * batch_size,
                min_after_dequeue=min_images_in_queue, dtypes=[tf.string])
        return tf.FIFOQueue(
            self.images_per_shard + 3 * batch_size, dtypes=[tf.string])

    def _reader(self):
        """
        File reader
        """
        return tf.TFRecordReader()

    def _serialized_inputs(self, mode):
        """
        Reads data to populate the queue, and pops serialized data from queue
        """
        filename_queue = self._filename_queue(mode)
        if self.num_readers > 1:
            queue = self._queue(mode)
            enqueue_ops = []
            for _ in range(self.num_readers):
                _, value = self._reader().read(filename_queue)
                enqueue_ops.append(queue.enqueue([value]))
            qr = tf.train.queue_runner
            qr.add_queue_runner(qr.QueueRunner(queue, enqueue_ops))
            serialized = queue.dequeue()
        else:
            _, serialized = self._reader().read(filename_queue)
        return serialized

    @staticmethod
    def _parse_proto(proto):
        # dense features
        string = tf.FixedLenFeature([], dtype=tf.string, default_value='')
        integer = tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1)
        feature_map = {
            'image/encoded': string,
            'image/class/label': integer,
            'image/class/text': string,
        }
        float32 = tf.VarLenFeature(dtype=tf.float32)
        # bounding boxes
        bbox_keys = [
            'image/object/bbox/xmin', 'image/object/bbox/ymin',
            'image/object/bbox/xmax', 'image/object/bbox/ymax']
        for k in bbox_keys:
            feature_map[k] = float32

        # parsing
        features = tf.parse_single_example(proto, feature_map)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        # bbox handling
        xmin, ymin, xmax, ymax = (
            tf.expand_dims(features[k].values, 0) for k in bbox_keys)
        # tensorflow imposes an ordering of (y, x) just to make life difficult
        bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])
        # force the variable number of bounding boxes into the shape
        # [1, num_boxes, coords].
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(bbox, [0, 2, 1])

        encoded = features['image/encoded']
        text = features['image/class/text']
        return encoded, label, bbox, text

    @staticmethod
    def _decode_jpeg(image_buffer):
        with tf.name_scope(values=[image_buffer], name='decode_jpeg'):
            i = tf.image.decode_jpeg(image_buffer, channels=3)
            return tf.image.conver_image_dtype(i, dtype=tf.float32)

    def _distort_color(self, i, tid):
        brightness = lambda i: \
            tf.image.random_brightness(i, max_delta=32.0 / 255.0)
        saturation = lambda i: \
            tf.image.random_saturation(i, lower=0.5, upper=1.5)
        hue = lambda i: \
            tf.image.random_hue(i, max_delta=0.2)
        contrast = lambda i: \
            tf.image.random_contrast(i, lower=0.5, upper=1.5)
        if tid % 2 == 0:
            order = [brightness, saturation, hue, contrast]
        else:
            order = [brightness, contrast, saturation, hue]
        with tf.name_scope(values=[i], name='distort_color'):
            for func in order:
                i = func(i)
            i = tf.clip_by_value(i, 0.0, 1.0)
            return i

    def _distort(self, i, bbox, tid=0):
        height = self.config.dataset.height
        width = self.config.dataset.width
        channels = self.config.dataset.channels
        with tf.name_scope(values=[i, bbox], name='distort_image'):
            # distort bbox
            distort_bbox_func = tf.image.sample_distorted_bounding_box
            bbox_begin, bbox_size, distort_bbox = distort_bbox_func(
                tf.shape(i), bounding_boxes=bbox, min_object_covered=0.1,
                aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1.0],
                max_attempts=100, use_image_if_no_bounding_boxes=True)
            # distorted image
            i = tf.slice(i, bbox_begin, bbox_size)
            i = tf.image.resize_images(
                i, [height, width], method=(tid % 4))
            i.set_shape([height, width, channels])
            i = tf.image.random_flip_left_right(i)
            i = self._distort_color(i, tid)
            return i

    def _eval(self, i):
        with tf.name_scope(values=[i], name='eval_image'):
            i = tf.image.central_crop(i, central_fraction=0.875)
            i = tf.expand_dims(i, 0)
            shape = [self.config.dataset.height, self.config.dataset.width]
            i = tf.image.resize_bilinear(i, shape, align_corners=False)
            i = tf.squeeze(i, [0])
            return i

    def _preprocess(self, image_buffer, bbox, mode, tid):
        image = self._decode_jpeg(image_buffer)
        if mode == 'train':
            image = self._distort(image, bbox, tid)
        else:
            image = self._eval(image)
        return tf.multiply(tf.subtract(image, 0.5), 2.0)

    def _unserialize(self, serialized, mode):
        num_threads = self.config.train.num_preprocess_threads
        if num_threads % 4:
            raise ValueError('Expect number of threads to be a multiple of 4.')
        images_labels = []
        for tid in range(num_threads):
            image_buffer, label, bbox, _ = self._parse_proto(serialized)
            image = self._preprocess(image_buffer, bbox, mode, tid)
            images_labels.append((image, label))
        batch_size = self.config.train.batch_size
        capacity = 2 * num_threads * batch_size
        images, labels = tf.train.batch_join(
            images_labels, batch_size=batch_size, capacity=capacity)
        images = tf.cast(images, tf.float32)
        shape = (batch_size, ) + self.config.input_shape()
        images = tf.reshape(images, shape=shape)
        return images, tf.reshape(labels, [batch_size])

    def inputs(self, mode=None):
        mode = mode or self.config.mode
        with tf.name_scope('batch_processing'):
            serialized = self._serialized_inputs(mode)
            return self._unserialize(serialized, mode)
