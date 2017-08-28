import tensorflow as tf

from mayo.util import memoize


class Preprocess(object):
    images_per_shard = 1024
    queue_memory_factor = 16
    num_readers = 4

    def __init__(self, config):
        super().__init__()
        self.config = config

    @staticmethod
    def _decode_jpeg(image_buffer, channels):
        with tf.name_scope(values=[image_buffer], name='decode_jpeg'):
            i = tf.image.decode_jpeg(image_buffer, channels=channels)
            return tf.image.convert_image_dtype(i, dtype=tf.float32)

    def _distort_bbox(self, i, bbox, tid):
        height, width, channels = self.config.image_shape()
        # distort bbox
        distort_bbox_func = tf.image.sample_distorted_bounding_box
        bbox_begin, bbox_size, _ = distort_bbox_func(
            tf.shape(i), bounding_boxes=bbox, min_object_covered=0.1,
            aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1.0],
            max_attempts=100, use_image_if_no_bounding_boxes=True)
        # distorted image
        i = tf.slice(i, bbox_begin, bbox_size)
        i = tf.image.resize_images(i, [height, width], method=(tid % 4))
        i.set_shape([height, width, channels])
        return i

    @staticmethod
    def _distort_color(i, tid):
        brightness = lambda i: \
            tf.image.random_brightness(i, max_delta=32.0 / 255.0)
        saturation = lambda i: \
            tf.image.random_saturation(i, lower=0.5, upper=1.5)
        hue = lambda i: \
            tf.image.random_hue(i, max_delta=0.2)
        contrast = lambda i: \
            tf.image.random_contrast(i, lower=0.5, upper=1.5)
        # check if grayscale or color
        channels = i.shape[2]
        if channels == 1:
            # unable to distort saturation and hue for image
            has_color = False
        elif channels == 3:
            has_color = True
        else:
            raise ValueError(
                'Expects the number of channels of an image to be '
                'either 1 or 3.')
        if not has_color:
            order = [brightness, contrast]
        elif tid % 2 == 0:
            order = [brightness, saturation, hue, contrast]
        else:
            order = [brightness, contrast, saturation, hue]
        for func in order:
            i = func(i)
        return tf.clip_by_value(i, 0.0, 1.0)

    def _eval(self, i):
        height, width, _ = self.config.image_shape()
        with tf.name_scope(values=[i], name='preprocess_eval'):
            i = tf.image.central_crop(i, central_fraction=0.875)
            i = tf.expand_dims(i, 0)
            shape = [height, width]
            i = tf.image.resize_bilinear(i, shape, align_corners=False)
            return tf.squeeze(i, [0])

    def _distort(self, i, bbox, tid=0):
        params = self.config.dataset.preprocess
        with tf.name_scope(values=[i, bbox], name='preprocess_train'):
            if params.distort:
                i = self._distort_bbox(i, bbox, tid)
            else:
                # otherwise ensure image is the correct size
                i = self._eval(i)
            if params.flip:
                i = tf.image.random_flip_left_right(i)
            if params.color:
                i = self._distort_color(i, tid)
            return i

    def _preprocess(self, image_buffer, bbox, mode, tid):
        channels = self.config.image_shape()[-1]
        image = self._decode_jpeg(image_buffer, channels)
        if mode == 'train':
            image = self._distort(image, bbox, tid)
            return tf.multiply(tf.subtract(image, 0.5), 2.0)
        else:
            #rgb mean
            vgg_mean = [123.68000030517578125,116.779998779296875,103.94000244140625]
            vgg_mean = tf.constant(vgg_mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            image = self._eval(image)
            image = image * 255.0 - vgg_mean
            return image

    def _filename_queue(self, mode):
        """
        Queue for file names to read from
        """
        files = self.config.data_files(mode)
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
        batch_size = self.config.system.batch_size
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
        # [1, num_boxes, coords]
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(bbox, [0, 2, 1])

        encoded = features['image/encoded']
        text = features['image/class/text']
        return encoded, label, bbox, text

    def _unserialize(self, serialized, mode):
        num_threads = self.config.system.num_preprocess_threads
        if num_threads % 4:
            raise ValueError('Expect number of threads to be a multiple of 4.')
        images_labels = []
        for tid in range(num_threads):
            image_buffer, label, bbox, _ = self._parse_proto(serialized)
            image = self._preprocess(image_buffer, bbox, mode, tid)
            label = label - 1
            images_labels.append((image, label))
        batch_size = self.config.system.batch_size
        capacity = 2 * num_threads * batch_size
        images, labels = tf.train.batch_join(
            images_labels, batch_size=batch_size, capacity=capacity)
        images = tf.cast(images, tf.float32)
        shape = (batch_size, ) + self.config.image_shape()
        images = tf.reshape(images, shape=shape)
        return images, tf.reshape(labels, [batch_size])

    def inputs(self, mode):
        with tf.name_scope('batch_processing'):
            serialized = self._serialized_inputs(mode)
            return self._unserialize(serialized, mode)

    def split_inputs(self, mode):
        images, labels = self.inputs(mode)
        num = self.config.system.num_gpus
        split = lambda t: tf.split(axis=0, num_or_size_splits=num, value=t)
        return split(images), split(labels)

    @memoize
    def preprocess_train(self):
        return self.split_inputs('train')

    @memoize
    def preprocess_validate(self):
        return self.split_inputs('validate')
