import collections

import tensorflow as tf

from mayo.log import log
from mayo.util import memoize, multi_objects_from_params


class _ImagePreprocess(object):
    def __init__(self, shape, means, bbox, tid):
        super().__init__()
        self.shape = shape
        self.means = means
        self.bbox = bbox
        self.tid = tid

    def distort_bbox(self, i):
        # distort bbox
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(i), bounding_boxes=self.bbox,
            min_object_covered=0.1, aspect_ratio_range=[0.75, 1.33],
            area_range=[0.05, 1.0],
            max_attempts=100, use_image_if_no_bounding_boxes=True)
        # distorted image
        i = tf.slice(i, bbox_begin, bbox_size)
        height, width, _ = self.shape
        # we resize to the final preprocessed shape in distort bbox
        # because we can use different methods randomly, which _ensure_shape()
        # cannot do
        i = tf.image.resize_images(i, [height, width], method=(self.tid % 4))
        i.set_shape(self.shape)
        return i

    def distort_color(self, i):
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
        elif self.tid % 2 == 0:
            order = [brightness, saturation, hue, contrast]
        else:
            order = [brightness, contrast, saturation, hue]
        for func in order:
            i = func(i)
        return tf.clip_by_value(i, 0.0, 1.0)

    def central_crop(self, i, central_fraction=0.875):
        return tf.image.central_crop(i, central_fraction=central_fraction)

    def random_flip(self, i):
        return tf.image.random_flip_left_right(i)

    def linear_map(self, i, scale=1, shift=0):
        if scale != 1:
            i = tf.multiply(i, scale)
        if shift != 0:
            i = tf.add(i, shift)
        return i

    def subtract_channel_means(self, i):
        if not self.means:
            log.warn(
                'Channel means not found in "dataset.channel_means", '
                'defaulting to 0.5 for each channel.')
            self.means = [0.5] * i.shape[2]
        shape = [1, 1, len(self.means)]
        means = tf.constant(self.means, shape=shape, name='image_means')
        return i - means

    def subtract_image_mean(self, i):
        return i - tf.reduce_mean(i)

    def permute_channels(self, i, order):
        channels = len(order)
        channel_splits = tf.split(i, channels, axis=-1)
        permuted_splits = [channel_splits[o] for o in order]
        return tf.concat(permuted_splits, -1)

    def _ensure_shape(self, i):
        # ensure image is the correct shape
        ph, pw, pc = i.shape.as_list()
        h, w, c = self.shape
        if pc != c:
            # convert channels
            if pc == 3 and c == 1:
                i = tf.image.rgb_to_grayscale(i)
            elif pc == 1:
                # duplicate image channel
                i = tf.concat([i] * c, axis=-1)
            else:
                raise ValueError(
                    'We do not know how to convert an image with {} channels '
                    'into one with {} channels.'.format(pc, c))
        if ph == h or pw == w:
            return i
        # rescale image
        i = tf.expand_dims(i, 0)
        i = tf.image.resize_bilinear(i, [h, w], align_corners=False)
        return tf.squeeze(i, [0])

    def preprocess(self, image, actions):
        with tf.name_scope(values=[image], name='preprocess_image'):
            for func, params in multi_objects_from_params(actions, self):
                log.debug(
                    'Preprocessing using {!r} with params {}'
                    .format(func.__name__, params))
                image = func(image, **params)
        return self._ensure_shape(image)


class Preprocess(object):
    images_per_shard = 1024
    queue_memory_factor = 16
    num_readers = 4

    def __init__(self, config):
        super().__init__()
        self.config = config

    @staticmethod
    def _decode_jpeg(buffer, channels):
        with tf.name_scope(values=[buffer], name='decode_jpeg'):
            i = tf.image.decode_jpeg(buffer, channels=channels)
            return tf.image.convert_image_dtype(i, dtype=tf.float32)

    def _preprocess(self, buffer, bbox, mode, tid):
        channels = self.config.image_shape()[-1]
        image = self._decode_jpeg(buffer, channels)
        shape = self.config.image_shape()
        means = self.config.dataset.get('channel_means', None)
        image_preprocess = _ImagePreprocess(shape, means, bbox, tid)
        actions_map = self.config.dataset.preprocess
        mode_actions = actions_map[mode] or []
        if not isinstance(mode_actions, collections.Sequence):
            mode_actions = [mode_actions]
        final_actions = actions_map['final'] or []
        if not isinstance(final_actions, collections.Sequence):
            final_actions = [final_actions]
        return image_preprocess.preprocess(image, mode_actions + final_actions)

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
            log.debug('Preprocessing thread #{}'.format(tid))
            buffer, label, bbox, _ = self._parse_proto(serialized)
            image = self._preprocess(buffer, bbox, mode, tid)
            offset = self.config.label_offset()
            log.debug('Incrementing label by offset {}'.format(offset))
            label += offset
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
        return self.inputs('validate')
