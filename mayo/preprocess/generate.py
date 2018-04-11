import tensorflow as tf

from mayo.log import log
from mayo.util import ensure_list
from mayo.preprocess.augment import Augment


class Preprocess(object):
    def __init__(self, session, mode, config, num_gpus):
        super().__init__()
        self.session = session
        if mode not in ['train', 'validate']:
            raise ValueError(
                'Unrecognized preprocessing mode {!r}'.format(mode))
        self.mode = mode
        self.config = config
        self.num_gpus = num_gpus
        self.batch_size_per_gpu = self.config.system.batch_size_per_gpu
        self.num_threads = self.config.system.preprocess.num_threads
        self.moment = self.config.dataset.get('moment')
        self.image_shape = self.config.image_shape()

    @staticmethod
    def _decode_jpeg(buffer, channels):
        with tf.name_scope(values=[buffer], name='decode_jpeg'):
            i = tf.image.decode_jpeg(buffer, channels=channels)
            return tf.image.convert_image_dtype(i, dtype=tf.float32)

    @staticmethod
    def _parse_proto(proto):
        string = tf.FixedLenFeature([], dtype=tf.string, default_value='')
        integer = tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1)
        float32s = tf.VarLenFeature(dtype=tf.float32)
        # dense features
        feature_map = {
            'image/encoded': string,
            'image/class/label': integer,
            'image/class/text': string,
        }
        # bounding boxes
        bbox_keys = [
            'image/object/bbox/ymin', 'image/object/bbox/xmin',
            'image/object/bbox/ymax', 'image/object/bbox/xmax']
        for k in bbox_keys:
            feature_map[k] = float32s

        # parsing
        features = tf.parse_single_example(proto, feature_map)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        # bbox handling
        # force the variable number of bounding boxes into the shape
        # [1, num_boxes, coords]
        bboxes = [features[k].values for k in bbox_keys]
        bboxes = tf.expand_dims(tf.stack(axis=-1, values=bboxes), 0)

        # other
        encoded = features['image/encoded']
        text = features['image/class/text']
        return encoded, label, bboxes, text

    def _actions(self, key):
        actions_map = self.config.dataset.preprocess
        return ensure_list(actions_map.get(key) or [])

    def _preprocess(self, serialized):
        #  return tf.ones((224, 224, 3), tf.float32), tf.ones(1, tf.int32)
        # unserialize and prepocess image
        buffer, label, bbox, _ = self._parse_proto(serialized)
        # decode jpeg image
        channels = self.image_shape[-1]
        image = self._decode_jpeg(buffer, channels)
        # augment image
        augment = Augment(image, bbox, self.image_shape, self.moment)
        actions = self._actions(self.mode) + self._actions('final_cpu')
        image = augment.augment(actions)
        # add label offset
        offset = self.config.label_offset()
        log.debug('Incrementing label by offset {}...'.format(offset))
        return image, label + offset

    def preprocess(self):
        # file names
        files = self.config.data_files(self.mode)
        num_gpus = self.num_gpus
        batch_size = self.batch_size_per_gpu * num_gpus
        dataset = tf.data.Dataset.from_tensor_slices(files)
        if self.mode == 'train':
            # shuffle .tfrecord files
            dataset = dataset.shuffle(buffer_size=len(files))
        # tfrecord files to images
        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.repeat()
        dataset = dataset.map(
            self._preprocess, num_parallel_calls=self.num_threads)
        dataset = dataset.prefetch(self.num_threads * batch_size)
        if self.mode == 'train':
            buffer_size = min(1024, 10 * batch_size)
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        # iterator
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()
        # ensuring the shape of images and labels to be constants
        batch_shape = (batch_size, )
        images = tf.reshape(images, batch_shape + self.image_shape)
        labels = tf.reshape(labels, batch_shape)
        batch_images_labels = list(zip(
            tf.split(images, num_gpus), tf.split(labels, num_gpus)))
        # final preprocessing on gpu
        gpu_actions = self._actions('final_gpu')
        if gpu_actions:
            for gid, (images, labels) in enumerate(batch_images_labels):
                with tf.device('/gpu:{}'.format(gid)):
                    augment = Augment(
                        images, None, self.image_shape, self.moment, None)
                    images = augment.augment(gpu_actions, ensure_shape=False)
                    batch_images_labels[gid] = (images, labels)
        return batch_images_labels
