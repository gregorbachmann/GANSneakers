import os

import tensorflow as tf
from models import FirstNet


class DataReader():

    def __init__(self,
                 data_path,
                 batch_size,
                 num_threads,
                 buffer_size,
                 prefetch_buffer_size,
                 img_size):

        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size
        self.num_threads = num_threads

        self.img_dir_names = self.get_img_paths()
        self.dataset = self.create_dataset()
        self.iterator = self.make_iterator()

    def get_img_paths(self):
        img_dir_names = []
        for img_name in os.listdir(self.data_path):
            if img_name != '.floyddata':
                img_dir_names.append(os.path.join(self.data_path, img_name))

        self.data_size = len(img_dir_names)
        print(self.data_size)

        return img_dir_names

    def parser(self, img_dir):
        img_string = tf.read_file(img_dir)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img = tf.image.resize_images(img_decoded, [self.img_size['width'], self.img_size['height']])
        img = -1 + img/(255/2)

        return img

    def create_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.img_dir_names)
        dataset = dataset.map(map_func=self.parser, num_parallel_calls=self.num_threads)
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.repeat()  # repeat indefinitely
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)

        return dataset

    def make_iterator(self):
        iterator = self.dataset.make_initializable_iterator()

        return iterator