import os

import tensorflow as tf
from models.FirstNet import FirstNet
from datasources.data import DataReader


def main():

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        project_dir = os.path.split(dir_path)[0]
        data_path = os.path.join(project_dir, 'data')

        batch_size = 16

        num_threads = 4
        buffer_size = 5000
        prefetch_buffer_size = 1  # number of batches consumed by one training step
        num_epochs = 1000
        img_size = {'width': 224, 'height': 224}
        noise_size = 1000

        # Initialize data Object
        data = DataReader(data_path=data_path,
                          batch_size=batch_size,
                          num_threads=num_threads,
                          buffer_size=buffer_size,
                          prefetch_buffer_size=prefetch_buffer_size,
                          img_size=img_size)

        learning_schedule = {'d_rate': 1e-4, 'g_rate': 1e-3, 'num_epochs': num_epochs, 'd_steps': 1, 'g_steps': 1}

        # Initialize model object
        model = FirstNet(tf_session=sess,
                         learning_schedule=learning_schedule,
                         data=data,
                         noise_size=noise_size)
        model.train()


if __name__ == "__main__":
    main()