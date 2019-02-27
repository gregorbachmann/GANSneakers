import os
import pathlib
import numpy as np

import tensorflow as tf
from src.models.DeeperNet import DeeperNet
from datasources.data import DataReader


def main():

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    floydhub = True

    with tf.Session(config=config) as sess:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        project_dir = os.path.split(dir_path)[0]
        if floydhub:
            data_path = '/my_data'
        else:
            data_path = os.path.join(project_dir, 'data')
        output_path = project_dir + '/outputs/DeeperNet'
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        batch_size = 16

        num_threads = 2
        buffer_size = 2000
        prefetch_buffer_size = 1  # number of batches consumed by one training step
        num_epochs = 1
        img_size = {'width': 256, 'height': 384}
        noise_size = 500
        save_each_step = 1000
        vis_each_step = 250

        # Initialize data Object
        data = DataReader(data_path=data_path,
                          batch_size=batch_size,
                          num_threads=num_threads,
                          buffer_size=buffer_size,
                          prefetch_buffer_size=prefetch_buffer_size,
                          img_size=img_size)

        learning_schedule = {'d_rate': 1e-3, 'g_rate': 1e-4, 'num_epochs': num_epochs, 'd_steps': 1, 'g_steps': 1}
        theta = np.pi/2

        # Initialize model object
        model = DeeperNet(tf_session=sess,
                          learning_schedule=learning_schedule,
                          data=data,
                          noise_size=noise_size,
                          cheap_ops_step=1,
                          expensive_ops_step=1,
                          output_path=output_path,
                          save_each_step=save_each_step,
                          vis_each_step=vis_each_step,
                          theta=theta)
        model.train()
        model.test()


if __name__ == "__main__":
    main()