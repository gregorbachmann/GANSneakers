import tensorflow as tf
import numpy as np
import cv2 as cv2
import os

from src.core.SummaryManager import SummaryManager
from src.core.CheckpointManager import CheckpointManager
from src.core.TimeManager import TimeManager


class AdversarialNetwork:

    def __init__(self,
                 tf_session: tf.Session,
                 data,
                 noise_size,
                 learning_schedule,
                 cheap_ops_step,
                 expensive_ops_step,
                 output_path,
                 save_each_step,
                 vis_each_step):

        self.sess = tf_session
        self.learning_schedule = learning_schedule
        self.data = data
        self.noise_size = noise_size
        self.cheap_ops_step = cheap_ops_step
        self.expensive_ops_step = expensive_ops_step
        self.output_path = output_path
        self.save_each_step = save_each_step
        self.vis_each_step = vis_each_step

        self.checkpoint_manager = CheckpointManager(self)
        self.time_manager = TimeManager(self)
        self.losses, self.train_step = self.build_graph()

    def generator(self, noisy_input):
        raise NotImplementedError('BaseModel::generator is not yet implemented.')

    def discriminator(self):
        raise NotImplementedError('BaseModel::discriminator is not yet implemented.')

    def noise_generator(self):
        noise_input = np.random.uniform(low=-1, high=1, size=[self.data.batch_size, self.noise_size])
        return noise_input

    def smoothed_labels(self, labels):
        smooth = np.random.uniform(low=-0.2, high=0.2, size=labels.shape)
        if labels[0] == 0:
            smooth_labels = labels + smooth + 0.2
        else:
            smooth_labels = labels + smooth

        return smooth_labels

    def switch(self, real_labels, fake_labels, step):
        # Draw decaying Bernoulli to decide whether to flip labels
        p = 1 - 1/((8+step)**(1/3))
        flip = np.random.binomial(n=1, p=p)
        if flip == 0:
            return fake_labels, real_labels
        else:
            return real_labels, fake_labels

    def visualize(self, step):
        noisy_input = self.noise_generator()
        picture, _ = self.sess.run(self.generator(self.noise_input), feed_dict={self.noise_input: noisy_input})
        print(self.output_path)
        path = os.path.join(self.output_path, 'output_images')
        print(path)
        cv2.imwrite(path + '/vis' + str(step) + '.jpg', (picture[0]+1)*255/2)

    def build_graph(self):
        with tf.name_scope('RealExamples'):
            real_batch = self.data.iterator.get_next()

        with tf.name_scope('FakeExamples'):
            self.noise_input = tf.placeholder(dtype=tf.float32, shape=[None, self.noise_size])
            fake_batch, _ = self.generator(self.noise_input)

        _, d_real_logits = self.discriminator(real_batch)
        _, d_fake_logits = self.discriminator(fake_batch)

        with tf.name_scope('Labels'):
            self.real_labels_ph = tf.placeholder(dtype=tf.float32, shape=None)
            self.fake_labels_ph = tf.placeholder(dtype=tf.float32, shape=None)

        with tf.variable_scope('D-Loss'):
            # Get loss for predicting on real data
            d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.real_labels_ph, logits=d_real_logits)
            # Get loss for predicting on fake data
            d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.fake_labels_ph, logits=d_fake_logits)
            # Combine the two
            d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

        with tf.variable_scope('G-loss'):
            # Get loss for generator
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake_logits),
                                                                            logits=d_fake_logits))

        trainable_vars = tf.trainable_variables()
        with tf.variable_scope('D-Optimizer'):
            d_vars = [var for var in trainable_vars if var.name.startswith("Discriminator")]
            d_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_schedule['d_rate'])
            d_train_step = d_optimizer.minimize(d_loss, var_list=d_vars)

        with tf.variable_scope('G-Optimizer'):
            g_vars = [var for var in trainable_vars if var.name.startswith("Generator")]
            g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_schedule['g_rate'])
            g_train_step = g_optimizer.minimize(g_loss, var_list=g_vars)

        losses = {'d': d_loss, 'g': g_loss}
        train_step = {'d': d_train_step, 'g': g_train_step}

        return losses, train_step

    def train(self):

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.run(self.data.iterator.initializer)

        self.checkpoint_manager.build_saver()
        initial_step = self.checkpoint_manager.load_all()
        save_step = initial_step

        d_losses_tot = []
        g_losses_tot = []
        step = 0

        for e in range(self.learning_schedule['num_epochs']):
            d_losses = []
            g_losses = []
            for i in range(self.data.data_size//self.data.batch_size):
                # Update the discriminator for d steps
                for _ in range(self.learning_schedule['d_steps']):
                    # Create smoothed random labels
                    real_labels = self.smoothed_labels(np.ones(self.data.batch_size))
                    fake_labels = self.smoothed_labels(np.zeros(self.data.batch_size))
                    # Maybe switch them
                    real_labels, fake_labels = self.switch(real_labels, fake_labels, step)
                    # Make a gradient step
                    _, d_loss = self.sess.run([self.train_step['d'], self.losses['d']],
                                              feed_dict={self.noise_input: self.noise_generator(),
                                                         self.real_labels_ph: real_labels,
                                                         self.fake_labels_ph: fake_labels})

                    d_losses.append(d_loss)
                    step += 1

                # Update the generator for g steps
                for _ in range(self.learning_schedule['g_steps']):
                    _, g_loss = self.sess.run([self.train_step['g'], self.losses['g']],
                                              feed_dict={self.noise_input: self.noise_generator()})
                    g_losses.append(g_loss)
                    step += 1

                print('epoch', str(e),
                      '\t\t| substep =', str(i),
                      '\t\t| total steps =', str(step),
                      '\t\t| d-loss on batch =', str(round(d_loss, 4)),
                      '\t\t| g-loss on batch =', str(round(g_loss, 4)))

                if step % 10 == 0:
                    self.visualize(step)

                if step % self.save_each_step == 0:
                    self.checkpoint_manager.save_all(save_step)

            d_losses_tot.append(np.mean(d_losses))
            g_losses_tot.append(np.mean(g_losses))

