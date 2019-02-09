import tensorflow as tf

from src.core.ganbase import AdversarialNetwork


class DeeperNet(AdversarialNetwork):

    def generator(self, noisy_input):
        with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Dense1', reuse=tf.AUTO_REUSE):
                dense1 = tf.layers.Dense(units=7*7*512)
                x = dense1(noisy_input)
                x = tf.reshape(x, (-1, 7, 7, 512))
                # Output: 7x7x256
            with tf.variable_scope('TransposeConv1', reuse=tf.AUTO_REUSE):
                deconv1 = tf.layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')
                x = deconv1(x)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.leaky_relu(x, alpha=0.01)
                # Output: 14x14x128
            with tf.variable_scope('TransposeConv2', reuse=tf.AUTO_REUSE):
                deconv2 = tf.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')
                x = deconv2(x)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.leaky_relu(x, alpha=0.01)
                # Output: 28x28x128
            with tf.variable_scope('TransposeConv3', tf.AUTO_REUSE):
                deconv3 = tf.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')
                x = deconv3(x)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.leaky_relu(x, alpha=0.01)
                # Output: 56x56x64
            with tf.variable_scope('TransposeConv4', tf.AUTO_REUSE):
                deconv4 = tf.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')
                x = deconv4(x)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.leaky_relu(x, alpha=0.01)
                # Output: 112x112x32
            with tf.variable_scope('TransposeConv5', tf.AUTO_REUSE):
                deconv5 = tf.layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding='same')
                x = deconv5(x)
                output = tf.nn.tanh(x)
                # Output: 224x224x3

            return output, x

    def discriminator(self, x):
        with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Conv1', reuse=tf.AUTO_REUSE):
                conv1 = tf.layers.Conv2D(8, kernel_size=3, strides=2, padding='same')
                x = conv1(x)
                x = tf.nn.leaky_relu(x)
            with tf.variable_scope('Conv2', reuse=tf.AUTO_REUSE):
                conv2 = tf.layers.Conv2D(16, kernel_size=5, strides=2, padding='same')
                x = conv2(x)
                x = tf.nn.leaky_relu(x)
            with tf.variable_scope('Conv3', reuse=tf.AUTO_REUSE):
                conv3 = tf.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')
                x = conv3(x)
                x = tf.nn.leaky_relu(x)
            with tf.variable_scope('Conv4', reuse=tf.AUTO_REUSE):
                conv3 = tf.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')
                x = conv3(x)
                x = tf.nn.leaky_relu(x)
            with tf.variable_scope('Conv5', reuse=tf.AUTO_REUSE):
                conv3 = tf.layers.Conv2D(128, kernel_size=3, strides=2, padding='same')
                x = conv3(x)
                x = tf.nn.leaky_relu(x)
            with tf.variable_scope('Conv6', reuse=tf.AUTO_REUSE):
                conv3 = tf.layers.Conv2D(256, kernel_size=3, strides=2, padding='same')
                x = conv3(x)
                x = tf.nn.leaky_relu(x)
            with tf.variable_scope('Output', reuse=tf.AUTO_REUSE):
                x = tf.layers.flatten(x)
                dense = tf.layers.Dense(units=1)
                logits = dense(x)
                output = tf.nn.sigmoid(logits)

            return output, logits
