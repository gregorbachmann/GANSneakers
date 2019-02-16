import tensorflow as tf

from src.core.ganbase import AdversarialNetwork


class ResNetGAN(AdversarialNetwork):

    def generator(self, noisy_input):
        with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Dense1', reuse=tf.AUTO_REUSE):
                dense1 = tf.layers.Dense(units=6*4*1024)
                x = dense1(noisy_input)
                x = tf.reshape(x, (-1, 4, 6, 1024))
                # Output: 4x6x1024
            with tf.variable_scope('TransposeConv1', reuse=tf.AUTO_REUSE):
                deconv1 = tf.layers.Conv2DTranspose(512, kernel_size=3, strides=2, padding='same')
                x = deconv1(x)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.leaky_relu(x, alpha=0.01)
                # Output: 8x12x512
            with tf.variable_scope('TransposeConvBlock1', reuse=tf.AUTO_REUSE):
                x = self.transposed_conv_block(x, [256, 512, 512])
                # Output: 16x24x256
            with tf.variable_scope('TransposeIdentityBlock1', tf.AUTO_REUSE):
                x = self.transposed_identity_block(x, [512, 512, 256])
                # Output: 16x24x256
            with tf.variable_scope('TransposeIdentityBlock2', tf.AUTO_REUSE):
                x = self.transposed_identity_block(x, [512, 512, 256])
                # Output: 16x24x256
            with tf.variable_scope('TransposeConvBlock2', tf.AUTO_REUSE):
                x = self.transposed_conv_block(x, [128, 256, 256])
                # Output: 32x48x128
            with tf.variable_scope('TransposeIdentityBlock3', tf.AUTO_REUSE):
                x = self.transposed_identity_block(x, [256, 256, 128])
                # Output: 32x48x128
            with tf.variable_scope('TransposeConvBlock3', tf.AUTO_REUSE):
                x = self.transposed_conv_block(x, [64, 128, 128])
                # Output: 64x96x64
            with tf.variable_scope('TransposeIdentityBlock4', tf.AUTO_REUSE):
                x = self.transposed_identity_block(x, [128, 128, 64])
                # Output: 64x96x64
            with tf.variable_scope('TransposeConv2', tf.AUTO_REUSE):
                deconv2 = tf.layers.Conv2DTranspose(16, kernel_size=3, strides=1, padding='same')
                x = deconv2(x)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.leaky_relu(x, alpha=0.1)
                # Output: 64x96x16
            with tf.variable_scope('TransposeConv3', tf.AUTO_REUSE):
                deconv3 = tf.layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same')
                x = deconv3(x)
                output = tf.nn.tanh(x)
                # Output: 64x96x3

            return output, x

    def discriminator(self, x):
        with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Conv1', reuse=tf.AUTO_REUSE):
                conv1 = tf.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')
                x = conv1(x)
                x = tf.nn.leaky_relu(x)
            with tf.variable_scope('ConvBlock1', reuse=tf.AUTO_REUSE):
                x = self.conv_block(x, [64, 32, 32])
            with tf.variable_scope('IdentityBlock1', reuse=tf.AUTO_REUSE):
                x = self.identity_block(x, [64, 32, 32])
            with tf.variable_scope('IdentityBlock2', reuse=tf.AUTO_REUSE):
                x = self.identity_block(x, [64, 32, 32])
            with tf.variable_scope('ConvBlock2', reuse=tf.AUTO_REUSE):
                x = self.conv_block(x, [128, 64, 64])
            with tf.variable_scope('IdentityBlock3', reuse=tf.AUTO_REUSE):
                x = self.identity_block(x, [128, 64, 64])
            with tf.variable_scope('ConvBlock3', reuse=tf.AUTO_REUSE):
                x = self.conv_block(x, [256, 128, 128])
            with tf.variable_scope('IdentityBlock4', reuse=tf.AUTO_REUSE):
                x = self.identity_block(x, [256, 128, 128])
            with tf.variable_scope('ConvBlock4', reuse=tf.AUTO_REUSE):
                x = self.conv_block(x, [512, 256, 256])
            with tf.variable_scope('IdentityBlock5', reuse=tf.AUTO_REUSE):
                x = self.identity_block(x, [512, 256, 256])
            with tf.variable_scope('ConvBlock5', reuse=tf.AUTO_REUSE):
                x = self.conv_block(x, [1024, 512, 512])
            with tf.variable_scope('Output', reuse=tf.AUTO_REUSE):
                x = tf.layers.flatten(x)
                dense = tf.layers.Dense(units=1)
                logits = dense(x)
                output = tf.nn.sigmoid(logits)

            return output, logits

    def transposed_identity_block(self, example, filters):

        deconv1 = tf.layers.Conv2DTranspose(filters[0], kernel_size=3, strides=1, padding='same')
        x = deconv1(example)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x, alpha=0.01)

        deconv2 = tf.layers.Conv2DTranspose(filters[1], kernel_size=3, strides=1, padding='same')
        x = deconv2(x)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x, alpha=0.01)

        deconv3 = tf.layers.Conv2DTranspose(filters[2], kernel_size=3, strides=1, padding='same')
        x = deconv3(x)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.add(x, example)
        x = tf.nn.leaky_relu(x, alpha=0.01)

        return x

    def transposed_conv_block(self, example, filters):

        conv_path_1 = tf.layers.Conv2DTranspose(filters[0], kernel_size=3, strides=2, padding='same')
        x1 = conv_path_1(example)
        x1 = tf.layers.batch_normalization(x1)
        x1 = tf.nn.leaky_relu(x1, alpha=0.01)

        conv1_path_2 = tf.layers.Conv2DTranspose(filters[1], kernel_size=3, strides=1, padding='same')
        x2 = conv1_path_2(example)
        x2 = tf.layers.batch_normalization(x2)
        x2 = tf.nn.leaky_relu(x2, alpha=0.01)

        conv2_path_2 = tf.layers.Conv2DTranspose(filters[2], kernel_size=3, strides=1, padding='same')
        x2 = conv2_path_2(x2)
        x2 = tf.layers.batch_normalization(x2)
        x2 = tf.nn.leaky_relu(x2, alpha=0.01)

        conv3_path_2 = tf.layers.Conv2DTranspose(filters[0], kernel_size=3, strides=2, padding='same')
        x2 = conv3_path_2(x2)
        x2 = tf.layers.batch_normalization(x2)
        x2 = tf.nn.leaky_relu(x2, alpha=0.01)

        x = tf.add(x1, x2)
        x = tf.nn.leaky_relu(x, alpha=0.01)

        return x

    def identity_block(self, example, filters):

        conv1 = tf.layers.Conv2D(filters[1], kernel_size=3, strides=1, padding='same')
        x = conv1(example)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x)

        conv2 = tf.layers.Conv2D(filters[2], kernel_size=3, strides=1, padding='same')
        x = conv2(x)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x)

        conv3 = tf.layers.Conv2D(filters[0], kernel_size=3, strides=1, padding='same')
        x = conv3(x)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x)

        x = tf.add(x, example)
        x = tf.nn.leaky_relu(x)

        return x

    def conv_block(self, example, filters):
        conv_path_1 = tf.layers.Conv2D(filters[0], kernel_size=3, strides=2, padding='same')
        x1 = conv_path_1(example)

        conv1_path_2 = tf.layers.Conv2D(filters[1], kernel_size=3, strides=1, padding='same')
        x2 = conv1_path_2(example)
        x2 = tf.layers.batch_normalization(x2)
        x2 = tf.nn.leaky_relu(x2)

        conv2_path_2 = tf.layers.Conv2D(filters[2], kernel_size=3, strides=1, padding='same')
        x2 = conv2_path_2(x2)
        x2 = tf.layers.batch_normalization(x2)
        x2 = tf.nn.leaky_relu(x2)

        conv3_path_2 = tf.layers.Conv2D(filters[0], kernel_size=3, strides=2, padding='same')
        x2 = conv3_path_2(x2)
        x2 = tf.layers.batch_normalization(x2)
        x2 = tf.nn.leaky_relu(x2)

        x = tf.add(x1, x2)
        x = tf.nn.leaky_relu(x)

        return x
