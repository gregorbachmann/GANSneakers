import tensorflow as tf


class SummaryManager:
    def __init__(self, sess, each_step, cheap_ops, expensive_ops):
        self.each_step = each_step
        self.sess = sess
        self.cheap_ops = cheap_ops
        self.expensive_ops = expensive_ops

    def scalar_summary(self, scalar, name):
        op = tf.summary.scalar(name, scalar)
        self.cheap_ops.append(op)

    def histogram(self, name, weights):
        op = tf.summary.histogram(name, weights)
        self.expensive_ops.append(op)

    def gradient_norm(self, optimizer, loss, name):
        l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
        gradients = optimizer.compute_gradients(loss=loss)
        n = len(gradients[0])
        self.scalar_summary(name + 'FirstLayer', l2_norm(gradients[0][0]))
        self.scalar_summary(name + 'MiddleLayer', l2_norm(gradients[n//2][0]))
        self.scalar_summary(name + 'LastLayer', l2_norm(gradients[n][0]))