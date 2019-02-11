import tensorflow as tf


class SummaryManager:
    def __init__(self, sess):
        self.sess = sess
        self.cheap_ops = []
        self.expensive_ops = []
        self.scalar_summaries = []

    def write_summaries(self, summary_outputs, step):
        for summary_output in summary_outputs:
            self._writer.add_summary(summary=summary_output, global_step=step)

    def scalar_add(self, scalar, name):
        self.scalar_summaries.append([scalar, name])

    def _post_model_build(self, output_path):
        # merge all summaries (we do not have to keep track of where we put them,
        # tensorflow is smart enough to find all we put in basenet.py and vgg16.py etc)
        # self.merged_summary = tf.summary.merge_all()

        # merge expensive and cheap operations
        self.merged_cheap_ops = self.merge_operations(self.cheap_ops)
        self.merged_expensive_ops = self.merge_operations(self.expensive_ops)

        # initialize a writer
        self._writer = tf.summary.FileWriter(output_path)

        # add graph only once at beginning of training
        self._writer.add_graph(self.sess.graph)

    @staticmethod
    def merge_operations(operations):
        return tf.summary.merge(operations)

    def scalar_summary(self):
        for scalar, name in self.scalar_summaries:
            op = tf.summary.scalar(name, scalar)
            self.cheap_ops.append(op)

    def scalar(self, scalar, name):
        op = tf.summary.scalar(name, scalar)
        self.cheap_ops.append(op)

    def histogram(self, name, weights):
        op = tf.summary.histogram(name, weights)
        self.expensive_ops.append(op)

    def gradient_norm(self, optimizer, loss, name):
        l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
        gradients = optimizer.compute_gradients(loss=loss)
        n = len(gradients[0])
        self.scalar_summary(l2_norm(gradients[0][0]), name + 'FirstLayer')
        self.scalar_summary(l2_norm(gradients[n//2][0]), name + 'MiddleLayer')
        self.scalar_summary(l2_norm(gradients[n][0]), name + 'LastLayer')