"""Manage saving and loading of model checkpoints."""
import os
import re

import tensorflow as tf


class CheckpointManager(object):
    """Manager to coordinate saving and loading of trainable parameters."""

    def __init__(self, model):
        """Initialize manager based on given model instance."""
        self.sess = model.sess
        self._model = model

        self._saver = None

    def build_saver(self):
        """Create tf.train.Saver instances"""
        # sorted(..., key=...): The value of the key parameter should be a function
        # that takes a single argument and returns a key to use for sorting purposes
        all_saveable_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) +
                                   tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS) +
                                   tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES) +
                                   tf.get_collection_ref('batch_norm_non_trainable'),
                                   key=lambda v: v.name)

        self._saver = tf.train.Saver(all_saveable_vars, max_to_keep=5)

    def load_all(self):
        """Load all available weights"""
        iteration_number = 0
        output_path = self._model.output_path + '/checkpoints'
        checkpoint = tf.train.get_checkpoint_state(output_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            try:
                # Attempt to restore saveable variables
                self._saver.restore(self.sess, '%s/%s' % (output_path, checkpoint_name))
                iteration_number = int(next(re.finditer("(\d+)(?!.*\d)", checkpoint_name)).group(0))
            except Exception as e:
                import traceback
                traceback.print_exc()
        return iteration_number

    def save_all(self, iteration_number):
        output_path = self._model.output_path + '/checkpoints'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        self._saver.save(self.sess,
                         output_path + '/model.ckpt',
                         global_step=iteration_number)

        # also save graph in *.pbtxt file
        # graph = tf.get_default_graph()
        tf.train.write_graph(self.sess.graph_def,
                             output_path,
                             'graph.pbtxt',
                             as_text=True)

        # tf.io.write_graph(self._tensorflow_session,
        #                   output_path,
        #                   name='graph.pbtxt',
        #                   as_text=True)

