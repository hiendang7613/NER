import tensorflow as tf
from tensorflow_addons.text import crf_log_likelihood

def crf_log_likelihood(self, y_pres, labels, scale=None, training=False):

    loss = 0
    for i, y_pre in enumerate(y_pres):
        _, potentials, sequence_length, chain_kernel = y_pre

        label = tf.stack([labels[i], 1-labels[i]])

        loss += -crf_log_likelihood(potentials, label, sequence_length, chain_kernel)[0]

    if scale is not None:
        loss *= scale

    return tf.reduce_mean(loss)