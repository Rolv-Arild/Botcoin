import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn


class SimpleBitcoinPredictor:
    def __init__(self, encoding_size, label_size):
        # Model constants
        cell_state_size = 128

        # Cells
        cell = tf.contrib.rnn.LSTMCell(cell_state_size)

        # Model input
        self.batch_size = tf.placeholder(tf.int32,
                                         [],
                                         name='batch_size')  # Needed by cell.zero_state call, and is dependent on usage (training or generation)
        self.x = tf.placeholder(tf.float32,
                                [None, None, encoding_size], name='x')  # Shape: [batch_size, max_time, encoding_size]
        self.y = tf.placeholder(tf.float32, [None, label_size], name='y')  # Shape: [batch_size, label_size]
        self.in_state = cell.zero_state(self.batch_size,
                                        tf.float32)  # Can be used as either an input or a way to get the zero state

        # Model variables
        W = tf.Variable(tf.random_normal([cell_state_size, label_size]), name='W')
        b = tf.Variable(tf.random_normal([label_size]), name='b')

        # Model operations
        lstm, self.out_state = dynamic_rnn(cell, self.x,
                                           initial_state=self.in_state)  # lstm has shape: [batch_size, max_time, cell_state_size]

        # Logits, where only the last time frame of lstm is used
        logits = tf.nn.bias_add(tf.matmul(lstm[:, -1, :], W), b)

        # Predictor
        self.f = logits

        # Cross Entropy loss
        self.loss = tf.losses.mean_squared_error(self.y, logits)

        # Accuracy
        self.accuracy = tf.norm(tf.reduce_mean(self.f-self.y))
