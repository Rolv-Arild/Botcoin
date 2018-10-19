import pandas

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import CsvDataset
from tensorflow.python.ops.rnn import dynamic_rnn


class SimpleBitcoinPredictor:
    def __init__(self, encoding_size, label_size):
        # Model constants
        cell_state_size = 128

        # Cells
        cell = tf.contrib.rnn.LSTMCell(cell_state_size)

        # Model input
        self.batch_size = tf.placeholder(tf.int32,
                                         [], name='batch_size')  # Needed by cell.zero_state call, and is dependent on usage (training or generation)
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


data = pandas.read_csv("../resources/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv")

xy = data.get_values().tolist()

encodings_size = 8
alphabet_size = 8

model = SimpleBitcoinPredictor(encodings_size, alphabet_size)

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(500).minimize(model.loss)

sample_size = 10000
batch_size = 1


# Create session for running TensorFlow operations
with tf.Session() as session:
    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    # Initialize model.in_state
    zero_state = session.run(model.in_state, {model.batch_size: batch_size})

    for epoch in range(500):
        for i in range(len(xy) - sample_size):
            sample = xy[i:i + sample_size + 1]
            session.run(minimize_operation,
                        {model.batch_size: batch_size, model.x: [sample[:-1]], model.y: [sample[-1]],
                         model.in_state: zero_state})

            print("loss", session.run(model.loss,
                                      {model.batch_size: batch_size, model.x: [sample[:-1]], model.y: [sample[-1]],
                                       model.in_state: zero_state}))

        print("epoch", epoch)

        session.close()
