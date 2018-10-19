import pandas

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import CsvDataset
from tensorflow.python.ops.rnn import dynamic_rnn

from util.util import normalize_data, find_increase


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


data = pandas.read_csv("../resources/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv")

x = data.get_values().tolist()
for r in x:
    r.pop(0)

x = x[2625376:]  # start of 2017

x, maxes = normalize_data(x)
y = find_increase(x, -1)

encodings_size = len(x[1])
alphabet_size = len(y[1])

model = SimpleBitcoinPredictor(encodings_size, alphabet_size)

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.05).minimize(model.loss)

sample_size = 10000
batch_size = 100

# Create session for running TensorFlow operations
with tf.Session() as session:
    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    # Initialize model.in_state
    zero_state = session.run(model.in_state, {model.batch_size: batch_size})

    for epoch in range(500):
        for i in range(0, (len(x) - sample_size) // batch_size, batch_size):
            sample = [x[i + j:i + j + sample_size + 1] for j in range(batch_size)]
            sample_y = [y[i + j] for j in range(batch_size)]
            session.run(minimize_operation,
                        {model.batch_size: batch_size,
                         model.x: sample,
                         model.y: sample_y,
                         model.in_state: zero_state})

            print("i:", i, ", loss", session.run(model.loss,
                                                 {model.batch_size: batch_size,
                                                  model.x: sample,
                                                  model.y: sample_y,
                                                  model.in_state: zero_state}))

        print("epoch", epoch)

    session.close()
