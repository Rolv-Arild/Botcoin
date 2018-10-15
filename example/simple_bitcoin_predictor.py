import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import CsvDataset
from tensorflow.python.ops.rnn import dynamic_rnn


class SimpleBitcoinPredictor:
    def __init__(self, encodings_size, alphabet_size):
        # Model constants
        cell_state_size = 128

        # Cells
        cell = tf.contrib.rnn.LSTMCell(cell_state_size)

        # Model input
        # Needed by cell.zero_state call, and can be dependent on usage (training or generation)
        self.batch_size = tf.placeholder(tf.int32, [])

        # Shape: [batch_size, max_time, encodings_size]
        self.x = tf.placeholder(tf.float32, [None, None, encodings_size])
        self.y = tf.placeholder(tf.float32, [None, alphabet_size])

        # Can be used as either an input or a way to get the zero state
        self.in_state = cell.zero_state(self.batch_size, tf.float32)

        # Model variables
        W = tf.Variable(tf.random_normal([cell_state_size, alphabet_size]))
        b = tf.Variable(tf.random_normal([alphabet_size]))

        # Model operations
        # lstm has shape: [batch_size, max_time, cell_state_size]
        lstm, self.out_state = dynamic_rnn(cell, self.x, initial_state=self.in_state)

        # Logits, where tf.einsum multiplies a batch of txs matrices (lstm) with W
        logits = tf.nn.bias_add(tf.matmul(lstm[:, -1, :], W), b)  # b: batch, t: time, s: state, e: encoding

        # Predictor
        self.f = logits

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)


filenames = ["resources/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv"]
record_defaults = [tf.float32] * 8   # Eight required float columns
dataset = CsvDataset(filenames, record_defaults)

x = dataset.batch(2630880)  # 5 years (2012.01.01 - 2017.01.01)
y = dataset.skip(2630880)

print(x, y)

print(dataset)
x_train = x
y_train = y

model = SimpleBitcoinPredictor(8, 8)

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.05).minimize(model.loss)

# Create session for running TensorFlow operations
with tf.Session() as session:
    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    # Initialize model.in_state
    zero_state = session.run(model.in_state, {model.batch_size: 7})

    for epoch in range(500):
        session.run(minimize_operation,
                    {model.batch_size: 1, model.x: x_train, model.y: y_train, model.in_state: zero_state})

        if epoch % 10 == 9:
            print("epoch", epoch)
            print("loss", session.run(model.loss, {model.batch_size: 1, model.x: x_train, model.y: y_train,
                                                   model.in_state: zero_state}))

            state = session.run(model.in_state, {model.batch_size: 1})
            text = 'hat'
            print(text)
