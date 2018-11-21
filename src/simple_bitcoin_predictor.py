import time

import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn


class SimpleBitcoinPredictor:
    def __init__(self, encoding_size, label_size, dtype=tf.float16):
        # Model constants
        cell_state_size = 128

        # Cells
        cell = tf.contrib.rnn.LSTMCell(cell_state_size)

        # Model input
        self.batch_size = tf.placeholder(tf.int32,
                                         [],
                                         name='batch_size')  # Needed by cell.zero_state call, and is dependent on usage (training or generation)
        self.x = tf.placeholder(dtype,
                                [None, None, encoding_size], name='x')  # Shape: [batch_size, max_time, encoding_size]
        self.y = tf.placeholder(dtype, [None, label_size], name='y')  # Shape: [batch_size, label_size]
        self.in_state = cell.zero_state(self.batch_size,
                                        dtype)  # Can be used as either an input or a way to get the zero state

        # Model variables
        W = tf.Variable(tf.random_normal([cell_state_size, label_size], dtype=dtype), name='W')
        b = tf.Variable(tf.random_normal([label_size], dtype=dtype), name='b')

        # Model operations
        lstm, self.out_state = dynamic_rnn(cell, self.x,
                                           initial_state=self.in_state)  # lstm has shape: [batch_size, max_time, cell_state_size]

        # Logits, where only the last time frame of lstm is used
        logits = tf.nn.bias_add(tf.matmul(lstm[:, -1, :], W), b)

        # Predictor
        self.f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)

        # Accuracy
        self.accuracy = tf.metrics.accuracy(tf.argmax(self.y, 1), tf.argmax(self.f, 1))

        self.conf_matrix = tf.confusion_matrix(tf.argmax(self.y, 1), tf.argmax(self.f, 1), dtype=dtype)


def run_epoch(session, model, minimize_operation, batch_size, sample_size, x_train, y_train, count, prnt=True):
    t = time.time()
    mx = batch_size * ((len(x_train) - sample_size) // batch_size)
    zero_state = session.run(model.in_state, {model.batch_size: batch_size})
    for i in range(0, mx, batch_size):
        sample = [x_train[i + j:i + j + sample_size + 1] for j in range(batch_size)]
        sample_y = [y_train[i + j + sample_size] for j in range(batch_size)]
        # print(sample[1][-1], sample_y[0])  # should be same!
        session.run(minimize_operation,
                    {model.batch_size: batch_size,
                     model.x: sample,
                     model.y: sample_y,
                     model.in_state: zero_state})

        if prnt:
            print("i:", i, ", loss", session.run(model.loss,
                                                 {model.batch_size: batch_size,
                                                  model.x: sample,
                                                  model.y: sample_y,
                                                  model.in_state: zero_state}))
    bs = len(x_train) - mx - sample_size - 1
    sample = [x_train[mx + j:mx + j + sample_size + 1] for j in range(bs)]
    sample_y = [y_train[mx + j + sample_size] for j in range(bs)]
    zero_state = session.run(model.in_state, {model.batch_size: bs})
    # print(sample[1][-1], sample_y[0])  # should be same!
    session.run(minimize_operation,
                {model.batch_size: bs,
                 model.x: sample,
                 model.y: sample_y,
                 model.in_state: zero_state})
    if prnt:
        print("i:", mx, ", loss", session.run(model.loss,
                                              {model.batch_size: batch_size,
                                               model.x: sample,
                                               model.y: sample_y,
                                               model.in_state: zero_state}))
        print("epoch %.d, time: %.d" % (count, time.time() - t))


def test_model(saver, session, model, sample_size, x_test, y_test, filepath, batches, prnt=True):
    saver.restore(session, filepath)
    session.run(tf.local_variables_initializer())

    # Size of each batch
    size = len(x_test) // batches - sample_size - 1
    zero_state = session.run(model.in_state, {model.batch_size: size})

    sample_x = [x_test[j:j + sample_size + 1] for j in range(size)]
    sample_y = [y_test[j + sample_size] for j in range(size)]

    acc = session.run(model.accuracy,
                      {model.batch_size: size,
                       model.x: sample_x,
                       model.y: sample_y,
                       model.in_state: zero_state})[1]

    confm = session.run(model.conf_matrix,
                        {model.batch_size: size,
                         model.x: sample_x,
                         model.y: sample_y,
                         model.in_state: zero_state})

    for k in range(1, batches):
        mn = k * len(x_test) // batches
        mx = mn + size
        sample_x = [x_test[j:j + sample_size + 1] for j in range(mn, mx)]
        sample_y = [y_test[j + sample_size] for j in range(mn, mx)]

        acc += session.run(model.accuracy,
                           {model.batch_size: size,
                            model.x: sample_x,
                            model.y: sample_y,
                            model.in_state: zero_state})[1]

        confm += session.run(model.conf_matrix,
                             {model.batch_size: size,
                              model.x: sample_x,
                              model.y: sample_y,
                              model.in_state: zero_state})

    if prnt:
        print("accuracy", acc / batches)
        print("conf matrix", confm)

    return acc / batches, confm
