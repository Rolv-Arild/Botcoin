import pandas
import tensorflow as tf
import time

from example.simple_bitcoin_predictor import SimpleBitcoinPredictor
from util.util import find_increase, generate_classes, get_data

sample_size = 24 * 30
batch_size = 100

x, y = get_data()
cutoff = round(sample_size * 0.8)  # 80% training and 20% test data
x_split = [x[i:i + sample_size] for i in range(0, 18 * sample_size, sample_size)]  # split into 30 day intervals
y_split = [y[i:i + sample_size] for i in range(0, 18 * sample_size, sample_size)]  # split into 30 day intervals

x_train = [xi[:cutoff] for xi in x_split]
x_test = [xi[cutoff:] for xi in x_split]

y_train = [yi[:cutoff] for yi in y_split]
y_test = [yi[cutoff:] for yi in y_split]

num_features = len(x_train[0][0])
alphabet_size = len(y_train[0][0])

model = SimpleBitcoinPredictor(num_features, alphabet_size)

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.05).minimize(model.loss)

# Create session for running TensorFlow operations
with tf.Session() as session:
    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    # Initialize model.in_state
    zero_state = session.run(model.in_state, {model.batch_size: batch_size})

    for epoch in range(10):
        t = time.time()
        for k in range(0, 18 * sample_size, sample_size):
            for i in range(0, batch_size * (cutoff // batch_size), batch_size):
                sample_x = [x[k + i + j:k + i + j + sample_size] for j in range(batch_size)]
                sample_y = [y[k + i + j + sample_size] for j in range(batch_size)]

                session.run(minimize_operation,
                            {model.batch_size: batch_size,
                             model.x: sample_x,
                             model.y: sample_y,
                             model.in_state: zero_state})

                print("i:", i, ", loss", session.run(model.loss,
                                                     {model.batch_size: batch_size,
                                                      model.x: sample_x,
                                                      model.y: sample_y,
                                                      model.in_state: zero_state}))

        sample_x = [x[j:j + sample_size] for j in range(len(x_test)-sample_size)]
        sample_y = [y[j + sample_size] for j in range(len(x_test)-sample_size)]

        print("accuracy", session.run(model.accuracy,
                                      {model.batch_size: len(x_test)-sample_size,
                                       model.x: x_test,
                                       model.y: y_test,
                                       model.in_state: zero_state}))

        print("conf matrix", session.run(model.conf_matrix,
                                         {model.batch_size: len(x_test)-sample_size,
                                          model.x: x_test,
                                          model.y: y_test,
                                          model.in_state: zero_state}))

        print("epoch %.d, time: %.d" % (epoch, time.time() - t))

