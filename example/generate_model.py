import numpy
import pandas
import tensorflow as tf
import time

from example.simple_bitcoin_predictor import SimpleBitcoinPredictor
from util.util import find_increase, generate_classes, get_data, get_full_data

sample_size = 24 * 30
batch_size = 2000
num_classes = 3
num_features = 43

x, y = get_full_data(num_classes)
cutoff = round(len(x) * 0.8)  # 80% training and 20% test data
x_train = x[:cutoff]
y_train = y[:cutoff]

model = SimpleBitcoinPredictor(num_features, num_classes)

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.001).minimize(model.loss)

saver = tf.train.Saver()

# Create session for running TensorFlow operations
with tf.Session() as session:
    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    # Initialize model.in_state
    zero_state = session.run(model.in_state, {model.batch_size: batch_size})

    for epoch in range(100):
        t = time.time()
        for i in range(0, batch_size * ((len(x_train) - sample_size) // batch_size), batch_size):
            sample = [x_train[i + j:i + j + sample_size + 1] for j in range(batch_size)]
            sample_y = [y_train[i + j + sample_size] for j in range(batch_size)]
            # print(sample[1][-1], sample_y[0])  # should be same!
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
        print("epoch %.d, time: %.d" % (epoch, time.time() - t))

    save_path = saver.save(session, "tmp/model.ckpt")
    session.close()
