import pandas
import tensorflow as tf

from example.simple_bitcoin_predictor import SimpleBitcoinPredictor
from util.util import find_increase

data = pandas.read_csv("../resources/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv", dtype='float64')

x = data.drop("Timestamp", 1)
# x = data.get_values().tolist()
# for r in x:
#     r.pop(0)

x = x.tail(len(x) - 2625376)  # start of 2017

# Normalize data
maxes = x.max()
x = x / maxes

x = x.values.tolist()

cutoff = round(len(x) * 0.8)  # 80% training and 20% test data
x_train = x[cutoff:]

y = find_increase(x, -1)
y_train = y[cutoff:]

encodings_size = len(x_train[1])
alphabet_size = len(y[1])

model = SimpleBitcoinPredictor(encodings_size, alphabet_size)

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.05).minimize(model.loss)

sample_size = 10000
batch_size = 100

saver = tf.train.Saver()

# Create session for running TensorFlow operations
with tf.Session() as session:
    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    # Initialize model.in_state
    zero_state = session.run(model.in_state, {model.batch_size: batch_size})

    for epoch in range(1):
        for i in range(0, batch_size * (len(x_train) - sample_size) // batch_size, batch_size):
            sample = [x_train[i + j:i + j + sample_size + 1] for j in range(batch_size)]
            sample_y = [y_train[i + j] for j in range(batch_size)]
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

    save_path = saver.save(session, "tmp/model.ckpt")
    session.close()
