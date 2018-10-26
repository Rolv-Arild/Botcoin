import pandas

from example.simple_bitcoin_predictor import SimpleBitcoinPredictor
import tensorflow as tf

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
x_test = x[:cutoff:]

y = find_increase(x, -1)
y_test = y[:cutoff]

encodings_size = len(x_test[1])
alphabet_size = len(y[1])

model = SimpleBitcoinPredictor(encodings_size, alphabet_size)

sample_size = 1800
batch_size = 1000

saver = tf.train.Saver()

with tf.Session() as session:
    saver.restore(session, "tmp/model.ckpt")

    zero_state = session.run(model.in_state, {model.batch_size: batch_size})

    # Test model on test data
    for i in range(0, batch_size * ((len(x_test) - sample_size) // batch_size), batch_size):
        sample = [x_test[i + j:i + j + sample_size + 1] for j in range(batch_size)]
        sample_y = [y_test[i + j] for j in range(batch_size)]

        print("accuracy", session.run(model.accuracy,
                                      {model.batch_size: batch_size,
                                       model.x: sample,
                                       model.y: sample_y,
                                       model.in_state: zero_state}))

# Test model on test data
# for i in range(0, (len(x_test) - sample_size) // batch_size, batch_size):
#     sample = [x_test[i + j:i + j + sample_size + 1] for j in range(batch_size)]
#     sample_y = [y_test[i + j] for j in range(batch_size)]
#
#     print("accuracy", session.run(model.accuracy,
#                                   {model.batch_size: batch_size,
#                                    model.x: sample,
#                                    model.y: sample_y,
#                                    model.in_state: zero_state}))
