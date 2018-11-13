import pandas

from example.simple_bitcoin_predictor import SimpleBitcoinPredictor
import tensorflow as tf

from util.util import find_increase, generate_classes, get_data, get_full_data

batch_size = 100
num_classes = 3
num_features = 43

x, y = get_full_data(num_classes)
cutoff = round(len(x) * 0.8)  # 80% training and 20% test data
x_test = x[cutoff:]
y_test = y[cutoff:]

model = SimpleBitcoinPredictor(num_features, num_classes)

sample_size = len(y_test) - batch_size

saver = tf.train.Saver()

with tf.Session() as session:
    saver.restore(session, "tmp/model.ckpt")
    session.run(tf.local_variables_initializer())

    zero_state = session.run(model.in_state, {model.batch_size: batch_size})

    # Test model on test data
    sample_x = [x_test[j:j + sample_size] for j in range(len(x_test) - sample_size - 1)]
    sample_y = [y_test[j + sample_size] for j in range(len(x_test) - sample_size - 1)]

    print("accuracy", session.run(model.accuracy,
                                  {model.batch_size: len(x_test) - sample_size,
                                   model.x: sample_x,
                                   model.y: sample_y,
                                   model.in_state: zero_state}))

    print("conf matrix", session.run(model.conf_matrix,
                                     {model.batch_size: len(x_test) - sample_size,
                                      model.x: sample_x,
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
