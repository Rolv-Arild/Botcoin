from example.simple_bitcoin_predictor import SimpleBitcoinPredictor, test_model
import tensorflow as tf

from util.util import get_data

# batch_size = 2000
num_classes = 3
num_features = 3

x, y = get_data(num_classes)
cutoff = round(len(x) * 0.8)  # 80% training and 20% test data
x_test = x[cutoff:]
y_test = y[cutoff:]

model = SimpleBitcoinPredictor(num_features, num_classes)

sample_size = 4 * 24 * 30

saver = tf.train.Saver()

with tf.Session() as session:
    test_model(saver, session, model, sample_size, x_test, y_test, "tmp/lstm-model-hourly.ckpt", 2)
