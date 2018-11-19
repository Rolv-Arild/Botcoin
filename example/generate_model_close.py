import numpy
import pandas
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from example.simple_bitcoin_predictor import SimpleBitcoinPredictor, run_epoch, test_model
from util.util import find_increase, generate_classes, get_data, get_full_data, plot_prediction

sample_size = 24 * 30
batch_size = 2000
num_classes = 3
num_features = 3

x, y = get_data(num_classes, 60)

model = SimpleBitcoinPredictor(num_features, num_classes)

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.001).minimize(model.loss)

saver = tf.train.Saver()

acc = []
# Create session for running TensorFlow operations
with tf.Session() as session:
    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    for epoch in range(10):
        run_epoch(session, model, minimize_operation, batch_size, sample_size, x, y, epoch)

    save_path = saver.save(session, "tmp/lstm-model-close.ckpt")

    plot_prediction(session, model, x)

    session.close()
