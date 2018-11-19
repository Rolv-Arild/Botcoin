import tensorflow as tf

from example.simple_bitcoin_predictor import SimpleBitcoinPredictor, run_epoch
from util.util import get_full_data

sample_size = 24 * 30
batch_size = 2000
num_classes = 3
num_features = 3

x, y = get_full_data(num_classes, 60)
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

    for epoch in range(100):
        run_epoch(session, model, minimize_operation, batch_size, sample_size, x_train, y_train, epoch)

    save_path = saver.save(session, "tmp/lstm-model-full.ckpt")
    session.close()
