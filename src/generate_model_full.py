import tensorflow as tf

from src.simple_bitcoin_predictor import SimpleBitcoinPredictor, run_epoch
from src.util.util import get_full_data

sample_size = 24 * 30
batch_size = 2000
num_classes = 3
num_features = 42

x, y, data = get_full_data(num_classes, 60)

model = SimpleBitcoinPredictor(num_features, num_classes)

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.001).minimize(model.loss)

saver = tf.train.Saver()

acc = []
# Create session for running TensorFlow operations
with tf.Session() as session:
    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    for epoch in range(100):
        run_epoch(session, model, minimize_operation, batch_size, sample_size, x, y, epoch)

    save_path = saver.save(session, "../resources/tmp/lstm-model-close.ckpt")

    session.close()
