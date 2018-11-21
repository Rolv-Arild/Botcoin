import tensorflow as tf
import matplotlib.pyplot as plt

from src.simple_bitcoin_predictor import SimpleBitcoinPredictor, run_epoch, test_model
from src.util.util import get_reduced_data

sample_size = 24 * 30
batch_size = 2000
num_classes = 3
num_features = 3

x, y, data = get_reduced_data(num_classes, 60)
cutoff = round(len(x) * 0.8)  # 80% training and 20% test data
x_train = x[:cutoff]
y_train = y[:cutoff]
x_test = x[cutoff:]
y_test = y[cutoff:]

model = SimpleBitcoinPredictor(num_features, num_classes)

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.01).minimize(model.loss)

saver = tf.train.Saver()

acc = []
# Create session for running TensorFlow operations
with tf.Session() as session:
    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    for epoch in range(100):
        run_epoch(session, model, minimize_operation, batch_size, sample_size, x_train, y_train, epoch)
        save_path = saver.save(session, "tmp/lstm-model-reduced-1hour.ckpt")
        acc.append(test_model(saver, session, model, sample_size, x_test, y_test, save_path, 1)[0])

    session.close()

plt.plot(acc)
plt.title("LSTM accuracy (1 hour reduced dataset)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
