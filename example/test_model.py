from example.simple_bitcoin_predictor import SimpleBitcoinPredictor
import tensorflow as tf

model = SimpleBitcoinPredictor(7, 1)
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "tmp/model.ckpt")
    print(sess.run(model.f, {model.batch_size: 1, model.x: [[[1, 1, 1, 1, 1, 1, 1]]]}))

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
