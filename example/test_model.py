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
