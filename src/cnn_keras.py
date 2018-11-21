import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, BatchNormalization, Activation
from keras.layers import Conv1D
from keras.optimizers import Nadam
from src.util.util import get_full_data
import numpy as np

# Eksempelkode fra https://towardsdatascience.com/build-your-own-convolution-neural-network-in-5-mins-4217c2cf964f

batch_size = 2000
epochs = 1000
num_classes = 3

# Read data from csv
x, y, data = get_full_data(3, 60)
x = np.expand_dims(x, axis=2)

cutoff = round(len(x) * 0.8)  # 80% training and 20% test data
x_train = x[:cutoff]
x_test = x[cutoff:]

y_train = y[:cutoff]
y_test = y[cutoff:]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Sequential model from https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-2-1-multivariate-time-series-ab016ce70f57
model = Sequential()

model.add(Conv1D(input_shape=(42, 1),
                 nb_filter=16,
                 filter_length=4,
                 border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Conv1D(nb_filter=8,
                 filter_length=4,
                 border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(3))
model.add(Activation('softmax'))

# # Compile model
opt = Nadam(lr=0.0002)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.7, patience=30, min_lr=0.0001, verbose=1)
checkpointer = ModelCheckpoint(filepath="tmp/model.hdf5", verbose=1, save_best_only=True)

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[reduce_lr, checkpointer])
# removed shuffle=true

print('Test accuracy:', max(history.history['val_acc']))

#model.predict(x_test, verbose=1)

plt.plot(history.history['val_acc'])
plt.title('CNN Model accuracy (1hour full dataset)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
