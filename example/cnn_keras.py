import keras
import pandas
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, LeakyReLU, BatchNormalization, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Nadam
from util.util import find_increase, generate_classes
import numpy as np

# Eksempelkode fra https://towardsdatascience.com/build-your-own-convolution-neural-network-in-5-mins-4217c2cf964f

from util.util import find_increase
#Read data from csv
data = pandas.read_csv("../resources/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv", dtype='float64')
#
x = data.drop("Timestamp", 1)

x = x.tail(len(x) - 2625376)
maxes = x.max()
x = x / maxes
x = np.expand_dims(x, axis=2)
# Normalize data

cutoff = round(len(x) * 0.8)  # 80% training and 20% test data
x_train = x[:cutoff]
x_test = x[cutoff:-1]

y = find_increase(x, -1)
y = generate_classes(y, 5)
y = np.array(y)
print(y.shape)
print(y)
#y = np.squeeze(y, axis=1)
y_train = y[:cutoff]
#y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = y[cutoff:]
batch_size = 128
num_classes = 10
epochs = 12

# Data dimnesions
data_rows, data_cols = 2724686, 8

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Sequential model from https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-2-1-multivariate-time-series-ab016ce70f57
model = Sequential()

model.add(Conv1D(input_shape=(7, 1),
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
model.add(Dense(5))
model.add(Activation('softmax'))

# # Compile model
opt = Nadam(lr=0.002)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=30, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="model.hdf5", verbose=1, save_best_only=True)

history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=128,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[reduce_lr, checkpointer])
# removed shuffle=true
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
