import keras
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

# Eksempelkode fra https://towardsdatascience.com/build-your-own-convolution-neural-network-in-5-mins-4217c2cf964f
from util.util import find_increase

data = pandas.read_csv("../resources/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv", dtype='float64').values
# x = data.drop("Timestamp", 1)
x = np.expand_dims(data, axis=2)
# Normalize data
maxes = x.max()
x = x / maxes

# x = x.values.tolist()

cutoff = round(len(x) * 0.8)  # 80% training and 20% test data
x_train = x[:cutoff]
x_test = x[cutoff:]

y = find_increase(x, -1)
y = np.array(y)
print(y.shape)
y = np.squeeze(y, axis=1)
print("after", y.shape)
y_train = y[:cutoff]
y_test = y[cutoff:]
batch_size = 128
num_classes = 10
epochs = 12

encodings_size = len(x_train[1])
alphabet_size = len(y[1])

# x = data.get_values().tolist()
# for r in x:
#     r.pop(0)

# x = x.tail(len(x)-2625376)  # start of 2017

# input image dimensions
data_rows, data_cols = 2724686, 8

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=5, input_shape=(data_rows, data_cols), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(x_train.shape)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
