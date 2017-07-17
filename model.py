from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.core import Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
import zipfile
import os

import data

def nvidia_model(train):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(66, 200, 3)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))

    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(1164))

    model.add(Dropout(0.5))
    model.add(Dense(100))

    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(train['X_train'], train['y_train'], validation_split=0.2,
        shuffle=True, nb_epoch=5)

    model.save('output/nvidia_model.h5')


def run_model(train):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(train['X_train'], train['y_train'], validation_split=0.2,
        shuffle=True, nb_epoch=5)

    model.save('/output/model.h5')


def main():
    csv_file = "input/data/driving_log.csv"
    driving_data = data.read_data(csv_file)
    image_data = data.get_images(driving_data)
    #run_model(image_data)
    nvidia_model(image_data)


if __name__ == "__main__":
    main()