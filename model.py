from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
import zipfile
import os

import data


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
    csv_file = "/input/data/driving_log.csv"
    driving_data = data.read_data(csv_file)
    image_data = data.get_images(driving_data)
    run_model(image_data)


if __name__ == "__main__":
    main()