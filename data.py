import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
import zipfile


def read_data(file_path):
    """Reads csv file into a list"""
    lines = []
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    return lines


def flip_image(image):
    """Flip over y-axis"""
    return cv2.flip(image, 1)
    

def process_image(image):
    """Crop top part of image, resize to 200 x 66 and convert to YUV"""
    image = image[60:, :, :]
    image = cv2.resize(image, (200, 66))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


def get_images(data):
    """Loads data from csv into a dictionary of training data"""
    images = []
    measurements = []

    # For straight on camera view, left camera, right camera
    ANGLE_CORRECTIONS = (0, 0.15, -0.15)

    for i, line in enumerate(data):
        # Skip first line which is the header
        if i == 0:
            continue

        measurement = float(line[3])

        # Ignore steering wheel angles of zero, they don't
        # help in training
        if measurement != 0:
            for j, cam_pos in enumerate(ANGLE_CORRECTIONS):
                measurement += cam_pos
                source_path = line[j]
                file_name = source_path.split('/')[-1]
                current_path = "input/data/IMG/" + file_name

                measurements.append(measurement)
                measurements.append(measurement*-1)

                image = mpimg.imread(current_path)
                image = process_image(image)
                images.append(image)
                images.append(flip_image(image))


    X_train = np.array(images)
    y_train = np.array(measurements)

    return {'X_train': X_train, 'y_train': y_train}


if __name__ == "__main__":
    csv_file = "input/data/driving_log.csv"
    driving_data = read_data(csv_file)
    images = get_images(driving_data)
