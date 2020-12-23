import keras
import sys
import h5py
import numpy as np
import tensorflow as tf
from matplotlib import image

data_filename = str(sys.argv[1])


def data_loader(filepath):
    data = image.imread(filepath)
    # print(data.shape)
    x_data = np.array([data[:, :, 0:3]])
    # print(x_data)
    # y_data = np.array(data['label'])
    # x_data = x_data.transpose((0, 2, 3, 1))

    return x_data


def data_preprocess(x_data):
    return x_data / 255


def main():
    x_test = data_loader(data_filename)
    # x_test = data_preprocess(x_test)

    model = keras.models.load_model('models/goodNet_anonymous_1.h5')
    model.load_weights('models/goodNet_anonymous_1_weights.h5')

    clean_label_p = np.argmax(model.predict(x_test), axis=1)
    print('Classification result:', clean_label_p[0])


if __name__ == '__main__':
    main()
