import sys
import h5py
import numpy as np
from matplotlib import image

data_filename = str(sys.argv[1])

images = h5py.File(data_filename, 'r')
x_data = np.array(images['data'])
x_data = x_data.transpose((0, 2, 3, 1))
print(x_data.shape)
x_data = x_data/255

for i in range(210, 220):
    image.imsave(str(i)+'.jpeg', x_data[i].reshape((55, 47, 3)))

