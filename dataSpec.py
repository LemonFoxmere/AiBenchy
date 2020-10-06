from keras.utils import to_categorical
from mlxtend.data import loadlocal_mnist
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import time
import random

# load in data
X, y = loadlocal_mnist( # training set
  images_path='train-images-idx3-ubyte',
  labels_path='train-labels-idx1-ubyte')

# print("training image tensor rank =", X[0].ndim)
# print("training input shape =", X.shape)
# plt.matshow(image1)

fig, ax = plt.subplots(3,3)

plt.set_cmap('gray')

for i in ax.flat:
    dataID = random.randint(0,6000)
    im = i.imshow(np.reshape(X[dataID], (28, 28)))
    i.axis('off')
    title = 'key=' + str(y[dataID])
    i.title.set_text(title)

cb_ax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)

fig.suptitle('Data Preview', fontsize=16)

plt.subplots_adjust(wspace=0.25, hspace=0.25)

plt.show()

enc_y = to_categorical(y)

tX, ty = loadlocal_mnist( # test set
  images_path='t10k-images-idx3-ubyte',
  labels_path='t10k-labels-idx1-ubyte')

enc_ty = to_categorical(ty)