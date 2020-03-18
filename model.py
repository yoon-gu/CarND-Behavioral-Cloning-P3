from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, Conv2D
import pandas as pd
from pprint import pprint
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.inception_v3 import InceptionV3
import keras.backend as K


df1 = pd.read_csv('./normal_lap/driving_log.csv', header=None)
df2 = pd.read_csv('./backward_lap/driving_log.csv', header=None)
print(len(df1))
print(len(df2))
df = pd.concat((df1, df2))


center_img_paths = df[0].values.tolist() + df[1].values.tolist() + df[2].values.tolist()
x_train = np.array([mpimg.imread(img_path.strip())
                    for img_path in center_img_paths])
y_train = np.array(df[3].values.tolist()
                   + (df[3].values + 0.15).tolist()
                   + (df[3].values - 0.15).tolist())


x_train_flip = np.array([np.fliplr(img) for img in x_train])
y_train_flip = -y_train.copy()

x_train = np.concatenate((x_train, x_train_flip))
y_train = np.concatenate((y_train, y_train_flip))


model = Sequential()
# Preprocessing - Crop some top and bottom pixels and normalization.
model.add(Cropping2D(cropping=((65, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# NVIDIA's architecture
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


# Imports the Model API

# Compile the model
model.compile(optimizer='adam', loss='mse')

hist = model.fit(x_train, y_train, validation_split=0.2,
                 shuffle=True, batch_size=32, epochs=3)


model.save('model.h5')
