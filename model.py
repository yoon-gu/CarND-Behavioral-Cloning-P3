from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, Conv2D
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
import sklearn
import keras


lines = []
data_path = './data/'
with open(os.path.join(data_path, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


center_images = []
left_images = []
right_images = []
angles = []

for line in lines:
    center_img = line[0]
    left_img = line[1].strip()
    right_img = line[2].strip()

    center_images.append(center_img)
    left_images.append(left_img)
    right_images.append(right_img)
    angles.append(np.float(line[3]))


correction = 0.2

total_images = []
total_angles = []

total_images.extend(center_images)
total_images.extend(left_images)
total_images.extend(right_images)

total_angles.extend(angles)
total_angles.extend([angle + correction for angle in angles])
total_angles.extend([angle - correction for angle in angles])


samples = list(zip(total_images, total_angles))

# Randomly split data into training and validation.
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Generator function
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = os.path.join(data_path, 'IMG/') + \
                    batch_sample[0].split('/')[-1]
                image = mpimg.imread(name)
                angle = float(batch_sample[1])
                images.append(image)
                angles.append(angle)
                images.append(np.fliplr(image))
                angles.append(-angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


b_size = 32
train_generator = generator(train_samples, batch_size=b_size)
validation_generator = generator(validation_samples, batch_size=b_size)


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
model.add(Dropout(0.01))
model.add(Dense(50))
model.add(Dropout(0.01))
model.add(Dense(10))
model.add(Dropout(0.01))
model.add(Dense(1))


model.summary()


# Compile the model
model.compile(optimizer='adam', loss='mse')


filepath = "saved-model-{epoch:02d}.h5"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, verbose=1, save_best_only=False, mode='max')


hist = model.fit_generator(train_generator, steps_per_epoch=len(train_samples) // b_size,
                           validation_data=validation_generator,
                           validation_steps=len(validation_samples) // b_size,
                           epochs=5, callbacks=[checkpoint])


model.save('model.h5')


# print the keys contained in the history object
print(hist.history.keys())

# plot the training and validation loss for each epoch
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('examples/figure.png', dpi=300)
