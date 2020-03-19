# **Behavioral Cloning**

## Writeup

[//]: # (Image References)

[image1]: ./examples/figure.png "Model Visualization"
[image2]: ./examples/data_augmentation.png "Data Augmentation"

### Files Submitted

My project includes the following files:

* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `video.mp4` a video recording of vehicle driving autonomously around the track.
* `README.md` summarizing the results


**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Model Architecture and Training Strategy
#### Model Architecture

The final model that I utilized is the NVIDIA's architecture and it was enough to obtain a good driving neural network. It consists of five convolution layers and 3 dense layers. And, I did add preprocess blocks (cropping and normalizing) before NVIDIA's network. I easily defined this model using `keras`:

```python
model = Sequential()
# Preprocessing - Crop some top and bottom pixels and normalization.
model.add(Cropping2D(cropping=((65,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# NVIDIA's architecture
model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

As you check in the following, I estimated `559,419` parameters to achieve the goal of this project.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
cropping2d_1 (Cropping2D)    (None, 70, 320, 3)        0
_________________________________________________________________
lambda_1 (Lambda)            (None, 70, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 33, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 4224)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               422500
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0
_________________________________________________________________
```


#### Training Strategy

My training data acquisition scheme are

- A lab center line driving in clockwise (`4408` frames)
- A lab center line driving in counter-clockwise (`4104` frames)
- Very slow control when the car is driving on curved roads in both driving cases

After train the neural network in the form of NVIDIA's architecture, I checked my performance by executing

```sh
python drive.py model.h5
```

#### Data augmentation

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving in clockwise and counter-clockwise. In addition, I augmented data by including left/right image with correction angle (`0.2`) and by flipping image frames.

![alt text][image2]


```python
df1 = pd.read_csv('./normal_lap/driving_log.csv', header=None)
df2 = pd.read_csv('./backward_lap/driving_log.csv', header=None)
df = pd.concat((df1, df2))

# Data augmentation 1
center_img_paths = df[0].values.tolist() + df[1].values.tolist() + df[2].values.tolist()
x_train = np.array([mpimg.imread(img_path.strip())
                    for img_path in center_img_paths])
y_train = np.array(df[3].values.tolist()
                   + (df[3].values + 0.2).tolist()
                   + (df[3].values - 0.2).tolist())

# Data augmentation 2
x_train_flip = np.array([np.fliplr(img) for img in x_train])
y_train_flip = -y_train.copy()

x_train = np.concatenate((x_train, x_train_flip))
y_train = np.concatenate((y_train, y_train_flip))
```

#### Parameter tuning

In the case of parameters related to the model, I used default values of NVIDIA's architecture.
To train the neural network, then I set Mean-Squared-Error(MSE) as the loss function. I used `Adam` to minimize the MSE loss function with batch size `32`. The number of epochs is `3`.

##### Creation of the Training Set

In the creation of the training,  I notice that too fast driving is harmful to the neural network's performance. The time interval of data acquisition is independent to the driving speed. Therefore I created training set by driving slowly.

Also, it was found that I had to have enough data set of the curved roads. So I drove slower in a curved lane than normal lane while I maintained the center position.

##### Training Process

While training model, I checked MSE of validation set that is 0.2 portion of all training set. The values of loss function of training and validation sets decreased so I think that it was not over-fitted.

![alt text][image1]

### Discussion

I found the performance is ridiculously poor in the second track. It needs to improve generality of the network by using adding driving data in the second track.