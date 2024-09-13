# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

Digit categorization of scanned handwriting images, together with answer verification. There are a number of handwritten digits in the MNIST dataset. The assignment is to place a handwritten digit picture into one of ten classes that correspond to integer values from 0 to 9, inclusively. The dataset consists of 60,000 handwritten digits that are each 28 by 28 pixels in size. In this case, we construct a convolutional neural network model that can categorise to the relevant numerical value.

## Neural Network Model

![Screenshot 2024-09-13 120308](https://github.com/user-attachments/assets/9c311350-c3cd-4808-8952-7f861e3fa3af)

## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries.

### STEP 2:
Download and load the dataset
### STEP 3:
Scale the dataset between it's min and max values
### STEP 4:
Using one hot encode, encode the categorical values
### STEP 5:
Split the data into train and test
### STEP 6:
Build the convolutional neural network model
### STEP 7:
Train the model with the training data
### STEP 8:
Plot the performance plot
### STEP 9:
Evaluate the model with the testing data
### STEP 10:
Fit the model and predict the single input


## PROGRAM

### Name: ANUSHARON.S
### Register Number:212222240010


```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[5]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()
X_train_scaled.max()
y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)
y_train_onehot.shape

single_image = X_train[700]
plt.imshow(single_image,cmap='gray')

y_train_onehot[700]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train_onehot, epochs=7, 
          batch_size=64, 
          validation_data=(X_test_scaled, y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('/content/Screenshot 2024-09-13 110948.png')

type(img)


img = image.load_img('imagefive.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-09-13 115041](https://github.com/user-attachments/assets/c608f079-2a7e-4c48-9414-82309114f62f)


![Screenshot 2024-09-13 115125](https://github.com/user-attachments/assets/d86b5883-e95e-4026-8072-8186fb89e1e6)


![Screenshot 2024-09-13 115236](https://github.com/user-attachments/assets/f3091de6-ef13-47d1-b931-4d977fbb64db)
### Classification Report

![Screenshot 2024-09-13 115329](https://github.com/user-attachments/assets/cf0fbcc5-9d24-4a75-8562-d9d6381174b9)

### Confusion Matrix
![Screenshot 2024-09-13 115538](https://github.com/user-attachments/assets/bda6e505-e3dd-49ba-99c9-1430c76b575b)



### New Sample Data Prediction

![Screenshot 2024-09-13 115624](https://github.com/user-attachments/assets/1ea00349-5ae1-42af-b8b8-73f0c3c1ae33)

![Screenshot 2024-09-13 115637](https://github.com/user-attachments/assets/ed9202e0-8af2-447d-bdae-926b8c24b30f)

![Screenshot 2024-09-13 115647](https://github.com/user-attachments/assets/2fda71f4-c33c-488e-8241-5afa33d0ecad)


## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
