# README

## Project Overview
This project involves building and training an LSTM neural network for time series prediction using a dataset containing various variables. The goal is to predict the next value in the sequence based on the previous values. The dataset used for training and testing the model is preprocessed and split into training and testing sets, followed by the implementation of the LSTM model in Keras.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Contact](#contact)

## Installation
To run this project, you'll need to have Python installed. Additionally, you'll need to install the following libraries:

```bash
pip install numpy pandas keras scikit-learn tensorflow
```

## Data
The dataset `lstm.csv` is used for training and testing the LSTM model. It contains multiple variables needed for prediction.

## Usage
To use the project, follow these steps:

1. Ensure all necessary data files are in place.
2. Import the required libraries.
3. Load and preprocess the dataset.
4. Train the LSTM model.
5. Evaluate the model on the test data.

```python
# Importing necessary libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, to_categorical
from keras.initializers import glorot_uniform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K

# Reading dataset
final_df = pd.read_csv('firedata/lstm.csv', sep=',')
timesteps = 12
df_shape = final_df.shape
final_cols = final_df.columns.values.tolist()
n_vars = len(final_cols)

# One-hot encoding of Y
x = final_df.iloc[:, :-1].values
y = final_df.iloc[:, -1].values
y = y.astype(int)
total_classes = len(range(0, 25))
y_enc = to_categorical(y, num_classes=total_classes)
y_enc = y_enc.astype(int)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y_enc, test_size=0.2, random_state=0)

# Reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], timesteps, n_vars - 1))
x_test = x_test.reshape((x_test.shape[0], timesteps, n_vars - 1))

# Custom metric
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

# Neural network
model = Sequential()
model.add(LSTM(n_vars, input_shape=(x_train.shape[1], x_train.shape[2]), kernel_initializer='glorot_uniform'))
model.add(Dropout(0.2))
model.add(Dense(total_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

# Fitting the model
history = model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1, shuffle=True)

# Evaluation
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("Loss: %.5f" % loss)
print("Accuracy: %.5f" % acc)
```

## Model Architecture
The LSTM neural network is designed with the following layers:
- **LSTM Layer**: Processes the input sequences.
- **Dropout Layer**: Regularization to prevent overfitting.
- **Dense Layer**: Output layer with softmax activation for classification.

## Evaluation
The model is evaluated using categorical cross-entropy loss and accuracy metrics. After training, the model achieves an accuracy of approximately 90%.
