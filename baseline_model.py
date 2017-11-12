# based on https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d/notebook
# the result is not deterministic, score on test (cross-val) set is from 0.22 to 0.27

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join as opj
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from sklearn.preprocessing import StandardScaler
from os.path import isfile
import data


def convert_train_to_numpy():
    train = pd.read_json("data/train.json")
    # Generate the training data
    # Create 3 bands having HH, HV and avg of both
    X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
    X_band_3 = (X_band_1 + X_band_2) / 2
    # Scaling should be either removed, either saved to apply the same scaling to test data.
    scaler_1 = StandardScaler()
    scaler_2 = StandardScaler()
    scaler_3 = StandardScaler()
    scaler_1 = scaler_1.fit(X_band_1.reshape(-1, 1))
    scaler_2 = scaler_2.fit(X_band_2.reshape(-1, 1))
    scaler_3 = scaler_3.fit(X_band_3.reshape(-1, 1))
    X_band_1 = scaler_1.transform(X_band_1.reshape(-1, 1)).reshape((X_band_1.shape[0], 75, 75))
    X_band_2 = scaler_2.transform(X_band_2.reshape(-1, 1)).reshape((X_band_2.shape[0], 75, 75))
    X_band_3 = scaler_3.transform(X_band_3.reshape(-1, 1)).reshape((X_band_3.shape[0], 75, 75))
    X_train = np.concatenate([X_band_1[:, :, :, np.newaxis],
                              X_band_2[:, :, :, np.newaxis],
                              X_band_3[:, :, :, np.newaxis]],
                             axis=-1)
    y_train = train['is_iceberg']
    np.save("data/train.npy", np.hstack((X_train.reshape((X_train.shape[0], -1)), y_train.values.reshape(-1, 1))))


def getModel():
    #Building the model
    gmodel=Sequential()
    #Conv Layer 1
    gmodel.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Conv Layer 2
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Conv Layer 3
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Conv Layer 4
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Flatten the data for upcoming dense layers
    gmodel.add(Flatten())

    #Dense Layers
    gmodel.add(Dense(256))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    #Dense Layer 2
    gmodel.add(Dense(128))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    #Sigmoid Layer
    gmodel.add(Dense(1))
    gmodel.add(Activation('sigmoid'))

    mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss='binary_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    gmodel.summary()
    return gmodel


def get_callbacks(filepath, patience=5):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def train():
    data_train = np.load("data/train.npy")
    X_train, y_train = np.hsplit(data_train, [data_train.shape[1] - 1])
    X_train = X_train.reshape(-1, 75, 75, 3)

    file_path = ".model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=10)

    X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, y_train, random_state=16, train_size=0.75)

    #Without denoising, core features.
    gmodel=getModel()
    gmodel.fit(X_train_cv, y_train_cv,
              batch_size=128,
              epochs=100,
              verbose=1,
              validation_data=(X_valid, y_valid),
              callbacks=callbacks)

    gmodel.load_weights(filepath=file_path)
    score = gmodel.evaluate(X_valid, y_valid, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
