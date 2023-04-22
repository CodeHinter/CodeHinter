import tensorflow as tf
import numpy as np
import os
import pickle as pkl
from tensorflow import keras
from tensorflow.keras import layers
import random

if __name__ == "__main__":
    with open('./dataset_train.pkl', "rb") as f:
        [X_train, y_train] = pkl.load(f)
    with open('./dataset_test.pkl', "rb") as f:
        [X_test, y_test] = pkl.load(f)
    input_dimension = X_train[1].shape[1]
    print(X_train.shape)
    print(y_train.shape)
    print(y_train.shape)
    print(y_test.shape)

    model = keras.Sequential()
    model.add(layers.LSTM(input_dimension))
    print("model constructed")

    model.compile(optimizer = 'adam', loss = 'mean_squared_error',
                  metrics = [tf.keras.metrics.TopKCategoricalAccuracy(k = 1, name = "top-1"),
                             tf.keras.metrics.TopKCategoricalAccuracy(k = 3, name = "top-3"),
                             tf.keras.metrics.TopKCategoricalAccuracy(k = 5, name = "top-5"),
                             tf.keras.metrics.TopKCategoricalAccuracy(k = 10, name = "top-10")])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath = "mymodel_{epoch}",
            save_best_only = True,  # Only save a model if `val_loss` has improved.
            monitor = "val_loss",
            verbose = 1,
        )
    ]
    history = model.fit(X_train, y_train, epochs = 25, batch_size = 32, callbacks = callbacks,
                        validation_split = 0.1)

    test_loss, top1, top3, top5, top10 = model.evaluate(X_test, y_test)
    print('Test loss:', test_loss)
    print('Test accuracy:', top1, top3, top5, top10)
