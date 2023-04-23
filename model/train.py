import tensorflow as tf
import numpy as np
import os
import pickle as pkl
from tensorflow import keras
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt


def draw(val, history, loss):
    plt.clf()
    plt.plot(history.history[val])
    plt.plot(history.history['val_' + val])
    plt.title('Model ' + val + " " + loss)
    plt.ylabel(val)
    plt.xlabel('Epoch')
    plt.legend(["Train", "Test"], loc = "best")
    plt.savefig(val + "_" + loss + ".png")


if __name__ == "__main__":
    with open('./dataset_train_onehot_without_position.pkl', "rb") as f:
        [X_train, y_train] = pkl.load(f)
    with open('./dataset_test_onehot_without_position.pkl', "rb") as f:
        [X_test, y_test] = pkl.load(f)
    print(X_train.shape)
    print(y_train.shape)
    print(y_train.shape)
    print(y_test.shape)

    model = keras.Sequential()
    model.add(layers.LSTM(units = 128, input_shape = (X_train.shape[1], X_train.shape[2])))
    model.add(layers.Dense(units = y_test.shape[1], activation = 'softmax'))
    print("model constructed")
    loss = 'mean_squared_error'
    model.compile(optimizer = 'adam', loss = loss,
                  metrics = [tf.keras.metrics.TopKCategoricalAccuracy(k = 1, name = "top-1"),
                             tf.keras.metrics.TopKCategoricalAccuracy(k = 3, name = "top-3"),
                             tf.keras.metrics.TopKCategoricalAccuracy(k = 5, name = "top-5"),
                             tf.keras.metrics.TopKCategoricalAccuracy(k = 10, name = "top-10")])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath = loss+"_onehot_without_position_{epoch}",
            save_best_only = True,  # Only save a model if `val_loss` has improved.
            monitor = "val_loss",
            verbose = 1,
        )
    ]
    history = model.fit(X_train, y_train, epochs = 15, batch_size = 256, callbacks = callbacks,
                        validation_split = 0.1)

    test_loss, top1, top3, top5, top10 = model.evaluate(X_test, y_test)
    print('Test loss:', test_loss)
    print('Test accuracy:', top1, top3, top5, top10)

    draw('loss', history, loss+"_onehot_without_position")
    draw('top-1', history, loss+"_onehot_without_position")
    draw('top-3', history, loss+"_onehot_without_position")
    draw('top-5', history, loss+"_onehot_without_position")
    draw('top-10', history, loss+"_onehot_without_position")
