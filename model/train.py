import tensorflow as tf
import numpy as np
import os
import pickle as pkl
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
import random

DATA_SET_PATH = '../javaCorpus'
FILE_NAME = 'preprocessed_train'
TEST_PATH = os.path.join(DATA_SET_PATH, FILE_NAME + '.txt')

with open(TEST_PATH) as f:
    test = f.readlines()
    test = "".join(test).split("=\n")
test_data = [data.split() for data in test if len(data) > 0]

with open(os.path.join(DATA_SET_PATH, FILE_NAME + '.pkl'), "rb") as f:
    position = pkl.load(f)
print("file loaded")
assert len(position) == len(test_data)

vob = {}
vob_src = "".join(test).split()
for word in vob_src:
    if word in vob:
        vob[word] += 1;
    else:
        vob[word] = 1
vob_map = {word: idx for idx, word in enumerate(vob.keys())}

onehot_value = [[vob_map[word] for word in one] for one in test_data]
# print(onehot_value)
for idx, data in enumerate(onehot_value):
    assert len(data) == len(test_data[idx])

total = [np.concatenate(
    (position[idx], tf.one_hot(onehot_value[idx], len(vob), on_value = 1.0, off_value = 0.0, axis = -1).numpy()),
    axis = 1)
         for idx, p in enumerate(position)]
print("preprocessed")

y = np.array([d[-1] for d in total])
X = [d[0:-1] for d in total]

input_dimension = X[1].shape[1]

padded_sequences = pad_sequences(X, padding = 'pre').astype('float32')

model = keras.Sequential()
model.add(layers.LSTM(input_dimension))
print("model constructed")

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['acc'])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath = "mymodel_{epoch}",
        save_best_only = True,  # Only save a model if `val_loss` has improved.
        monitor = "val_loss",
        verbose = 1,
    )
]
history = model.fit(padded_sequences[0:-128], y[0:-128], epochs = 5, batch_size = 32, callbacks = callbacks,
                    validation_split = 0.1)

test_loss, test_acc = model.evaluate(padded_sequences[-128:], y[-128:])
print('Test accuracy:', test_acc)
