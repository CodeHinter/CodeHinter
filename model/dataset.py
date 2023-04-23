import pickle as pkl
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def load_ast(path):
    with open(path) as f:
        data = f.readlines()
        data = "".join(data).split("=\n")
    data_seq = [d.split() for d in data if len(d) > 0]
    return data, data_seq


def get_onehot(train_path, dev_path):
    train_data, train_data_seq = load_ast(train_path)
    dev_data, dev_data_seq = load_ast(dev_path)

    vob = {}
    vob_src = "".join(train_data).split() + "".join(dev_data).split()
    for word in vob_src:
        if word in vob:
            vob[word] += 1
        else:
            vob[word] = 1
    vob_map = {word: idx for idx, word in enumerate(vob.keys())}
    print("vob_map", len(vob_map))

    train_onehot_value = [[vob_map[word] for word in one] for one in train_data_seq]
    dev_onehot_value = [[vob_map[word] for word in one] for one in dev_data_seq]
    onehot_value = train_onehot_value + dev_onehot_value
    return onehot_value, vob


def get_position(train_path, dev_path):
    with open(train_path, "rb") as f:
        train_position = pkl.load(f)
    with open(dev_path, "rb") as f:
        dev_position = pkl.load(f)
    position = train_position + dev_position
    return position


def get_dataset(onehot_value, position, vob):
    encoding = [np.concatenate(
        (position[idx], tf.one_hot(onehot_value[idx], len(vob), on_value = 1.0, off_value = 0.0, axis = -1).numpy()),
        axis = 1) for idx, p in enumerate(position)]
    y = np.array([d[-1] for d in encoding])
    X = [d[0:-1] for d in encoding]

    X_pad = pad_sequences(X, maxlen = 512, padding = 'pre', truncating = 'pre').astype('float32')
    print("X_pad:", X_pad.shape)

    return X_pad, y


if __name__ == "__main__":
    DATA_SET_PATH = '../'
    TRAIN_FILE_NAME = 'javaCorpus_train'
    DEV_FILE_NAME = 'javaCorpus_dev'

    TRAIN_AST_PATH = os.path.join(DATA_SET_PATH, TRAIN_FILE_NAME + '.txt')
    TRAIN_POS_PATH = os.path.join(DATA_SET_PATH, TRAIN_FILE_NAME + '.pkl')
    DEV_AST_PATH = os.path.join(DATA_SET_PATH, DEV_FILE_NAME + '.txt')
    DEV_POS_PATH = os.path.join(DATA_SET_PATH, DEV_FILE_NAME + '.pkl')

    position = get_position(TRAIN_POS_PATH, DEV_POS_PATH)
    onehot_value, vob = get_onehot(TRAIN_AST_PATH, DEV_AST_PATH)
    for idx, data in enumerate(onehot_value):
        if len(data) != len(position[idx]):
            print(len(data))
            print(len(position[idx]))
            print(idx)
        assert len(data) == len(position[idx])
    X, y = get_dataset(onehot_value, position, vob)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

    with open('./dataset_train.pkl', 'wb') as f:
        pkl.dump([X_train, y_train], f)

    with open('./dataset_test.pkl', 'wb') as f:
        pkl.dump([X_test, y_test], f)
