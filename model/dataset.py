import pickle as pkl
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import random
import json
from gensim.models import Word2Vec


def load_ast(path):
    with open(path) as f:
        data = f.readlines()
        data = "".join(data).split("=\n")
    data_seq = [d.split() for d in data if len(d) > 0]
    return data, data_seq


def get_onehot(train_data, train_data_seq,dev_data, dev_data_seq):
    vob = {}
    vob_src = "".join(train_data).split() + "".join(dev_data).split()
    for word in vob_src:
        if word in vob:
            vob[word] += 1
        else:
            vob[word] = 1
    vob_map = {word: idx for idx, word in enumerate(vob.keys())}
    print("vob_map", len(vob_map))
    with open("vob_map.json", "w") as m:
        json.dump(vob_map, m)

    train_onehot_value = [[vob_map[word] for word in one] for one in train_data_seq]
    dev_onehot_value = [[vob_map[word] for word in one] for one in dev_data_seq]
    onehot_value = train_onehot_value + dev_onehot_value
    onehot_encoding = [tf.one_hot(val, len(vob), on_value = 1.0, off_value = 0.0, axis = -1) for val in onehot_value]
    return onehot_encoding


def get_word2vec():
    model = Word2Vec.load("../word2vec.model")
    return model.wv


def get_position(train_path, dev_path):
    with open(train_path, "rb") as f:
        train_position = pkl.load(f)
    with open(dev_path, "rb") as f:
        dev_position = pkl.load(f)
    position = train_position + dev_position
    return position,train_position, dev_position


def get_dataset(encoding, position):
    concatenated_encoding = [np.concatenate(
        (position[idx], encoding[idx]),
        axis = 1) for idx, p in enumerate(encoding)]
    y = []
    X = []
    for i, e in enumerate(encoding):
        for repeat in range(2):
            num = random.randint(1, len(e) // 2 + 1)
            y.append(encoding[i][-num])
            X.append(concatenated_encoding[i][0:-num])
    y = np.array(y)
    X_pad = pad_sequences(X, maxlen = 512, padding = 'pre', truncating = 'pre').astype('float32')
    print("X_pad:", X_pad.shape)
    print("y:", y.shape)
    return X_pad, y


def get_dataset_onehot(position,train_data, train_data_seq,dev_data, dev_data_seq):
    onehot_encoding = get_onehot(train_data, train_data_seq,dev_data, dev_data_seq)
    for idx, data in enumerate(onehot_encoding):
        if len(data) != len(position[idx]):
            print(len(data))
            print(len(position[idx]))
            print(idx)
        assert len(data) == len(position[idx])
    X, y = get_dataset(onehot_encoding, position)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

    with open('./dataset_train_onehot.pkl', 'wb') as f:
        pkl.dump([X_train, y_train], f)

    with open('./dataset_test_onehot.pkl', 'wb') as f:
        pkl.dump([X_test, y_test], f)


def get_dataset_word2vec(position, train_data_seq,dev_data_seq):
    word2vec = get_word2vec()
    train_word2vec = [[word2vec[word] for word in one] for one in train_data_seq]
    dev_word2vec = [[vob_map[word] for word in one] for one in dev_data_seq]
    word2vec = train_word2vec + dev_word2vec
    X, y = get_dataset(word2vec, position)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
    with open('./dataset_train_word2vec.pkl', 'wb') as f:
        pkl.dump([X_train, y_train], f)

    with open('./dataset_test_word2vec.pkl', 'wb') as f:
        pkl.dump([X_test, y_test], f)


if __name__ == "__main__":
    DATA_SET_PATH = '../'
    TRAIN_FILE_NAME = 'javaCorpus_train'
    DEV_FILE_NAME = 'javaCorpus_dev'

    TRAIN_AST_PATH = os.path.join(DATA_SET_PATH, TRAIN_FILE_NAME + '.txt')
    TRAIN_POS_PATH = os.path.join(DATA_SET_PATH, TRAIN_FILE_NAME + '.pkl')
    DEV_AST_PATH = os.path.join(DATA_SET_PATH, DEV_FILE_NAME + '.txt')
    DEV_POS_PATH = os.path.join(DATA_SET_PATH, DEV_FILE_NAME + '.pkl')

    position,train_position, dev_position = get_position(TRAIN_POS_PATH, DEV_POS_PATH)
    train_data, train_data_seq = load_ast(TRAIN_AST_PATH)
    dev_data, dev_data_seq = load_ast(DEV_AST_PATH)

    get_dataset_onehot(position,train_data, train_data_seq,dev_data, dev_data_seq)

    get_dataset_word2vec(train_position, train_data_seq,dev_data_seq)

