#!/usr/bin/python3

import numpy as np
import scipy
import tensorflow as tf
import pickle
import keras.backend as K
import keras
import operator
import math
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.callbacks import Callback

def micro_av_precision(conf_matrix):
    true_pos = np.zeros(conf_matrix.shape[0])
    false_pos = np.zeros(conf_matrix.shape[0])
    all_true_pos = 0
    factor = 0
    #rows are ground truth
    for i in range(conf_matrix.shape[0]):
        true_pos[i] += conf_matrix[i,i]
        false_pos[i] += sum(conf_matrix[:,i]) - conf_matrix[i,i]
        all_true_pos += true_pos[i]
        factor += true_pos[i] + false_pos[i]
    return all_true_pos/(factor + K.epsilon())

def micro_av_recall(conf_matrix):
    true_pos = np.zeros(conf_matrix.shape[0])
    false_neg = np.zeros(conf_matrix.shape[0])
    all_true_pos = 0
    factor = 0
    #rows are ground truth
    for i in range(conf_matrix.shape[0]):
        true_pos[i] += conf_matrix[i,i]
        false_neg[i] += sum(conf_matrix[i,:]) - conf_matrix[i,i]
        all_true_pos += true_pos[i]
        factor += true_pos[i] + false_neg[i]
    return all_true_pos/(factor + K.epsilon())

def micro_av_f1(conf_matrix):
    prec = micro_av_precision(conf_matrix)
    recall = micro_av_recall(conf_matrix)
    return 2 * prec * recall / (prec + recall + K.epsilon()), recall, prec

class Metrics(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        predict = np.asarray(self.model.predict(self.test_data[0]))

        if (epoch % 20):
            return

        test_pred = []
        test_true = self.test_data[1]
        #get class pred vector
        for oh_vec in predict:
            max_idx, max_val = max(enumerate(oh_vec), key=operator.itemgetter(1))
            test_pred.append(max_idx)

        if (len(test_true) != len(test_pred)):
            print("Error: pred vector size not equal to test vector!")
            print(len(test_true), len(test_pred), len(predict))
            exit(0)
        conf_mat = confusion_matrix(test_true, test_pred)
        print(conf_mat)
        _val_f1, _val_recall, _val_precision = micro_av_f1(conf_mat)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
        return

def _main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_data', required=True, help='Path to train data'
    )

    FLAGS = parser.parse_args()

    train_data = FLAGS.train_data

    with open(train_data, 'rb') as f:
        data = pickle.load(f)

    dataArray = data.values

    input_X = dataArray[:,:-1]
    input_Y = dataArray[:,-1]
    input_Y = list(map(int, input_Y))
    classes = np.unique(input_Y)

    class_num_to_label = {}
    label_to_class_num = {}
    index = 0

    for cl in classes:
        class_num_to_label[index] = cl
        label_to_class_num[cl] = index
        index += 1

    classes_count = len(classes)

    hidden_size = 512

    class_weights = compute_class_weight('balanced', classes, input_Y)

    model = Sequential()
    model.add(Dense(hidden_size, input_shape = (input_X.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(hidden_size, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(hidden_size, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(classes_count, input_dim = hidden_size, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    print(model.summary())

    data_size = len(input_Y)

    #input_Y from labels to class num
    input_Y_classes = []
    for pred in input_Y:
        input_Y_classes.append(label_to_class_num[pred])

    splits = 5

    for split in range(splits):
        X_train, X_test, Y_train, Y_test = train_test_split(input_X, input_Y_classes, shuffle=True, test_size=0.35)
        y_train_oh = keras.utils.to_categorical(Y_train, classes_count)
        y_test_oh = keras.utils.to_categorical(Y_test, classes_count)                                                                 
        print("Split: ", split)    
        model.fit(X_train, y_train_oh, epochs=60, shuffle=True, batch_size = 5000,
                    callbacks=[Metrics((X_test, Y_test))], class_weight = class_weights)

    model.save("trained/" + 'model_trained.h5')

if __name__ == '__main__':
    _main()