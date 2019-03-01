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

import os

from sklearn.metrics import accuracy_score, confusion_matrix

from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, TimeDistributed, Bidirectional, Flatten
from keras.layers import Embedding
from keras import optimizers

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

def build_model(input_shape, trained_model, classes_count):
	hidden_size = 512

	model = Sequential()
	model.add(Dense(hidden_size, input_shape = input_shape))
	model.add(Activation('relu'))
	model.add(Dropout(0.3))
	model.add(Dense(hidden_size, activation = 'relu'))
	model.add(Dropout(0.3))
	model.add(Dense(hidden_size, activation = 'relu'))
	model.add(Dropout(0.3))
	model.add(Dense(classes_count, input_dim = hidden_size, activation = 'softmax'))

	model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

	model.load_weights(trained_model)
	print(model.summary())

	return model

def _main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--trained_model', required=True, help='Path to trained model(h5 format)'
    )

    parser.add_argument(
        '--validation_data', required=True, help='validation data path (pkl format)'
    )

    FLAGS = parser.parse_args()

    trained_model = FLAGS.trained_model
    validation_data = FLAGS.validation_data

    with open(validation_data, 'rb') as f:
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

    input_shape = input_X.shape[1]

    model = build_model((input_shape,), trained_model, classes_count)

    #input_Y from labels to class num
    input_Y_classes = []
    for pred in input_Y:
        input_Y_classes.append(label_to_class_num[pred])

    predict = np.asarray(model.predict(input_X))

    test_pred = []
    #get class pred vector
    for oh_vec in predict:
        max_idx, max_val = max(enumerate(oh_vec), key=operator.itemgetter(1))
        test_pred.append(max_idx)

    test_true = input_Y_classes

    if (len(test_true) != len(test_pred)):
        print("Error: pred vector size not equal to test vector!")
        print(len(test_true), len(test_pred), len(predict))
        exit(0)

    conf_mat = confusion_matrix(test_true, test_pred)
    _val_f1, _val_recall, _val_precision = micro_av_f1(conf_mat)
    print(" — micro_av_f1: %f — micro_av_precision: %f — micro_av_recall %f" %(_val_f1, _val_precision, _val_recall))
    return

if __name__ == '__main__':
    _main()