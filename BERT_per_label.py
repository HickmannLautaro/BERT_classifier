#!/usr/bin/env python
# coding: utf-8


import os
import matplotlib.pyplot as plt
import numpy as np
# And pandas for data import + sklearn because you always need sklearn
import pandas as pd
import sklearn
import tensorflow as tf
# Then what you need from tensorflow.keras
from tensorflow.keras.utils import to_categorical
# -------- Load libraries ------- ###
# Load Huggingface transformers
from transformers import BertConfig

from BERT_per_lvl import get_bert_model, get_tokenized, plot_confusion_matrix


def convert_data(data, lvl):
    cat_num = str('Cat' + str(lvl))
    cat_label = str(cat_num + '_label')
    data[cat_label] = pd.Categorical(data[cat_num])
    data[cat_num] = data[cat_label].cat.codes
    return data, np.unique(data[cat_label])


def get_upper_label_data(data_path, train,lvl):

    # Import data from csv
    if train:
        data = pd.read_csv(data_path + "/train.csv")
    else:
        data = pd.read_csv(data_path + "/test.csv")

    data = data.rename(columns={"text": "Text", "l1": "Cat1", "l2": "Cat2", "l3": "Cat3"})
    if lvl == 3:
        data = data[['Text', "Cat1", "Cat2", "Cat3"]]
    else:
        data = data[['Text', "Cat1", "Cat2"]]
    class_names = []

    # Training data
    # Set model output as categorical and save in new label col
    for lvl in range(1,lvl+1):
        data, names = convert_data(data, lvl)
        class_names.append(names)
    return data, class_names


def get_data_per_sub_label(data_path,train, lvl):

    # Import data from csv
    if train:
        data = pd.read_csv(data_path + "/train.csv")
    else:
        data = pd.read_csv(data_path + "/test.csv")

    data = data.rename(columns={"text": "Text", "l1": "Cat1", "l2": "Cat2", "l3": "Cat3"})
    if lvl == 3:
        data = data[['Text', "Cat1", "Cat2", "Cat3"]]
    else:
        data = data[['Text', "Cat1", "Cat2"]]
    class_names = []

    # Training data
    # Set model output as categorical and save in new label col
    data, names = convert_data(data, lvl-1)
    cat_num = str('Cat' + str(lvl-1))
    cat_num_desired = str('Cat' + str(lvl))

    class_names.append(names)

    return [data[data[cat_num] == i][["Text", cat_num_desired]] for i in range(class_names[0].shape[0])]


def get_mapping(per_cat_data, original_data, lvl):
    cat_num = str('Cat' + str(lvl))
    cat_label = str(cat_num + '_label')
    aux, aux_names = convert_data(per_cat_data, lvl)
    mapping = np.ones(len(aux_names)) * -1
    class_name = aux_names[0]
    for class_name in aux_names:
        mapping[aux[aux[cat_label] == class_name][[cat_num]].drop_duplicates()] = \
            original_data[original_data[cat_label] == class_name][[cat_num]].drop_duplicates()
    return mapping


def get_data_and_mapp_per_sub_label(data_path,lvl, train=True):
    per_cat_data = get_data_per_sub_label(data_path,train, lvl)
    original_data, original_class_names = get_upper_label_data(data_path, train, lvl)
    return [[sub_cat_data, get_mapping(sub_cat_data, original_data, lvl)] for sub_cat_data in per_cat_data]


def train_per_label(arguments):
    print("#" * 150)
    print("#" * 150)

    # --------- Setup BERT ----------
    # Name of the BERT model to use
    model_name = arguments['model_name']
    # Max length of tokens
    max_length = arguments['max_length']
    epochs = arguments['epochs']
    batch_size = arguments['batch_size']
    lvl = arguments['lvl']
    hierar = arguments['hierar']
    lable_type = arguments['lable_type']
    test_labels_type = arguments['test_labels_type']
    data_path = arguments['data_path']
    # --------- Setup logs paths ----------

    path = "/" + model_name + "/" + data_path + "/lvl" + str(lvl) + "/trained_" + hierar + "_" + lable_type + "/" + str(max_length) + "T_" + str(epochs) + "e_" + str(batch_size) + "b/"

    aux_path = os.getcwd() + "/saved_models" + path

    try:
        os.makedirs(aux_path)
    except OSError:
        print("%s already exists" % aux_path)
    else:
        print("Successfully created the directory %s " % aux_path)

    dir_list = os.listdir("./saved_models" + path)

    if len(dir_list) == 0:
        run = 1
    else:
        aux = [int(x[3:]) for x in dir_list]
        aux.sort()
        run = aux[-1] + 1
    path += "Run" + str(run)

    print("Run started: " + path)

    ### --------- Import data --------- ###
    per_cat_data = [[data[0], np.unique(data[0]["Cat2_label"]), to_categorical(data[0]["Cat2"]), data[1]] for data in get_data_and_mapp_per_sub_label(data_path, lvl)]
    per_cat_data_test = [[data[0], np.unique(data[0]["Cat2_label"]), to_categorical(data[0]["Cat2"]), data[1]] for data in get_data_and_mapp_per_sub_label(data_path, lvl, train=False)]
    for iter_num, per_cat_data_iter in enumerate(zip(per_cat_data, per_cat_data_test)):

        path_model = "./saved_models" + path + "/Class" + str(iter_num)

        try:
            os.makedirs(path_model)
        except OSError:
            print("%s already exists" % path_model)
        else:
            print("Successfully created the directory %s " % path_model)

        logdir = path_model + "/logs"
        path_save_model = path_model + "/model/"

        path_saved_data = path_model + "/tested_" + test_labels_type
        path_model_plot = path_saved_data + "/model.png"

        try:
            os.makedirs(path_saved_data)
        except OSError:
            pass

        print("Config: " + path + "\nRelative paths " +
              "\n \n##### Model data #####" +
              "\n \nlog dir:" + logdir +
              "\n \nSaved model dir:" + path_save_model +
              "\n \n \n##### Plots and predictions #####" +
              "\n \nPlots dir:" + path_model_plot +
              "\n \nSaved data dir:" + path_saved_data)

        data, train_class_names, target, train_mapping = per_cat_data_iter[0]
        test, test_class_names, test_target, test_mapping = per_cat_data_iter[1]

        ### --------- Load BERT ---------- ###

        # Load transformers config and set output_hidden_states to False
        config = BertConfig.from_pretrained(model_name)
        config.output_hidden_states = False

        # Load the Transformers BERT model

        model = get_bert_model(model_name, config, max_length, train_class_names, False)

        # Tokenize the input (takes some time) for training and test (for logging) data

        x = get_tokenized(model_name, config, data, max_length)

        ### ------- Callbacks ------- ###
        # Tensorboard callback

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=False, write_images=False, profile_batch='10,20')

        delta = 0.0001
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', verbose=1, mode="max", patience=10, restore_best_weights=True, min_delta=delta)

        ### ------- Train the model ------- ###

        # Fit the model
        history = model.fit(
            x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
            y=target,  # y_Cat1,
            validation_split=0.2,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[tensorboard_callback, earlystopping])

        # Load the best weights and save the model
        model.save(path_save_model)

        tf.keras.utils.plot_model(
            model,
            to_file=path_model_plot,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )

        print("Run finished: " + path + "/Class" + str(iter_num))
        # ----- Evaluate the model ------
        test_x = get_tokenized(model_name, config, test, max_length)

        test_pred_raw = model.predict(x={'input_ids': test_x['input_ids'], 'attention_mask': test_x['attention_mask']}, verbose=1)

        test_pred = np.argmax(test_pred_raw, axis=1)
        test_target = np.argmax(test_target, axis=1)
        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(test_target, test_pred)
        cm[np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2) < 0.005] = 0

        f1_score = sklearn.metrics.f1_score(test_target, test_pred, average='macro')
        accuracy_score = sklearn.metrics.accuracy_score(test_target, test_pred)

        path_confusion_mat = path_saved_data + '/conf.png'

        figure = plot_confusion_matrix(cm, f1_score, accuracy_score, class_names=test_class_names)
        figure.savefig(path_confusion_mat)
        plt.close(figure)

        # noinspection PyTypeChecker
        report = sklearn.metrics.classification_report(test_target, test_pred, target_names=test_class_names, digits=4)

        train_pred_raw = model.predict(x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']}, verbose=1)

        np.savez(path_saved_data + "/rep_and_histo.npz", test_pred_raw=test_pred_raw, f1_score=f1_score,
                 accuracy_score=accuracy_score,
                 train_pred_raw=train_pred_raw, report=report, hist=history.history,
                 train_class_names=train_class_names, test_class_names=test_class_names, train_mapping=train_mapping, test_mapping=test_mapping)


def main():
    print("Tensorflow version: ", tf.__version__)

    # rtx 3080 tf 2.4.0-rc4 bug
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    arguments = {'model_name': 'bert-base-uncased',
                 'max_length': 100,
                 'epochs': 60,  #
                 'batch_size': 45,
                 'repetitions': 1,
                 'data_path': 'dbpedia',
                 'lvl': 2,
                 'hierar': 'hierarchical',
                 'lable_type': 'per_label',
                 'test_labels_type': '_'}
    train_per_label(arguments)


if __name__ == "__main__":
    main()
