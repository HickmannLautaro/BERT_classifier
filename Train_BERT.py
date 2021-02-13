#!/usr/bin/env python
# coding: utf-8


# -------- Load libraries ------- ###
# Load Huggingface transformers
from transformers import TFBertModel, BertConfig, BertTokenizerFast

# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
# And pandas for data import + sklearn because you always need sklearn
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import io
import itertools
import os
import yaml
import sys


def plot_confusion_matrix(cm, f1_score, accuracy_score, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
        :param cm: (array, shape = [n, n]): a confusion matrix of integer classes
        :param class_names: (array, shape = [n]): String names of the integer classes
        :param accuracy_score: accuracy score for plotting
        :param f1_score: f1_score score for plotting
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix, accuracy {:.4f} \n f1 Score {:.4f}".format(accuracy_score, f1_score))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def append_label(data, arguments):
    if arguments['labels'] is not None:
        for arg in arguments['labels']:
            if arg[0] == 'Target':
                for cat in arg[1:][::-1]:
                    data['Text'] = data[cat].str.cat(data['Text'], sep=". ")
                    # test['Text']=test[cat].str.cat(test['Text'],sep=". ")
            else:
                for cat in arg[::-1]:
                    file = np.load(cat, allow_pickle=True)
                    labels = file['train_class_names'][file['train_pred_raw'].argmax(axis=1)]
                    # labels_test=file['test_class_names'][file['test_pred_raw'].argmax(axis=1)]

                    data['aux'] = labels
                    data['Text'] = data['aux'].str.cat(data['Text'], sep=". ")

                    # test['aux']=labels_test
                    # test['Text']=test['aux'].str.cat(test['Text'],sep=". ")

    return data


def append_test_label(test, arguments):
    if arguments['test_labels'] is not None:
        for arg in arguments['test_labels']:
            if arg[0] == 'Target':
                for cat in arg[1:][::-1]:
                    test['Text'] = test[cat].str.cat(test['Text'], sep=". ")
                    # test['Text']=test[cat].str.cat(test['Text'],sep=". ")
            else:
                for cat in arg[::-1]:
                    file = np.load(cat, allow_pickle=True)
                    labels_test = file['test_class_names'][file['test_pred_raw'].argmax(axis=1)]
                    test['aux'] = labels_test
                    test['Text'] = test['aux'].str.cat(test['Text'], sep=". ")

    return test


def get_data(arguments,hyp_search=False):
    lvl = arguments['lvl']
    data_path = arguments['data_path']

    # Import data from csv
    data = pd.read_csv(data_path + "/train.csv")
    data = data.rename(columns={"text": "Text", "l1": "Cat1", "l2": "Cat2", "l3": "Cat3"})

    # Add labels to text
    data = append_label(data, arguments)

    # Select target columns
    cat_num = str('Cat' + str(lvl))

    data = data[['Text', cat_num]]
    data = data.sample(frac=1)

    # Training data
    # Set model output as categorical and save in new label col
    cat_label = str(cat_num + '_label')
    data[cat_label] = pd.Categorical(data[cat_num])
    # Transform your output to numeric
    data[cat_num] = data[cat_label].cat.codes

    train_class_names = np.unique(data[cat_label])
    if not hyp_search:
        print("Class names: \n Train: {} ".format(train_class_names))
        print("Training data")
        print(data.head())

    train_target = to_categorical(data[cat_num])

    return data, train_class_names, train_target


def get_test_data(arguments):
    lvl = arguments['lvl']
    data_path = arguments['data_path']

    # Import data from csv
    test = pd.read_csv(data_path + "/test.csv")
    test = test.rename(columns={"text": "Text", "l1": "Cat1", "l2": "Cat2", "l3": "Cat3"})

    # Add labels to text
    test = append_test_label(test, arguments)

    # Select target columns
    cat_num = str('Cat' + str(lvl))

    test = test[['Text', cat_num]]

    # Setup test data for logging
    cat_label = str(cat_num + '_label')
    # Set model output as categorical and save in new label col
    test[cat_label] = pd.Categorical(test[cat_num])
    # Transform your output to numeric
    test[cat_num] = test[cat_label].cat.codes

    test_class_names = np.unique(test[cat_label])

    test_target = test[cat_num]

    return test, test_class_names, test_target


def get_bert_model(model_name, config, max_length, class_names, hyp_search):
    # Load the Transformers BERT model
    transformer_model = TFBertModel.from_pretrained(model_name, config=config)

    # ------- Build the model ------- ###

    # TF Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Model

    # Load the MainLayer
    bert = transformer_model.layers[0]

    # Build your model input
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(max_length,), name='attention_mask',
                           dtype='int32')  # Ignores padded part of sentences
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

    # Load the Transformers BERT model as a layer in a Keras model
    bert_model = bert(inputs)[1]
    dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
    pooled_output = dropout(bert_model, training=False)

    # Then build your model output
    output = Dense(units=len(class_names),
                   kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='Cat')(pooled_output)

    # And combine it all in a model object
    model = Model(inputs=inputs, outputs=output, name='BERT_MultiClass')

    if not hyp_search:
        # Take a look at the model
        print(model.summary())

    # ------- Setup training ------- ###

    # Set an optimizer
    optimizer = Adam(
        learning_rate=5e-05,
        epsilon=1e-08,
        decay=0.01,
        clipnorm=1.0)

    # Set loss and metrics
    loss = CategoricalCrossentropy(from_logits=True)
    metric = [CategoricalAccuracy('accuracy'), tfa.metrics.F1Score(num_classes=len(class_names), average='macro')]

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metric)
    return model


def get_tokenized(model_name, config, data, max_length):
    # Load BERT tokenizer

    # Load BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name, config=config)

    # Tokenize the input (takes some time) for training and test (for logging) data

    x = tokenizer(
        text=data['Text'].to_list(),
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True)
    return x


def run_experiment(arguments, hyp_search=False):
    if not hyp_search:
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
    path = "/" + model_name + "/" + data_path + "/lvl" + str(
        lvl) + "/trained_" + hierar + "_" + lable_type + "/" + str(
        max_length) + "T_" + str(
        epochs) + "e_" + str(batch_size) + "b/"

    if hyp_search:
        aux_path = os.getcwd() + "/hyperparameters_search" + path
    else:
        aux_path = os.getcwd() + "/saved_models" + path

    try:
        os.makedirs(aux_path)
    except OSError:
        print("%s already exists" % aux_path)
    else:
        print("Successfully created the directory %s " % aux_path)

    if hyp_search:
        dir_list = os.listdir("./hyperparameters_search" + path)

    else:
        dir_list = os.listdir("./saved_models" + path)

    if len(dir_list) == 0:
        run = 1
    else:
        aux = [int(x[3:]) for x in dir_list]
        aux.sort()
        run = aux[-1] + 1
    path += "Run" + str(run)

    print("Run started: " + path)

    if hyp_search:
        path_model = "./hyperparameters_search" + path
    else:
        path_model = "./saved_models" + path

    try:
        os.makedirs(path_model)
    except OSError:
        print("%s already exists" % path_model)
    else:
        print("Successfully created the directory %s " % path_model)

    logdir = path_model + "/logs"
    path_save_model = path_model + "/model/"

    path_model_plot = path_model + "/model.png"

    if hyp_search:
        print("Config: " + path + "\nRelative paths " +
              "\n \n##### Model data #####" +
              "\n \nlog dir:" + logdir +
              "\n \nSaved model dir:" + path_save_model +
              "\n \n \n##### Plots and predictions #####" +
              "\n \nPlots dir:" + path_model_plot)
    else:
        path_saved_data = path_model + "/test_pred_raw.npz"
        print("Config: " + path + "\nRelative paths " +
              "\n \n##### Model data #####" +
              "\n \nlog dir:" + logdir +
              "\n \nSaved model dir:" + path_save_model +
              "\n \n \n##### Plots and predictions #####" +
              "\n \nPlots dir:" + path_model_plot +
              "\n \nSaved data dir:" + path_saved_data)


    ### --------- Import data --------- ###

    data, train_class_names, target = get_data(arguments, hyp_search)
    ### --------- Load BERT ---------- ###

    # Load transformers config and set output_hidden_states to False
    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = False

    # Load the Transformers BERT model

    model = get_bert_model(model_name, config, max_length, train_class_names, hyp_search)

    # Tokenize the input (takes some time) for training and test (for logging) data	

    x = get_tokenized(model_name, config, data, max_length)

    ### ------- Callbacks ------- ###
    # Tensorboard callback

    if hyp_search:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False)
        delta=0.0005
    else:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=False,
                                                              write_images=True, profile_batch='10,20')
        delta = 0.0001
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', verbose=1, mode="max",patience=4,
                                                     restore_best_weights=True, min_delta=delta)
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
    if not hyp_search:
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

    print("Run finished: " + path)
    test, test_class_names, test_target = get_test_data(arguments)
    # ----- Evaluate the model ------
    test_x = get_tokenized(model_name, config, test, max_length)

    test_pred_raw = model.predict(x={'input_ids': test_x['input_ids'], 'attention_mask': test_x['attention_mask']},
                                  verbose=1)


    test_pred = np.argmax(test_pred_raw, axis=1)
    # Calculate the confusion matrix.
    cm = sklearn.metrics.confusion_matrix(test_target, test_pred)
    cm[np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2) < 0.05] = 0

    f1_score = sklearn.metrics.f1_score(test_target, test_pred, average='macro')
    accuracy_score = sklearn.metrics.accuracy_score(test_target, test_pred)

    path_confusion_mat = path_model + '/conf.png'

    figure = plot_confusion_matrix(cm, f1_score, accuracy_score, class_names=test_class_names)
    figure.savefig(path_confusion_mat)
    plt.close(figure)

    # noinspection PyTypeChecker
    report = sklearn.metrics.classification_report(test_target, test_pred, target_names=test_class_names, digits=4)


    if hyp_search:

        np.savez(path_model+ "tested_" + test_labels_type +"/rep_and_histo.npz" , report=report, hist=history.history)

        return f1_score, accuracy_score
    else:
        train_pred_raw = model.predict(x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']}, verbose=1)
        np.savez(path_saved_data, test_pred_raw=test_pred_raw, f1_score=f1_score, accuracy_score=accuracy_score,
                 train_pred_raw=train_pred_raw, report=report, hist=history.history, train_class_names=train_class_names, test_class_names=test_class_names)

def main():
    print("Tensorflow version: ", tf.__version__)

    # rtx 3080 tf 2.4.0-rc4 bug
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    list_args = sys.argv[1:]

    if len(list_args) < 1:
        print("Config missing")
        sys.exit(2)

    for conf in list_args:
        with open(conf) as f:
            arguments = yaml.load(f, Loader=yaml.FullLoader)
        for i in range(arguments['repetitions']):
            run_experiment(arguments)


if __name__ == "__main__":
    main()
