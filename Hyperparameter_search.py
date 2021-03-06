#!/usr/bin/env python
# coding: utf-8
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # Lower log level to get less clutter
from BERT_per_lvl import run_experiment  # Import functions to run experiments
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
import sys


# This file should be redone with nicer functions but not time anymore to corroborate the results after changing functions
# Functions in functions not good but otherwise HParams broke
def hyp_search_lvl1_flatt():
    """
    Run hyperparameter on the amazon dataset for the flat approach on level 1
    :return: HParam run logs
    """
    # Set HParams up
    HP_MAX_LENGTH = hp.HParam('max_length', hp.Discrete([64, 100, 256, 512]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([45, 20, 40, 50]))

    METRIC_ACCURACY = 'accuracy_score'
    METRIC_f1 = 'f1_score'

    # Simulate config file
    arguments = {'model_name': 'bert-base-uncased',
                 'max_length': 100,
                 'epochs': 40,  #
                 'batch_size': 40,
                 'repetitions': 1,
                 'data_path': 'amazon',
                 'lvl': 1,
                 'labels': None,
                 'test_labels': None,
                 'hierar': 'flatt',
                 'lable_type': '_',
                 'test_labels_type': '_'}
    # Get config values
    model_name = arguments['model_name']
    lvl = arguments['lvl']
    data_path = arguments['data_path']
    hierar = arguments['hierar']
    lable_type = arguments['lable_type']
    test_labels_type = arguments['test_labels_type']

    # Create custom summary for the HParam logs
    with tf.summary.create_file_writer("hyperparameters_search/" + model_name + "/" + data_path + "/lvl" + str(
            lvl) + "/trained_" + hierar + "_" + lable_type + "/tested_" + test_labels_type + '/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_MAX_LENGTH, HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy_score'),
                     hp.Metric(METRIC_f1, display_name='f1_score')],
        )

    def run(run_dir, hparams, arguments):
        """
        Run experiments twice on a set of hparams and log the metrics
        :param run_dir: path of log file
        :param hparams: dict with parameters to test in this run
        :param arguments: config file for the experiment
        :return: log of run is saved to path
        """

        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            arguments['max_length'] = hparams[HP_MAX_LENGTH]  # arguments['max_length']
            arguments['batch_size'] = hparams[HP_BATCH_SIZE]  # arguments['epochs']

            f1_score_1, accuracy_score_1 = run_experiment(arguments, hyp_search=True, )
            f1_score_2, accuracy_score_2 = run_experiment(arguments, hyp_search=True, )

            f1_score, accuracy_score = np.mean([f1_score_1, f1_score_2]), np.mean([accuracy_score_1, accuracy_score_2])
            tf.summary.scalar(METRIC_ACCURACY, accuracy_score, step=1)
            tf.summary.scalar(METRIC_f1, f1_score, step=1)

    # Experiment counter
    session_num = 0

    for max_length in HP_MAX_LENGTH.domain.values[::-1]:
        for batch_size in HP_BATCH_SIZE.domain.values[::-1]:
            hparams = {
                HP_MAX_LENGTH: max_length,
                HP_BATCH_SIZE: batch_size,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            try:
                run("hyperparameters_search/" + model_name + "/" + data_path + "/lvl" + str(lvl) + "/trained_" + hierar + "_" + lable_type + "/tested_" + test_labels_type + '/hparam_tuning/' + run_name, hparams, arguments)
            except tf.errors.ResourceExhaustedError as e:  # If out of memory error abort this run and test with new hypeparameters.
                print("Out of memory")

            session_num += 1


# Functions in functions not good but otherwise HParams broke
def hyp_search_lvl2_flatt():
    """
    Run hyperparameter on the amazon dataset for the flat approach on level 2
    :return: HParam run logs
    """
    # Set HParams up
    HP_MAX_LENGTH = hp.HParam('max_length', hp.Discrete([100, 256, 512]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([45, 50, 40, 60]))

    METRIC_ACCURACY = 'accuracy_score'
    METRIC_f1 = 'f1_score'

    # Simulate config file
    arguments = {'model_name': 'bert-base-uncased',
                 'max_length': 100,
                 'epochs': 40,  #
                 'batch_size': 40,
                 'repetitions': 1,
                 'data_path': 'amazon',
                 'lvl': 2,
                 'labels': None,
                 'test_labels': None,
                 'hierar': 'flatt',
                 'lable_type': '_',
                 'test_labels_type': '_'}
    # Get config values
    model_name = arguments['model_name']
    lvl = arguments['lvl']
    data_path = arguments['data_path']
    hierar = arguments['hierar']
    lable_type = arguments['lable_type']
    test_labels_type = arguments['test_labels_type']

    # Create custom summary for the HParam logs
    with tf.summary.create_file_writer("hyperparameters_search/" + model_name + "/" + data_path + "/lvl" + str(lvl) + "/trained_" + hierar + "_" + lable_type + "/tested_" + test_labels_type + '/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_MAX_LENGTH, HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy_score'),
                     hp.Metric(METRIC_f1, display_name='f1_score')],
        )

    def run(run_dir, hparams, arguments):
        """
        Run experiments twice on a set of hparams and log the metrics
        :param run_dir: path of log file
        :param hparams: dict with parameters to test in this run
        :param arguments: config file for the experiment
        :return: log of run is saved to path
        """

        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            arguments['max_length'] = hparams[HP_MAX_LENGTH]  # arguments['max_length']
            arguments['batch_size'] = hparams[HP_BATCH_SIZE]  # arguments['epochs']

            f1_score_1, accuracy_score_1 = run_experiment(arguments, hyp_search=True, )
            f1_score_2, accuracy_score_2 = run_experiment(arguments, hyp_search=True, )

            f1_score, accuracy_score = np.mean([f1_score_1, f1_score_2]), np.mean([accuracy_score_1, accuracy_score_2])

            tf.summary.scalar(METRIC_ACCURACY, accuracy_score, step=1)
            tf.summary.scalar(METRIC_f1, f1_score, step=1)

    # Experiment counter
    session_num = 0

    for max_length in HP_MAX_LENGTH.domain.values[::-1]:
        for batch_size in HP_BATCH_SIZE.domain.values[::-1]:
            hparams = {
                HP_MAX_LENGTH: max_length,
                HP_BATCH_SIZE: batch_size,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            try:
                run("hyperparameters_search/" + model_name + "/" + data_path + "/lvl" + str(lvl) + "/trained_" + hierar + "_" + lable_type + "/tested_" + test_labels_type + '/hparam_tuning/' + run_name, hparams, arguments)
            except tf.errors.ResourceExhaustedError as e:  # If out of memory error abort this run and test with new hypeparameters.
                print("Out of memory")

            session_num += 1


# Functions in functions not good but otherwise HParams broke
def hyp_search_lvl2_target_target():
    """
    Run hyperparameter on the amazon dataset for the target trained and tested per-level approach on level 2
    :return: HParam run logs
    """

    HP_MAX_LENGTH = hp.HParam('max_length', hp.Discrete([100, 256, 512]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([45, 50, 40, 60]))

    METRIC_ACCURACY = 'accuracy_score'
    METRIC_f1 = 'f1_score'

    # Simulate config file
    arguments = {'model_name': 'bert-base-uncased',
                 'max_length': 100,
                 'epochs': 40,  #
                 'batch_size': 40,
                 'repetitions': 1,
                 'data_path': 'amazon',
                 'lvl': 2,
                 'labels': [['Target', 'Cat1']],
                 'test_labels': [['Target', 'Cat1']],
                 'hierar': 'hierarchical',
                 'lable_type': 'Target',
                 'test_labels_type': 'Target'}

    # Get config values
    model_name = arguments['model_name']
    lvl = arguments['lvl']
    data_path = arguments['data_path']
    hierar = arguments['hierar']
    lable_type = arguments['lable_type']
    test_labels_type = arguments['test_labels_type']

    # Create custom summary for the HParam logs
    with tf.summary.create_file_writer("hyperparameters_search/" + model_name + "/" + data_path + "/lvl" + str(
            lvl) + "/trained_" + hierar + "_" + lable_type + "/tested_" + test_labels_type + '/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_MAX_LENGTH, HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy_score'),
                     hp.Metric(METRIC_f1, display_name='f1_score')],
        )

    def run(run_dir, hparams, arguments):
        """
        Run experiments twice on a set of hparams and log the metrics
        :param run_dir: path of log file
        :param hparams: dict with parameters to test in this run
        :param arguments: config file for the experiment
        :return: log of run is saved to path
        """

        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            arguments['max_length'] = hparams[HP_MAX_LENGTH]  # arguments['max_length']
            arguments['batch_size'] = hparams[HP_BATCH_SIZE]  # arguments['epochs']

            f1_score_1, accuracy_score_1 = run_experiment(arguments, hyp_search=True, )
            f1_score_2, accuracy_score_2 = run_experiment(arguments, hyp_search=True, )
            f1_score, accuracy_score = np.mean([f1_score_1, f1_score_2]), np.mean([accuracy_score_1, accuracy_score_2])

            tf.summary.scalar(METRIC_ACCURACY, accuracy_score, step=1)
            tf.summary.scalar(METRIC_f1, f1_score, step=1)

    # Experiment counter

    session_num = 0

    for max_length in HP_MAX_LENGTH.domain.values[::-1]:
        for batch_size in HP_BATCH_SIZE.domain.values[::-1]:
            hparams = {
                HP_MAX_LENGTH: max_length,
                HP_BATCH_SIZE: batch_size,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            try:
                run("hyperparameters_search/" + model_name + "/" + data_path + "/lvl" + str(
                    lvl) + "/trained_" + hierar + "_" + lable_type + "/tested_" + test_labels_type + '/hparam_tuning/' + run_name, hparams, arguments)
            except tf.errors.ResourceExhaustedError as e: # If out of memory error abort this run and test with new hypeparameters.
                print("Out of memory")

            session_num += 1


# Functions in functions not good but otherwise HParams broke
def hyp_search_lvl2_predicted_predicted(path_predicted):
    """
    Run hyperparameter on the amazon dataset for the predicted trained and tested per-level approach on level 2
    :return: HParam run logs
    """
    HP_MAX_LENGTH = hp.HParam('max_length', hp.Discrete([64, 100, 256, 512]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([10, 45, 20, 40, 50, 60]))

    METRIC_ACCURACY = 'accuracy_score'
    METRIC_f1 = 'f1_score'

    # Simulate config file
    arguments = {'model_name': 'bert-base-uncased',
                 'max_length': 100,
                 'epochs': 40,  #
                 'batch_size': 40,
                 'repetitions': 1,
                 'data_path': 'amazon',
                 'lvl': 2,
                 'labels': [[path_predicted]],
                 'test_labels': [[path_predicted]],
                 'hierar': 'hierarchical',
                 'lable_type': 'Predicted',
                 'test_labels_type': 'Predicted'}

    # Get config values
    model_name = arguments['model_name']
    lvl = arguments['lvl']
    data_path = arguments['data_path']
    hierar = arguments['hierar']
    lable_type = arguments['lable_type']
    test_labels_type = arguments['test_labels_type']

    # Create custom summary for the HParam logs
    with tf.summary.create_file_writer("hyperparameters_search/" + model_name + "/" + data_path + "/lvl" + str(
            lvl) + "/trained_" + hierar + "_" + lable_type + "/tested_" + test_labels_type + '/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_MAX_LENGTH, HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy_score'),
                     hp.Metric(METRIC_f1, display_name='f1_score')],
        )

    def run(run_dir, hparams, arguments):
        """
        Run experiments twice on a set of hparams and log the metrics
        :param run_dir: path of log file
        :param hparams: dict with parameters to test in this run
        :param arguments: config file for the experiment
        :return: log of run is saved to path
        """

        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            arguments['max_length'] = hparams[HP_MAX_LENGTH]  # arguments['max_length']
            arguments['batch_size'] = hparams[HP_BATCH_SIZE]  # arguments['epochs']

            f1_score_1, accuracy_score_1 = run_experiment(arguments, hyp_search=True, )
            f1_score_2, accuracy_score_2 = run_experiment(arguments, hyp_search=True, )

            f1_score, accuracy_score = np.mean([f1_score_1, f1_score_2]), np.mean([accuracy_score_1, accuracy_score_2])
            tf.summary.scalar(METRIC_ACCURACY, accuracy_score, step=1)
            tf.summary.scalar(METRIC_f1, f1_score, step=1)

    # Experiment counter
    session_num = 0

    for max_length in HP_MAX_LENGTH.domain.values[::-1]:
        for batch_size in HP_BATCH_SIZE.domain.values[::-1]:
            hparams = {
                HP_MAX_LENGTH: max_length,
                HP_BATCH_SIZE: batch_size,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            try:
                run("hyperparameters_search/" + model_name + "/" + data_path + "/lvl" + str(
                    lvl) + "/trained_" + hierar + "_" + lable_type + "/tested_" + test_labels_type + '/hparam_tuning/' + run_name, hparams, arguments)
            except tf.errors.ResourceExhaustedError as e: # If out of memory error abort this run and test with new hypeparameters.
                print("Out of memory")

            session_num += 1


def main():
    """
    Runs hyperparameter search on the amazon dataset for the flatt and per-level approaches depending on the command line arguments.
    Run Hyperparameter_search.py to do a grid-search over the predifined hyperparameters. Hyperparameters can only be done over amazon and per_lvl, but neither on DBpedia nor on per_label.
    Give one or more options to search hyperparameters: Flat_lvl1, Flat_lvl2, tgt_pred, tgt_tgt, pred_pred.
    For runs containing pred (predictions) give the rep_and_histo.npz path that should be used for the input predictions.
    For example run "python Hyperparameter_search.py Flat_lvl2 tgt_tgt pred_pred saved_models/bert-base-uncased/amazon/lvl1/trained_flatt__/100T_60e_45b/Run3/tested__/rep_and_histo.npz"
    for the hyp-search on amazon level2 flat (Flat_lvl2), target trained and predicted on level 2 (tgt_tgt), and trained and tested with the predicted label input of the flat level 1 (pred_pred saved_models/bert-base-uncased/amazon/lvl1/trained_flatt__/100T_60e_45b/Run3/tested__/rep_and_histo.npz)
    """
    print("Tensorflow version: ", tf.__version__)

    # rtx 3080 tf 2.4.0-rc4 bug
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoids Hugging Face process forking bug https://github.com/ThilinaRajapakse/simpletransformers/issues/515

    list_args = sys.argv[1:]  # Read command line arguments
    if len(list_args) < 1:  # No given parameters
        print(
            "Give one or more options to search hyperparameters:\n Flat_lvl1, Flat_lvl2, tgt_pred, tgt_tgt, pred_pred \n for runs containing pre give config and path to model")
        sys.exit(2)

    for i, conf in enumerate(list_args):
        if conf == "Flat_lvl1":
            hyp_search_lvl1_flatt()
            print("hyp_search_lvl1_flatt done")
            print("#" * 150)
            print("#" * 150)
            print("#" * 150)
            print("#" * 150)
        elif conf == "Flat_lvl2":
            hyp_search_lvl2_flatt()
            print("hyp_search_lvl2_flatt done")
            print("#" * 150)
            print("#" * 150)
            print("#" * 150)
            print("#" * 150)
            continue
        elif conf == "tgt_tgt":
            hyp_search_lvl2_target_target()
            print("hyp_search_lvl2_target_target done")
            print("#" * 150)
            print("#" * 150)
            print("#" * 150)
            print("#" * 150)
        elif conf == "pred_pred":
            print(list_args)
            hyp_search_lvl2_predicted_predicted(list_args[i + 1])
            print("hyp_search_lvl2_prediction_prediction done")
            print("#" * 150)
            print("#" * 150)
            print("#" * 150)
            print("#" * 150)
            continue
        else:
            print("Wrong input options to search hyperparameters:\n Flat_lvl1, Flat_lvl2, tgt_pred, tgt_tgt, pred_pred")
            return 1

    print("Search done for", list_args)


if __name__ == "__main__":
    main()
