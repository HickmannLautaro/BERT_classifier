import yaml
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
from Train_BERT import run_experiment
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
import sys


def hyp_search_lvl1_flatt():
    HP_MAX_LENGTH = hp.HParam('max_length', hp.Discrete([64, 100, 256, 512]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([60, 20, 40, 5]))

    METRIC_ACCURACY = 'accuracy_score'
    METRIC_f1 = 'f1_score'

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
    model_name = arguments['model_name']

    lvl = arguments['lvl']
    data_path = arguments['data_path']

    with tf.summary.create_file_writer("hyperparameters_search/" + model_name + "/" + data_path + "/lvl" + str(
            lvl) + '/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_MAX_LENGTH, HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy_score'),
                     hp.Metric(METRIC_f1, display_name='f1_score')],
        )

    def train_test_model_lvl1_flatt(hparams, arguments):
        arguments['max_length'] = hparams[HP_MAX_LENGTH]  # arguments['max_length']
        arguments['batch_size'] = hparams[HP_BATCH_SIZE]  # arguments['epochs']

        f1_score_1, accuracy_score_1 = run_experiment(arguments, hyp_search=True, )
        f1_score_2, accuracy_score_2 = run_experiment(arguments, hyp_search=True, )

        return np.mean([f1_score_1,f1_score_2]), np.mean([accuracy_score_1, accuracy_score_2])

    def run(run_dir, hparams, arguments):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            f1_score, accuracy_score = train_test_model_lvl1_flatt(hparams, arguments)
            tf.summary.scalar(METRIC_ACCURACY, accuracy_score, step=1)
            tf.summary.scalar(METRIC_f1, f1_score, step=1)

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
                    lvl) + '/hparam_tuning/' + run_name, hparams, arguments)
            except tf.errors.ResourceExhaustedError as e:
                print("Out of memory")

            session_num += 1


def hyp_search_lvl2_flatt():
    HP_MAX_LENGTH = hp.HParam('max_length', hp.Discrete([64, 100, 256, 512]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([60, 20, 40, 5]))

    METRIC_ACCURACY = 'accuracy_score'
    METRIC_f1 = 'f1_score'

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
    model_name = arguments['model_name']

    lvl = arguments['lvl']
    data_path = arguments['data_path']

    with tf.summary.create_file_writer("hyperparameters_search/" + model_name + "/" + data_path + "/lvl" + str(
            lvl) + '/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_MAX_LENGTH, HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy_score'),
                     hp.Metric(METRIC_f1, display_name='f1_score')],
        )

    def train_test_model_lvl1_flatt(hparams, arguments):
        arguments['max_length'] = hparams[HP_MAX_LENGTH]  # arguments['max_length']
        arguments['batch_size'] = hparams[HP_BATCH_SIZE]  # arguments['epochs']

        f1_score_1, accuracy_score_1 = run_experiment(arguments, hyp_search=True, )
        f1_score_2, accuracy_score_2 = run_experiment(arguments, hyp_search=True, )

        return np.mean([f1_score_1, f1_score_2]), np.mean([accuracy_score_1, accuracy_score_2])

    def run(run_dir, hparams, arguments):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            f1_score, accuracy_score = train_test_model_lvl1_flatt(hparams, arguments)
            tf.summary.scalar(METRIC_ACCURACY, accuracy_score, step=1)
            tf.summary.scalar(METRIC_f1, f1_score, step=1)

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
                    lvl) + '/hparam_tuning/' + run_name, hparams, arguments)
            except tf.errors.ResourceExhaustedError as e:
                print("Out of memory")

            session_num += 1


def hyp_search_lvl2_target_predicted(path_predicted):
    HP_MAX_LENGTH = hp.HParam('max_length', hp.Discrete([64, 100, 256, 512]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([60, 20, 40, 5]))

    METRIC_ACCURACY = 'accuracy_score'
    METRIC_f1 = 'f1_score'

    arguments = {'model_name': 'bert-base-uncased',
                 'max_length': 100,
                 'epochs': 40,  #
                 'batch_size': 40,
                 'repetitions': 1,
                 'data_path': 'amazon',
                 'lvl': 2,
                 'labels': [['Target', 'Cat1']],
                 'test_labels': [[path_predicted]],
                 'hierar': 'hierarchical',
                 'lable_type': 'Target',
                 'test_labels_type': 'Predicted'}
    model_name = arguments['model_name']

    lvl = arguments['lvl']
    data_path = arguments['data_path']

    with tf.summary.create_file_writer("hyperparameters_search/" + model_name + "/" + data_path + "/lvl" + str(
            lvl) + '/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_MAX_LENGTH, HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy_score'),
                     hp.Metric(METRIC_f1, display_name='f1_score')],
        )

    def train_test_model_lvl1_flatt(hparams, arguments):
        arguments['max_length'] = hparams[HP_MAX_LENGTH]  # arguments['max_length']
        arguments['batch_size'] = hparams[HP_BATCH_SIZE]  # arguments['epochs']

        f1_score_1, accuracy_score_1 = run_experiment(arguments, hyp_search=True, )
        f1_score_2, accuracy_score_2 = run_experiment(arguments, hyp_search=True, )

        return np.mean([f1_score_1, f1_score_2]), np.mean([accuracy_score_1, accuracy_score_2])

    def run(run_dir, hparams, arguments):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            f1_score, accuracy_score = train_test_model_lvl1_flatt(hparams, arguments)
            tf.summary.scalar(METRIC_ACCURACY, accuracy_score, step=1)
            tf.summary.scalar(METRIC_f1, f1_score, step=1)

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
                    lvl) + '/hparam_tuning/' + run_name, hparams, arguments)
            except tf.errors.ResourceExhaustedError as e:
                print("Out of memory")

            session_num += 1


def hyp_search_lvl2_target_target():
    HP_MAX_LENGTH = hp.HParam('max_length', hp.Discrete([64, 100, 256, 512]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([60, 20, 40, 5]))

    METRIC_ACCURACY = 'accuracy_score'
    METRIC_f1 = 'f1_score'

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
    model_name = arguments['model_name']

    lvl = arguments['lvl']
    data_path = arguments['data_path']

    with tf.summary.create_file_writer("hyperparameters_search/" + model_name + "/" + data_path + "/lvl" + str(
            lvl) + '/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_MAX_LENGTH, HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy_score'),
                     hp.Metric(METRIC_f1, display_name='f1_score')],
        )

    def train_test_model_lvl1_flatt(hparams, arguments):
        arguments['max_length'] = hparams[HP_MAX_LENGTH]  # arguments['max_length']
        arguments['batch_size'] = hparams[HP_BATCH_SIZE]  # arguments['epochs']

        f1_score_1, accuracy_score_1 = run_experiment(arguments, hyp_search=True, )
        f1_score_2, accuracy_score_2 = run_experiment(arguments, hyp_search=True, )

        return np.mean([f1_score_1, f1_score_2]), np.mean([accuracy_score_1, accuracy_score_2])

    def run(run_dir, hparams, arguments):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            f1_score, accuracy_score = train_test_model_lvl1_flatt(hparams, arguments)
            tf.summary.scalar(METRIC_ACCURACY, accuracy_score, step=1)
            tf.summary.scalar(METRIC_f1, f1_score, step=1)

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
                    lvl) + '/hparam_tuning/' + run_name, hparams, arguments)
            except tf.errors.ResourceExhaustedError as e:
                print("Out of memory")

            session_num += 1


def hyp_search_lvl2_predicted_predicted(path_predicted):
    HP_MAX_LENGTH = hp.HParam('max_length', hp.Discrete([64, 100, 256, 512]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([60, 20, 40, 5]))

    METRIC_ACCURACY = 'accuracy_score'
    METRIC_f1 = 'f1_score'

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
    model_name = arguments['model_name']

    lvl = arguments['lvl']
    data_path = arguments['data_path']

    with tf.summary.create_file_writer("hyperparameters_search/" + model_name + "/" + data_path + "/lvl" + str(
            lvl) + '/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_MAX_LENGTH, HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy_score'),
                     hp.Metric(METRIC_f1, display_name='f1_score')],
        )

    def train_test_model_lvl1_flatt(hparams, arguments):
        arguments['max_length'] = hparams[HP_MAX_LENGTH]  # arguments['max_length']
        arguments['batch_size'] = hparams[HP_BATCH_SIZE]  # arguments['epochs']

        f1_score_1, accuracy_score_1 = run_experiment(arguments, hyp_search=True, )
        f1_score_2, accuracy_score_2 = run_experiment(arguments, hyp_search=True, )

        return np.mean([f1_score_1, f1_score_2]), np.mean([accuracy_score_1, accuracy_score_2])

    def run(run_dir, hparams, arguments):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            f1_score, accuracy_score = train_test_model_lvl1_flatt(hparams, arguments)
            tf.summary.scalar(METRIC_ACCURACY, accuracy_score, step=1)
            tf.summary.scalar(METRIC_f1, f1_score, step=1)

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
                    lvl) + '/hparam_tuning/' + run_name, hparams, arguments)
            except tf.errors.ResourceExhaustedError as e:
                print("Out of memory")

            session_num += 1


def main():
    print("Tensorflow version: ", tf.__version__)

    # rtx 3080 tf 2.4.0-rc4 bug
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    list_args = sys.argv[1:]
    if len(list_args) < 1:
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
        elif conf == "tgt_pred":
            hyp_search_lvl2_target_predicted(list_args[i+1])
            print("hyp_search_lvl2_target_predicted done")
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
            hyp_search_lvl2_predicted_predicted(list_args[i+1])
            print("hyp_search_lvl2_target_target done")
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
