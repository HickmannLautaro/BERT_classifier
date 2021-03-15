#!/usr/bin/env python
# coding: utf-8
import os
import sys
import tensorflow as tf
import yaml
from BERT_per_label import train_per_label
from BERT_per_lvl import run_experiment


def main():
    """
    Unifies both training scripts and avoids cyclic import problems.
    Calls the corresponding train function of either BERT_per_label.py or BERT_per_lvl.py
    Run `Train.py` with the desired configurations from the `Configs` folder to train the desired models, e.g. `python Train.py ./Configs/amazon_config_lvl1_bert-base-uncased.yaml ./Configs/amazon_config_lvl2_per_label.yaml`

    """
    print("Tensorflow version: ", tf.__version__)
    # rtx 3080 tf 2.4.0-rc4 bug
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoids Hugging Face process forking bug https://github.com/ThilinaRajapakse/simpletransformers/issues/515

    list_args = sys.argv[1:]  # Gets command line arguments
    if len(list_args) < 1:
        print("Config missing")
        sys.exit(2)

    for conf in list_args:  # For each configuration given as command line arguments do the experiments
        with open(conf) as f:
            arguments = yaml.load(f, Loader=yaml.FullLoader)
        for i in range(arguments['repetitions']):
            if arguments['lable_type'] == "per_label":
                train_per_label(arguments)
            else:
                run_experiment(arguments)


if __name__ == "__main__":
    main()
