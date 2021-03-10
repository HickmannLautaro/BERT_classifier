#!/usr/bin/env python
# coding: utf-8


# -------- Load libraries ------- ###
# Load Huggingface transformers

import os
import sys

# Then what you need from tensorflow.keras
# And pandas for data import + sklearn because you always need sklearn
import tensorflow as tf
import yaml

from BERT_per_label import train_per_label
from BERT_per_lvl import run_experiment

# TODO Add docstrings

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
            if arguments['lable_type']=="per_label":
                train_per_label(arguments)
            else:
                run_experiment(arguments)


if __name__ == "__main__":
    main()
