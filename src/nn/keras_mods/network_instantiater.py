import tensorflow as tf
import os
# import numpy as np
# from src.nn.keras_mods.mann_keras import MANN
from tensorflow.keras import optimizers, Model, losses, datasets

import json
from termcolor import colored

from zipfile import ZipFile


def InstantiateNetworkFromConfig(config_file_path, network):
    """

    Instantates a network model from a config file

    Arguments:
        config_file_path {path / dict} -- configuration data, either a path to a json file or directly a dictionary
        network -- network model class. Should be initializable by a dict (the config)

    Returns:
        (instance of network, config_data as dict)
    """
    if type(config_file_path) is dict:
        config = config_file_path
    elif os.path.isfile(config_file_path):
        config = json.load(config_file_path)
    else:
        raise Exception("Config File not extisting: %s" % config_file_path)

    net = network(config)

    if "checkpoint" in config and config["checkpoint"] != "":
        if not os.path.isdir(config["checkpoint"]):
            print(
                colored("Checkpoint is defined, but not existing: %s \nNo weights were loaded." % config["checkpoint"],
                        "red"))
        else:
            network.load_weights(config["checkpoint"])

    return net, config


# def TrainNetwork(network: tf.keras.Model, training_data_path, config):
def TrainNetwork(network: tf.keras.Model, normalized_x, normalized_y, config):
    """
    Trains an already initialized network model.
    Loads the training data from the training data path. Expects a zip file containing:
        - data.npz [Xun, Yun] optionally as well Pun
        - Xmean.bin, Xstd.bin, Ymean.bin, Ystd.bin as containing data

    Arguments:
        network {tf.keras.Model} -- network model (e.g. pfnn)
        training_data_path {path} -- zipped training data
        config {dict / str} -- json file or already loaded dict

    """

    if type(config) is str:
        if os.path.isfile(config):
            config = json.load(config)
        else:
            raise Exception("Config File is not existing: %s" % config)

    X = normalized_x
    Y = normalized_y

    if "optimizer" in config:
        if type(config["optimizer"]) is str:
            if config["optimizer"] == "Adam":
                lr = config["learning_rate"]
                optimizer = optimizers.Adam(lr)
        else:
            optimizer = config["optimzier"]

    loss = config["loss"]
    epochs = config["epochs"]
    batchsize = config["batchsize"]

    network.compile(optimizer=optimizer, loss=loss)
    network.fit(X, Y, epochs=epochs, batch_size=batchsize)
