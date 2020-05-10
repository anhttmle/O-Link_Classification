# -*- coding: utf-8 -*-
import numpy as np

import os
import shutil

from data.loader import download_from_shareable_link, parse_csv_data
from constant.index import DataIdx
from process.core import experiment
from absl import flags, app

"""# MAIN FUNCTION"""

FLAGS = flags.FLAGS

flags.DEFINE_enum("optimizer", "nadam", ["adam", "rmsprop", "sgd", "adamax", "adadelta", "nadam"], "Name of optimizer")
flags.DEFINE_integer("batch_size", 128, "Batch size of traning data")#1
flags.DEFINE_float("learning_rate", 0.0001, "learning rate of optimizer")
flags.DEFINE_integer("n_epoch", 2, "Number of training epoch")#100
flags.DEFINE_integer("rnn_layers", 1, "Number of LSTM layer")
flags.DEFINE_integer("rnn_units", 2, "Number of RNN units")#512
flags.DEFINE_float("rnn_dropout", 0.025, "Dropout of LSTM")
flags.DEFINE_float("threshold", 0.5, "Threshold of last layer")
flags.DEFINE_enum("is_tuning", "Y", ["Y", "N"], "running for auto hyper param tuning or specific setting")


def main(argv):
    if os.path.isdir("k_fold"):
        shutil.rmtree("k_fold")

    if os.path.isdir("__MACOSX"):
        shutil.rmtree("__MACOSX")

    data_path = "https://drive.google.com/open?id=1PJgI0RKLnfTuWYpVg-Ry__USVC1WGdwo"
    download_from_shareable_link(url=data_path, destination="data.zip")
    os.system("unzip data.zip")
    fold_paths = []
    for i in range(5):
        fold_paths.append(
            (
                ("k_fold/train_features_{}.csv".format(i + 1), "k_fold/train_labels_{}.csv".format(i + 1)),
                ("k_fold/dev_features_{}.csv".format(i + 1), "k_fold/dev_labels_{}.csv".format(i + 1)),
            )
        )

    train_path = ("k_fold/train_features.csv", "k_fold/train_labels.csv")
    test_path = ("k_fold/test_features.csv", "k_fold/test_labels.csv")

    log_dir = os.path.join(os.getcwd(),"log", "hparam_tuning")

    n_experiments = 1000

    threshold = 0.5
    batch_size = [1, 2, 4, 8, 16, 32, 64]
    learning_rate = (0.0001, 0.5)
    n_epoch = 100
    optimizer = ["adam", "rmsprop", "sgd", "adamax", "adadelta", "nadam"]
    rnn_layers = [1, 2, 3]
    rnn_units = [32, 64, 128, 256, 512, 1024]
    rnn_dropout = [1e-5, 0.5]
    lr_decay = 0#1e-6

    hparams = {
        "batch_size": np.array(batch_size)[np.random.randint(low=0, high=len(batch_size), size=(n_experiments,))],
        "learning_rate": np.logspace(np.log10(learning_rate[0]), np.log10(learning_rate[1]), base=10,
                                     num=n_experiments),
        "optimizer": np.array(optimizer)[np.random.randint(low=0, high=len(optimizer), size=(n_experiments,))],
        "rnn_layers": np.array(rnn_layers)[np.random.randint(low=0, high=len(rnn_layers), size=(n_experiments,))],
        "rnn_units": np.array(rnn_units)[np.random.randint(low=0, high=len(rnn_units), size=(n_experiments,))],
        "rnn_dropout": np.random.random(size=(n_experiments,)) * (rnn_dropout[1] - rnn_dropout[0]) + rnn_dropout[
            0]
    }

    configs = [{"hparams": {
        "threshold": threshold,
        "batch_size": hparams["batch_size"][i],
        "learning_rate": hparams["learning_rate"][i],
        "n_epoch": n_epoch,
        "optimizer": hparams["optimizer"][i],
        "rnn_layers": hparams["rnn_layers"][i],
        "rnn_units": hparams["rnn_units"][i],
        "rnn_dropout": hparams["rnn_dropout"][i],
        "lr_decay": lr_decay
    }} for i in range(n_experiments)]

    if FLAGS.is_tuning == "Y":
        experiment(
            configs=configs,
            fold_paths=fold_paths,
            train_path=train_path,
            test_path=test_path,
            log_dir=log_dir
        )
    else:
        config = {
            "hparams": {
                "threshold": FLAGS.threshold,
                "batch_size": FLAGS.batch_size,
                "learning_rate": FLAGS.learning_rate,
                "n_epoch": FLAGS.n_epoch,
                "optimizer": FLAGS.optimizer,
                "rnn_layers": FLAGS.rnn_layers,
                "rnn_units": FLAGS.rnn_units,
                "rnn_dropout": hparams["rnn_dropout"][i],
                "lr_decay": lr_decay
              }
        }

        experiment(
            configs=[config],
            fold_paths=fold_paths,
            train_path=train_path,
            test_path=test_path,
            log_dir=log_dir
        )


if __name__ == "__main__":
    app.run(main)

