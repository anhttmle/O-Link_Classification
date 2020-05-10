import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tensorboard.plugins.hparams import api as hp
import os
import datetime

from models.core import build_model
from data.dataset import build_train_ds, build_test_ds
from constant.index import DataIdx, MetricIdx
from util.log import log_result, write
from callbacks.core import build_callbacks
from metrics.core import BinaryAccuracy, BinaryMCC, BinarySensitivity, BinarySpecificity
from data.loader import parse_csv_data


"""# Build single training process"""


def train(config, train_ds, test_ds, need_summary=False, need_threshold=True):
    hparams = config["hparams"]
    n_epoch = hparams["n_epoch"]
    class_weight = config["class_weight"]
    verbose = config["verbose"]

    callbacks = []

    if need_summary:
        callbacks = build_callbacks(log_dir=config["log_dir"])

    model = build_model(hparams)
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=n_epoch,
        verbose=verbose,
        class_weight=class_weight,
        shuffle=True,
        callbacks=callbacks
    )

    if need_threshold:
        train_result = evaluate(model, train_ds)
        test_result = evaluate(model, test_ds)
    else:
        train_result = evaluate_on_threshold(model, train_ds, threshold=hparams["threshold"])
        test_result = evaluate_on_threshold(model, test_ds, threshold=hparams["threshold"])
    return train_result, test_result


"""# Build evaluation process"""


def evaluate(model, test_ds):
    results = [[y, model.predict(x)] for x, y in test_ds]
    y_true = np.concatenate([results[i][0] for i in range(len(results))], axis=0)
    y_pred = np.concatenate([results[i][1] for i in range(len(results))], axis=0).flatten()
    thresholds = np.arange(start=0.00, stop=1, step=0.01)

    results = {}
    for i in range(len(thresholds)):
        acc = BinaryAccuracy(threshold=thresholds[i])(y_true=y_true, y_pred=y_pred)
        mcc = BinaryMCC(threshold=thresholds[i])(y_true=y_true, y_pred=y_pred)
        sen = BinarySensitivity(threshold=thresholds[i])(y_true=y_true, y_pred=y_pred)
        spec = BinarySpecificity(threshold=thresholds[i])(y_true=y_true, y_pred=y_pred)
        results[thresholds[i]] = np.array([acc, mcc, sen, spec])

    return results


def evaluate_on_threshold(model, test_ds, threshold=0.5):
    results = [[y, model.predict(x)] for x, y in test_ds]
    y_true = np.concatenate([results[i][0] for i in range(len(results))], axis=0)
    y_pred = np.concatenate([results[i][1] for i in range(len(results))], axis=0).flatten()

    acc = BinaryAccuracy(threshold=threshold)(y_true=y_true, y_pred=y_pred)
    mcc = BinaryMCC(threshold=threshold)(y_true=y_true, y_pred=y_pred)
    sen = BinarySensitivity(threshold=threshold)(y_true=y_true, y_pred=y_pred)
    spec = BinarySpecificity(threshold=threshold)(y_true=y_true, y_pred=y_pred)

    return {threshold: np.array([acc, mcc, sen, spec])}


def get_best_threshold(results):
    best_mcc = -1
    best_threshold = None
    for threshold, result in results.items():
        mcc = result[MetricIdx.MCC]
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    return best_threshold


def avg_evaluate(all_results):
    n_fold = len(all_results)
    final_result = {}
    if len(all_results) > 0:
        for threshold, result in all_results[0].items():
            final_result[threshold] = np.array([0, 0, 0, 0])

    for i, results in enumerate(all_results):
        for threshold, result in results.items():
            final_result[threshold] = final_result[threshold] + result/n_fold

    return final_result


"""# Build k-fold training process"""


def k_fold_experiment(config, fold_paths, train_path, test_path):
    print(config)
    hparams = config["hparams"]

    train_results = []
    dev_results = []
    for fold_index, fold_path in enumerate(fold_paths):
        print("FOLD: {}".format(fold_index + 1))
        fold_train_path, fold_dev_path = fold_path

        features, labels = parse_csv_data(fold_train_path[DataIdx.FEATURE], fold_train_path[DataIdx.LABEL])
        train_ds = build_train_ds(
            x=features,
            y=labels,
            hparams=hparams
        )

        features, labels = parse_csv_data(fold_dev_path[DataIdx.FEATURE], fold_dev_path[DataIdx.LABEL])
        dev_ds = build_test_ds(
            x=features,
            y=labels,
        )

        train_result, dev_result = train(
            config=config,
            train_ds=train_ds,
            test_ds=dev_ds,
            need_threshold=False
        )

        train_results.append(train_result)
        dev_results.append(dev_result)

    train_result = avg_evaluate(train_results)
    dev_result = avg_evaluate(dev_results)

    hparams["threshold"] = get_best_threshold(dev_result)
    train_result = train_result[hparams["threshold"]]
    dev_result = dev_result[hparams["threshold"]]

    # Train on all & evaluate on test set
    features, labels = parse_csv_data(train_path[DataIdx.FEATURE], train_path[DataIdx.LABEL])
    train_ds = build_train_ds(
        x=features,
        y=labels,
        hparams=hparams,
    )

    features, labels = parse_csv_data(test_path[DataIdx.FEATURE], test_path[DataIdx.LABEL])
    test_ds = build_test_ds(
        x=features,
        y=labels,
    )

    _, test_result = train(config=config, train_ds=train_ds, test_ds=test_ds, need_summary=True)
    test_result = test_result[hparams["threshold"]]
    return train_result, dev_result, test_result


"""# Build Hyperparameter experiment process"""


def experiment(configs, fold_paths, train_path, test_path, log_dir):
    for index, config in enumerate(configs):
        session_log_dir = os.path.join(log_dir, "session_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        print("\nEXPERIMENT: {}/{}".format(index+1, len(configs)))
        print(config)
        config["log_dir"] = session_log_dir
        config["class_weight"] = {
              0: 1,
              1: 1
        }
        config["verbose"] = 2
        train_result, dev_result, test_result = k_fold_experiment(
            config=config,
            fold_paths=fold_paths,
            train_path=train_path,
            test_path=test_path
        )

        log_result(train_result, dev_result, test_result)

        write(session_log_dir, "train", train_result, config["hparams"])
        write(session_log_dir, "dev", dev_result, config["hparams"])
        write(session_log_dir, "test", test_result, config["hparams"])