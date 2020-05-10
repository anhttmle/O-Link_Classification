from constant.index import MetricIdx
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

"""# Build Logging Funtion"""


def log_result(train_result, dev_result, test_result):
    print("\nDATASET, ACC, MCC, SEN, SPEC")

    print("TRAIN: {}, {}, {}, {}".format(
        train_result[MetricIdx.ACC],
        train_result[MetricIdx.MCC],
        train_result[MetricIdx.SEN],
        train_result[MetricIdx.SPEC])
    )

    print("DEV: {}, {}, {}, {}".format(
        dev_result[MetricIdx.ACC],
        dev_result[MetricIdx.MCC],
        dev_result[MetricIdx.SEN],
        dev_result[MetricIdx.SPEC])
    )

    print("TEST: {}, {}, {}, {}".format(
        test_result[MetricIdx.ACC],
        test_result[MetricIdx.MCC],
        test_result[MetricIdx.SEN],
        test_result[MetricIdx.SPEC])
    )


def write(log_dir, ds_name, result, hparams):
    session_log_dir = log_dir
    with tf.summary.create_file_writer(session_log_dir).as_default():
        hp.hparams(hparams=hparams)
        tf.summary.scalar(name="{}_acc".format(ds_name), data=result[MetricIdx.ACC], step=1)
        tf.summary.scalar(name="{}_mcc".format(ds_name), data=result[MetricIdx.MCC], step=1)
        tf.summary.scalar(name="{}_sen".format(ds_name), data=result[MetricIdx.SEN], step=1)
        tf.summary.scalar(name="{}_spec".format(ds_name), data=result[MetricIdx.SPEC], step=1)