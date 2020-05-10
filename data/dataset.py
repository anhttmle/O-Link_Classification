import tensorflow as tf


"""# Build Tensorflow Dataset"""


def build_train_ds(x, y, hparams):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(10000).batch(hparams["batch_size"])
    return ds


def build_test_ds(x, y):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(32)
    return ds
