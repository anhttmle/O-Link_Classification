import tensorflow as tf


def build_callbacks(log_dir, patience=5, min_diff=1e-5):
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=min_diff,
        patience=patience,
        verbose=2,
        mode='auto',
        baseline=None,
        restore_best_weights=False
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None,
    )

    return [tensorboard_cb]