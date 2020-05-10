import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from optimizers.core import build_optimizer
from metrics.core import BinarySpecificity, BinarySensitivity, BinaryMCC, BinaryAccuracy


"""# Build model"""


def build_model(hparams):
    keras.backend.clear_session()
    rnn_input = layers.Input(shape=(29, 20))

    rnn_imd = rnn_input
    for i in range(hparams["rnn_layers"]):
        rnn_imd = layers.LSTM(
            units=hparams["rnn_units"],
            return_sequences=(i + 1 < hparams["rnn_layers"]),
            dropout=hparams["rnn_dropout"],
            activation="sigmoid"
        )(rnn_imd)

    imd = rnn_imd

    output_tf = layers.Dense(
        units=1,
        activation=tf.keras.activations.sigmoid
    )(imd)

    model = tf.keras.models.Model(inputs=rnn_input, outputs=output_tf)

    def compute_flood_loss(y_true, y_pred, b=0.05):
        loss = keras.losses.binary_crossentropy(y_true, y_pred)
        loss = keras.backend.abs(loss - b) + b
        return loss

    model.compile(
      optimizer=build_optimizer(
          optimizer_name=hparams["optimizer"],
          learning_rate=hparams["learning_rate"],
          decay=hparams["lr_decay"]
      ),
      loss=keras.losses.binary_crossentropy,
      metrics=[
        BinaryAccuracy(hparams["threshold"]),
        BinaryMCC(hparams["threshold"]),
        BinarySensitivity(hparams["threshold"]),
        BinarySpecificity(hparams["threshold"])
      ]
    )

    return model
