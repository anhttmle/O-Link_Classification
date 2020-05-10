import tensorflow.keras as keras


"""# Build Optimizer"""


def build_optimizer(optimizer_name, learning_rate, decay):
    if optimizer_name == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, decay=decay)
    elif optimizer_name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate, decay=decay)
    elif optimizer_name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate, decay=decay)
    elif optimizer_name == "adagrad":
        return keras.optimizers.Adagrad(learning_rate=learning_rate, decay=decay)
    elif optimizer_name == "adadelta":
        return keras.optimizers.Adadelta(learning_rate=learning_rate, decay=decay)
    elif optimizer_name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate, decay=decay)
    elif optimizer_name == "adamax":
        return keras.optimizers.Adamax(learning_rate=learning_rate, decay=decay)
    elif optimizer_name == "nadam":
        return keras.optimizers.Nadam(learning_rate=learning_rate, decay=decay)
    else:
        raise NameError("NO SUPPORT THE OPTIMIZER NAME!!!")