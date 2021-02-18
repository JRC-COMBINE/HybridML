import datetime
import tensorflow as tf
import os


def prepare_tensorboard_callback(log_dir, model_name=None):
    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + model_name or ""
    log_path = os.path.join(log_dir, file_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0)
    return tensorboard_callback
