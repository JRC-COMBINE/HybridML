import time

import numpy as np
import tensorflow as tf


class PredictionCallback(tf.keras.callbacks.Callback):
    """
    This callback uses the model during training and predicts the val_data, based on the current state of the model.
    Thus the progress of the model during training can be visualized.
    """

    def __init__(self, data_to_predict=None, frequency=1, prediction_model=None):
        self.prediction_model = prediction_model
        self.data_to_predict = data_to_predict
        self.frequency = frequency
        self.log = []
        self.epochs = []

    def perform_log(self, epoch):
        print("Logging epoch", epoch)
        model_for_prediction = self.prediction_model or self.model
        y_pred = model_for_prediction.predict(self.data_to_predict)
        self.log.append(y_pred)
        self.epochs.append(epoch)

    def on_epoch_end(self, epoch, logs={}):
        if self.data_to_predict is None:
            return
        if epoch % self.frequency != 0:
            return
        self.perform_log(epoch)


def train(
    model,
    y,
    input_data,
    visualization_data,
    progress_check_frequency,
    train_epochs,
    validation_split,
    additional_callbacks=None,
):
    """
    Trains model and measures parameters and fit every few epochs
    """
    callbacks = []

    pred_callback = None
    if progress_check_frequency != 0:
        assert train_epochs % progress_check_frequency == 0, (
            f"Number of epochs ({train_epochs}) must be divisible by "
            f"progress check frequency ({progress_check_frequency})!"
        )

        # Prepare callback for measuring progress
        pred_callback = PredictionCallback(
            data_to_predict=visualization_data,
            frequency=progress_check_frequency,
            prediction_model=model.additional_outputs_model,
        )
        callbacks += [pred_callback]

    if additional_callbacks is not None and type(additional_callbacks) == list and len(additional_callbacks) > 0:
        callbacks += additional_callbacks

    start = time.time()
    # Fit the model
    history_container = model.fit(
        input_data, y, validation_split=validation_split, shuffle=False, epochs=train_epochs, callbacks=callbacks
    )
    end = time.time()
    print(f"Training took {end-start:.2f}s")
    history = history_container.history
    if pred_callback:
        pred_callback.perform_log(history.epoch[-1])
        return history, pred_callback.log, pred_callback.epochs
    else:
        return history, [], []


def train_complex(
    model,
    x,
    y,
    x_full,
    y_full,
    val_indices,
    test_indices,
    progress_check_frequency,
    train_epochs,
    additional_callbacks=None,
):
    """
    Trains model and measures parameters and fit every few epochs
    """
    assert train_epochs % progress_check_frequency == 0, (
        f"Number of epochs ({train_epochs}) must be divisible by "
        f"progress check frequency ({progress_check_frequency})!"
    )

    # Prepare visualization data for training
    num_curves_plotted = 6
    callbacks = []
    non_test_indices = np.setdiff1d(np.arange(len(y)), test_indices)
    train_indices = np.setdiff1d(non_test_indices, val_indices)
    train_indices_plot = np.random.choice(train_indices, num_curves_plotted, replace=False)
    visualization_data_train = [x_[train_indices_plot] for x_ in x_full]
    pred_callback_train = PredictionCallback(
        data_to_predict=visualization_data_train,
        frequency=progress_check_frequency,
        prediction_model=model.additional_outputs_model,
    )
    callbacks.append(pred_callback_train)

    # Prepare visualization data for validation
    val_indices_plot = np.random.choice(val_indices, num_curves_plotted, replace=False)
    visualization_data_val = [x_[val_indices_plot] for x_ in x_full]
    pred_callback_val = PredictionCallback(
        data_to_predict=visualization_data_val,
        frequency=progress_check_frequency,
        prediction_model=model.additional_outputs_model,
    )
    callbacks.append(pred_callback_val)

    # Add any additional callbacks
    if additional_callbacks is not None and type(additional_callbacks) == list and len(additional_callbacks) > 0:
        callbacks += additional_callbacks

    # Split into training and validation data (and test data)
    x_val = [x_[val_indices] for x_ in x]
    y_val = [y[val_indices]]

    x_train = [x_[train_indices] for x_ in x]
    y_train = [y[train_indices]]

    # Start measuring time
    start = time.time()

    # Fit the model
    hist = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), shuffle=False, epochs=train_epochs, callbacks=callbacks
    )

    # Stop timer
    end = time.time()
    print(f"Training took {end-start:.2f}s")

    # Signal end of training to prediction callbacks
    pred_callback_train.on_epoch_end(hist.epoch[-1])
    pred_callback_val.on_epoch_end(hist.epoch[-1])

    return (
        hist,
        train_indices,
        train_indices_plot,
        val_indices_plot,
        pred_callback_train.log,
        pred_callback_train.epochs,
        pred_callback_val.log,
        pred_callback_val.epochs,
    )
