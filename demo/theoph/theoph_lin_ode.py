import sys
import os
import time

sys.path.append(os.path.join(os.getcwd()))

from HybridML import Project  # noqa: E402
from plot_animation import plot_in_one  # noqa: E402

import tensorflow as tf  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def get_random_number():
    """For explanation concerning the algorithm: https://xkcd.com/221/"""
    return 4


np.random.seed(get_random_number())
tf.random.set_seed(get_random_number())


class PredictionCallback(tf.keras.callbacks.Callback):
    """
    This callback uses the model during training and predicts the val_data, based on the current state of the model.
    Thus the progress of the model during training can be visualized.
    """

    def __init__(self, val_data=None, frequency=1):
        self.val_data = val_data
        self.frequency = frequency
        self.log = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs={}):
        if self.val_data is None:
            return
        if epoch % self.frequency != 0:
            return

        y_pred = self.model.predict(self.val_data)
        self.log.append(y_pred)
        self.epochs.append(epoch)


def split_data(X, y, i=2):
    """
    The data consists of a list of samples for each input. To split it, each list of inputs has to be split.
    """
    return [x[:i] for x in X], y[:i]
    # return [X[0][:i], X[1][:i], X[2][:i]], y[:i]


def ignore_additional_outputs_loss(y_true, y_pred):
    """
    Loss for additional outputs is 0, else MSE.
    """
    res = []
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        return tf.math.reduce_mean((y_true - y_pred) * 0)
    for elm in zip(y_true, y_pred):
        res.append(tf.keras.losses.mse(elm[0], elm[1]))
    res = tf.math.reduce_mean(res)
    return res


def train(trained_model, input_data, validation_data, progress_check_frequency, train_epochs):
    """
    Trains model and measures parameters and fit every few epochs
    """

    # Prepare callback for measuring progress
    cb = PredictionCallback(val_data=validation_data, frequency=progress_check_frequency)

    # Prepare output: Append additional outputs
    output_data_w_add_outputs = [y] + [np.zeros((y.shape[0], 1))] * (len(trained_model.outputs) - 1)

    # Fit the model
    hist = trained_model.fit(
        input_data,
        output_data_w_add_outputs,
        validation_split=0.0,  # (just for this experiment, use all data for training)
        shuffle=False,
        epochs=train_epochs,
        callbacks=[cb],
    )
    return hist, cb.log, cb.epochs


def plot_training_parameters(hist, train_log):
    """
    Plots how estimated parameters of ODE evolved while training
    """
    h = hist.history
    fig, axs = plt.subplots(5, 1, sharex=True)
    # Plot loss and val loss
    ax = axs[0]
    ax.plot(h["loss"])
    if "val_loss" in h:
        ax.plot(h["val_loss"])
    ax.legend(["loss", "val_loss"])

    # Plot the results of the ks for a validation sample over epochs
    names = ["ka", "k12", "k21", "k10"]
    log2 = [log_entry[1:] for log_entry in train_log]
    np_log = np.array(log2)
    np_log = np_log[:, :, 0, 0]
    x = np.arange(0, len(h["loss"]), len(h["loss"]) // len(np_log[:, 0]))
    for i in range(np_log.shape[1]):
        ax = axs[i + 1]
        ax.plot(x, np_log[:, i])
        ax.legend([names[i]])

    plt.show()


if __name__ == "__main__":
    # Load project
    base_dir = os.path.join(".", "demo")
    projects_path = os.path.join(base_dir, "projects")
    project = Project.open_create(projects_path, "theoph_lin_ode")

    # Generate the model out of the model definitions stored in the demo project
    models = project.generate_models()
    models = [model.additional_outputs_model for model in models]

    # Load data
    data_sources = project.load_data_sources()
    ds = data_sources["theoph"]
    X_, y_ = ds.get_train_data()
    X, y = split_data(X_, y_, i=12)  # split data
    training_time = []
    histories = []
    for model in models:
        start_time = time.time()
        # Compile the model
        model.compile(optimizer="adam", loss=ignore_additional_outputs_loss)
        model.summary()

        # Train
        history, log, epochs = train(
            trained_model=model, input_data=X, validation_data=X, progress_check_frequency=1, train_epochs=15
        )
        training_time.append(time.time() - start_time)
        histories.append(history)
    project.save_models()
    names = ["closed form", "casadi", "tensorflow"]
    for i, r in zip(names, training_time):
        print(f"The model using the {i} solution took {r:.2f}s to train")
    ########################
    # plot
    for hist in histories:
        plt.plot(hist.history["loss"])
    plt.legend(names)
    plt.show()
    ##########################
    # Save results of training (for animation plotting)
    log_prediction = np.array([item[0] for item in log])
    t = X[2]
    np.savetxt("epochs.np_array", epochs)
    np.savetxt("y_true.np_array", y)
    np.savetxt("t.np_array", X[2])
    np.savetxt("data.np_array", log_prediction.reshape((-1,)))

    # Plot development of parameters and animation of resulting concentration curves
    plot_training_parameters(history, log)
    plot_in_one(log_prediction, y, epochs, t)
