import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def generate_visualization_data(X, y, n):
    """Randomly selects n samples to be used for visualization."""
    visualization_index = np.random.choice(len(X[0]), n, replace=False)
    visualization_data = [x[visualization_index] for x in X]
    vis_y = y[visualization_index]
    return visualization_data, vis_y


def plot_training_parameters(hist, train_log, progress_check_frequency, write_plot=None):
    """
    Plots how estimated parameters of ODE evolved while training
    """
    h = hist.history
    fig, axs = plt.subplots(5, 1, sharex=True)
    # Plot loss and val loss
    ax = axs[0]
    ax.set_yscale("log")
    ax.plot(h["loss"])
    if "val_loss" in h:
        ax.plot(h["val_loss"])
    ax.legend(["loss", "val_loss"])

    # Plot the results of the ks for a validation sample over epochs
    names = ["ka", "k10", "k12", "k21"]
    log2 = [log_entry[1:] for log_entry in train_log]
    np_log = np.array(log2)
    np_log = np_log[:, :, :, 0]
    # np_log shape: (num_progress_checks, num_params = 4, num_patients = 10)

    # Find out the number of epochs up to which progress checks exist. If training was aborted using early stopping,
    # that number might not be equal to the actual number of trained epochs
    n_epochs = len(h["loss"])
    freq = progress_check_frequency
    x = np.arange(0, n_epochs, freq)
    x = np.append(x, n_epochs)
    # x = np.linspace(0, len(h["loss"]), len(h["loss"]))
    for i in range(np_log.shape[1]):
        ax = axs[i + 1]
        ax.plot(x, np_log[:, i])
        ax.legend([names[i]])

    # Either write or show plot
    if write_plot is None:
        plt.show()
    else:
        plt.savefig(write_plot, dpi=300)


def plot_in_one(y_pred, y_true, epochs, t, plot_label=None, write_plot=None):
    """Animates the trianing process over a time."""
    n_samples = y_true.shape[0]
    n_epochs = len(epochs)

    fig, ax = plt.subplots(1, 1)

    # Animation function to be repeatedly called during animation.
    def animate(epoch_idx):
        epoch_idx = int(epoch_idx)

        ax.clear()
        for sample_idx in range(n_samples):
            sample_time_points = t[sample_idx]
            number_of_time_indices = min(40, len(sample_time_points))

            # Select random indices of points in time
            times_indices = np.arange(len(sample_time_points))
            times_indices = np.random.choice(times_indices, replace=False, size=number_of_time_indices)
            times_indices = np.sort(times_indices)

            selected_times = sample_time_points[times_indices]

            # Plot predicted curve
            ax.plot(
                sample_time_points,  # All points in time that are available
                y_pred[epoch_idx, sample_idx],  # Prediction for this epoch
            )

            # Plot real curve on top (as a scatter plot)
            y_true_curve = y_true[sample_idx, times_indices]
            ax.scatter(selected_times, y_true_curve)

        # Title
        title = f"Prediction at epoch {int(epochs[epoch_idx])}"
        if plot_label is not None:
            title = f"[{plot_label}] {title}"
        fig.suptitle(title, fontsize=16)

    interval = 5
    stride = 1  # n_epochs / 100
    index = np.concatenate([np.rint(np.arange(0, n_epochs, stride)), np.array([n_epochs - 1] * 10)])
    ani = animation.FuncAnimation(fig, animate, index, interval=interval)  # noqa: F841

    # Either write or show plot
    if write_plot is None:
        plt.show()
    else:
        # Plot state at some of the epochs (don't plot the full animation when writing to file)
        num_snapshots = 5
        epoch_indices_plotted = [int(idx) for idx in np.rint(np.linspace(0, len(epochs) - 1, num_snapshots))]

        # Write plots
        plot_filename, plot_filetype = os.path.splitext(write_plot)
        plot_path, plot_filename = os.path.split(plot_filename)
        for chosen_epoch_idx in epoch_indices_plotted:
            animate(chosen_epoch_idx)
            if plot_label is not None:
                plot_path_label = os.path.join(plot_path, plot_label)
                os.makedirs(plot_path_label, exist_ok=True)
            snapshot_filename = f"epoch{int(epochs[chosen_epoch_idx]):06d}_{plot_filename}{plot_filetype}"
            plt.savefig(os.path.join(plot_path_label, snapshot_filename), dpi=300)
