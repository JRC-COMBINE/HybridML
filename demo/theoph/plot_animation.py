import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_subplots(y_pred, y_true, epochs, t):
    n_samples = y_true.shape[0]
    n_epochs = len(epochs)

    per_row = 3

    def get_col_row(i):
        col = i // per_row
        row = i % per_row
        return col, row

    cols = per_row
    rows = n_samples // per_row
    cols = max(cols, 2)
    rows = max(rows, 2)
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)

    def animate(i):
        i = int(i)
        for i_sample in range(n_samples):
            col, row = get_col_row(i_sample)
            ax = axs[col, row]
            ax.clear()
            ax.plot(t[i_sample], y_pred[i, i_sample])
            ax.scatter(t[i_sample], y_true[i_sample])
        axs[0, 0].legend(["y_pred", "y_true"])
        fig.suptitle(f"Prediction at epoch {int(epochs[i])}", fontsize=16)

    interval = 1
    stride = 1  # n_epochs / 100
    index = np.concatenate([np.rint(np.arange(0, n_epochs, stride)), np.array([n_epochs - 1] * 10)])
    ani = animation.FuncAnimation(fig, animate, index, interval=interval)  # noqa: F841
    plt.show()


def plot_in_one(y_pred, y_true, epochs, t):
    n_samples = y_true.shape[0]
    n_epochs = len(epochs)
    fig, ax = plt.subplots(1, 1)

    def animate(i):
        i = int(i)
        ax.clear()
        for i_sample in range(n_samples):
            ax.plot(t[i_sample], y_pred[i, i_sample])
            ax.scatter(t[i_sample], y_true[i_sample])
        fig.suptitle(f"Prediction at epoch {int(epochs[i])}", fontsize=16)

    interval = 5
    stride = 1  # n_epochs / 100
    index = np.concatenate([np.rint(np.arange(0, n_epochs, stride)), np.array([n_epochs - 1] * 10)])
    ani = animation.FuncAnimation(fig, animate, index, interval=interval)  # noqa: F841
    plt.show()


if __name__ == "__main__":

    epochs = np.loadtxt("epochs.np_array")
    true_y = np.loadtxt("y_true.np_array")
    data = np.loadtxt("data.np_array")
    data = data.reshape(((len(epochs), true_y.shape[0], true_y.shape[1])))
    t = np.loadtxt("t.np_array")
    # epochs: 3
    # true_y: 12 x 11
    # data : 3 x 12 x 11
    true_y = true_y[: t.shape[0]]
    plot_in_one(data, true_y, epochs, t)
