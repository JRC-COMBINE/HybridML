import matplotlib.pyplot as plt
import seaborn as sns
from HybridML.keras.Containers import KerasHistoryContainer

val = "val_"


def show_training(container: KerasHistoryContainer, model_name=None):
    history = container.history
    history_dict = history.history
    metrics = [key for key in history_dict.keys() if not key.startswith(val)]

    if len(metrics) > 2:
        cols = 2
    else:
        cols = 1
    n_rows = len(metrics) // cols
    fig, axs = plt.subplots(cols, n_rows, sharex=True)

    x = range(len(history_dict[metrics[0]]))
    for i, metric in enumerate(metrics):
        row, col = i // cols, i % cols

        ax = axs[row][col] if n_rows > 1 else axs[row] if cols > 1 else axs

        sns.lineplot(x, history_dict[metric], ax=ax)
        sns.lineplot(x, history_dict[val + metric], ax=ax)

        # Set up title
        if model_name is None:
            title = "Model"
        else:
            title = model_name
        title = title + " " + metric
        ax.set_title(title)

        ax.set_ylabel(metric)
        ax.set_xlabel("Epoch")
        ax.legend(["Train", "Val"], loc="upper left")
    plt.show()
