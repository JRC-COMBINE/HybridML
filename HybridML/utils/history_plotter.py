import numpy as np
import matplotlib.pyplot as plt


def plot_mult(histories, names=[]):
    if len(histories) != len(names):
        names = [i for i, _ in enumerate(histories)]
    length = len(histories)
    c = 4
    fig, axis = plt.subplots(ncols=c, nrows=length // c, sharey=True, sharex=True)
    for i in range(length):
        x = i // c
        y = i % c
        if histories[i] is None:
            continue
        history_dict = histories[i].history
        loss = history_dict["loss"]
        val_loss = history_dict["val_loss"]
        ax = axis[x, y]
        # if l>1:
        #     ax =  axis[i]
        # else:
        #     ax = axis
        epochs = range(1, len(loss) + 1)
        # "bo" is for "blue dot"
        ax.plot(epochs, loss, "bo", label="Training loss")
        # b is for "solid blue line"
        ax.plot(epochs, val_loss, "b", label="Validation loss")
        ax.set_title(names[i])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        if i == 0:
            ax.legend()
    plt.show()


def plot_history(history, name):
    history_dict = history.history
    # metrics = history_dict.keys()
    # metric_names = [
    #     key for key in history_dict.keys() if not key.startswith("val_")
    # ]
    # if "acc" in history_dict:
    #     acc = history_dict["acc"]
    #     val_acc = history_dict["val_acc"]
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]

    epochs = len(history.epoch)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, "bo", label="Training loss")
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss: " + name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()


def super_plot_history(history, name):
    history_dict = history.history
    metric_names = [key for key in history_dict.keys() if not key.startswith("val_")]

    epochs = range(1, len(next(iter(history_dict.values()))) + 1)
    # plt.figure(figsize=(15,5))
    ncols = 2
    nrows = int(np.ceil(len(metric_names) / 2))
    for i, met_name in enumerate(metric_names):
        plt.subplot(nrows, ncols, i + 1)
        met = history_dict[met_name]
        val_met = history_dict["val_" + met_name]
        plt.plot(epochs, met, "bo", label="training")
        plt.plot(epochs, val_met, "b", label="validation")
        plt.xlabel("Epochs")
        plt.title(met_name)
        plt.legend()
    plt.show()


def super_multi_plot_history(histories, names):
    max_len = 0
    metric_names_list = []
    for history in histories:
        metric_names = [key for key in history.keys() if not key.startswith("val_")]
        metric_names_list.append(metric_names)
        if len(metric_names) > max_len:
            max_len = len(metric_names)
    ncols = max_len
    nrows = len(histories)
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)

    for history_index, history in enumerate(histories):
        epochs = range(1, len(next(iter(history.values()))) + 1)
        for met_index, met_name in enumerate(metric_names_list[history_index]):
            met = history[met_name]
            val_met = history["val_" + met_name]
            ax = axis[history_index, met_index]
            ax.plot(epochs, met, "bo", label="training")
            ax.plot(epochs, val_met, "b", label="validation")
            ax.set_title(met_name)
            if met_index == 0:
                ax.set_title(met_name + " " + names[history_index])

    plt.show()
