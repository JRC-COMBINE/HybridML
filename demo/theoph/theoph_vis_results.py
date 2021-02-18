import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add location of HybridML to path
sys.path.append(os.getcwd())

from HybridML import Project  # noqa: E402

base_dir = os.path.join(os.getcwd(), "demo")
projects_path = os.path.join(base_dir, "projects")


def take_first_elements(X, y, i=2):
    return [X[0][:i], X[1][:i], X[2][:i]], y[:i]


def get_data(prune=False):
    project = Project.open_create(projects_path, "theoph_demo")

    # Get training data
    data_sources = project.load_data_sources()
    ds = data_sources["theoph"]
    X_real, y_real = ds.get_train_data()

    # take first 2 elements of training data
    if prune:
        X_real, y_real = take_first_elements(X_real, y_real)

    predict_times = np.linspace(0, 25, 30)
    X_predict = X_real.copy()
    X_predict[-1] = np.array([predict_times for i in range(X_real[-1].shape[0])])
    return project, X_predict, X_real, y_real


def plot_lot(plot_data, real_data, _axs=None):
    X_real, y_real = real_data
    if _axs is None:
        fig, axs = plt.subplots(1, len(plot_data), sharex=True, sharey=True)
        fig.suptitle(plot_data[0][2])

    else:
        axs = _axs
    for i, (X_predict, prediction, name) in enumerate(plot_data):
        ax = axs[i] if len(plot_data) > 1 else axs

        if i == 0:
            ax.legend(["True", "Prediction"])
            ax.set_ylabel("Concentration")
            ax.set_xlabel("Time")

        samples = prediction.shape[0]
        for i in range(samples):
            sns.scatterplot(X_real[-1][i], y_real[i], ax=ax)
            sns.lineplot(X_predict[-1][i], prediction[i], ax=ax)
    if _axs is None:
        plt.show()


if __name__ == "__main__":
    project, X_predict, X_real, y_real = get_data()

    # load models
    models = project.load_models()[:1]
    # predict with all models
    plot_data = []
    for model in models:
        prediction = model.predict(X_predict)
        plot_data.append((X_predict, prediction, model.name))

    # plot result
    plot_lot(plot_data, (X_real, y_real))
