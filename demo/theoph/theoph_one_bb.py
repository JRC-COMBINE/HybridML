import sys
import os
import time
import logging
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.getLogger().setLevel(logging.INFO)

# Add location of HybridML to path
sys.path.append(os.getcwd())

from HybridML import Project  # noqa: E402


def plot_all_curves():
    fig, ax = plt.subplots()
    for i in range(samples):
        sns.scatterplot(X_[-1][i], y[i], ax=ax)
        sns.lineplot(X[-1][i], prediction[i], ax=ax)
    ax.set_ylabel("Concentration")
    ax.set_xlabel("Time")
    ax.set_title(model.name)
    plt.show()


def plot_single_curves_in_subplots():
    n_cols = 2
    n_rows = samples // 2
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(20, 20))
    for i in range(samples):
        row = i // 2
        col = i % 2
        ax = axs[row][col]
        sns.scatterplot(X_[-1][i], y[i], ax=ax)
        sns.lineplot(X[-1][i], prediction[i], ax=ax)
        if i == 0:
            ax.set_ylabel("Concentration")
            ax.set_xlabel("Time")
        ax.set_title(f"Patient: {i+1}")

    plt.show()


if __name__ == "__main__":
    # Load project
    base_dir = os.path.join(".", "demo")
    projects_path = os.path.join(base_dir, "projects")
    project = Project.open_create(projects_path, "theoph_one_bb")

    # Generate the model out of the model definitions stored in the demo project
    models = project.generate_models()
    model = models[0]  # (there is only a single model for this project)

    # Load data
    data_sources = project.load_data_sources()
    ds = data_sources["theoph"]
    X, y = ds.get_train_data()

    # Use custom optimizer for model
    model.model.compile(optimizer=tf.optimizers.Adam(lr=0.025), loss="mse")
    model.summary()

    # Train the model
    validation_split = 0.1
    epochs = 5
    begin_time = time.time()
    history = model.fit(X, y, validation_split=validation_split, epochs=epochs)

    # Plot fitted curves (concentration over time)
    time_steps = np.linspace(0, 25, 100)
    X_ = X.copy()
    X[-1] = np.array([time_steps for i in range(X_[-1].shape[0])])
    prediction, par_ka, par_k12, par_k21, par_k10 = model.predict(X, consider_additional_outputs=True)
    samples = prediction.shape[0]

    # Print out final values for ODE parameters k
    # (values that Sebastian found out:
    #  ka          k10         k12         k21
    #  2.14311063  0.05806329 -0.08444969  1.15703847)
    for par_name, par_values in zip(["ka", "k12", "k21", "k10"], [par_ka, par_k12, par_k21, par_k10]):
        logging.info(f"Mean parameter {par_name}: {np.mean(par_values):0.8f}")

    # Plot
    plot_all_curves()
    plot_single_curves_in_subplots()
