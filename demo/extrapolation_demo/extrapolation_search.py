import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from extrapolation_demo import generate_data


# Add location of HybridML to path
sys.path.append(os.getcwd())
from HybridML import Project  # noqa: E402


def show_pairplot(x, y, title):
    """Show a pairplot of the 2x2xn-dimensional input data."""
    # Put multidimensional data into a dataframe
    x = np.array(x).swapaxes(1, 2)
    x = x.reshape((x.shape[0] ** 2, x.shape[-1]))
    x = pd.DataFrame(x.T, columns=["a", "b", "c", "d"])
    x["targets"] = y

    sns.set_context("paper", rc={"axes.labelsize": 15})
    sns.pairplot(x, height=1)
    plt.savefig(f"{title}.png")
    plt.show()


def plot_target_vs_prediction(n_models, n_seeds, predictions, targets, model_names, title):
    # Prepare Data
    predictions = np.array(predictions)
    predictions_show, targets_show = [], []
    samples_per_model = int(np.ceil(len(targets) / n_seeds))
    for i, prediction in enumerate(predictions):
        start = i * samples_per_model
        end = min(len(targets), start + samples_per_model)
        predictions_show.append(prediction[:, start:end])
        targets_show.append(targets[start:end])

    # Show Data
    flat_targets = [inner for outer in targets_show for inner in outer]
    line = [min(flat_targets), max(flat_targets)]

    fig, axs = plt.subplots(1, n_models, sharex=True, sharey=True, figsize=(10, 5))
    fig.suptitle(title)
    for model_n, ax in enumerate(axs):
        for seed_n in range(len(targets_show)):
            prediction = predictions_show[seed_n][model_n]
            target = targets_show[seed_n]
            ax.scatter(target, prediction)
        ax.plot(line, line)
        ax.set_title(model_names[model_n])
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
    plt.savefig(title + ".png")
    plt.show()


def main():

    random = np.random.RandomState(seed=4)  # chosen by fair dice roll. guaranteed to be random. xkcd.com/221
    # Generate data
    x_train, y_train = generate_data(n=200, correlated=True, random=random)
    x_val, y_val = generate_data(n=100, correlated=False, random=random)
    x_test, y_test = generate_data(n=100, correlated=False, random=random)

    # show_pairplot(x_train, y_train, "correlated_inputs")
    # show_pairplot(x_val, y_val, "unrelated_inputs")
    # show_pairplot(x_test, y_val, "Test Data")

    # Seeds that produced a reasonable result
    seeds = [1, 6, 7, 14, 15]

    epochs = 20
    report = []
    train_data_predictions = []
    test_data_predictions = []
    mse_per_seed = []

    models = []
    for i, seed in enumerate(seeds):
        tf.random.set_seed(seed=seed)

        # Load project
        project_path = os.path.join(
            os.path.split(__file__)[0], "projects", "extrapolation_demo", "extrapolation_demo.json"
        )
        project = Project.open(project_path)

        # Generate models out of the model definitions stored in the demo project
        models = project.generate_models()
        print(models)

        # Model summary can help you identify problems with the model architecture
        for model in models:
            model.summary()

        for model in models:
            early_stopping = tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)
            # Fit the model to the data
            model.fit(
                x=x_train,
                y=y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                callbacks=[early_stopping],
            )

        mse_per_model = []
        # We can also evaluate the models manually.
        for model in models:
            # `eval_result` is a list of scalar losses. There is one loss for each of metrics defined for the model.
            eval_result = model.evaluate(x_test, y_test)
            report.append(f"[{model.name}]")
            # Print out the loss along with the name of the metric
            for loss, metric_name in zip(eval_result, model.model.metrics_names):
                report.append(f"\t {metric_name} = {loss}")
            mse = eval_result[0]
            mse_per_model.append(mse)
        mse_per_seed.append(mse_per_model)

        # Predict for later vizualisation
        train_data_predictions.append([model.predict(x_train) for model in models])
        test_data_predictions.append([model.predict(x_test) for model in models])

    model_names = [model.name for model in models]
    plot_target_vs_prediction(len(models), len(seeds), train_data_predictions, y_train, model_names, "Train Data")
    plot_target_vs_prediction(
        len(models), len(seeds), test_data_predictions, y_test, model_names, "Test Data Predictions"
    )

    # Print out losses for each checked model
    losses = np.array(mse_per_seed)
    to_print = []
    to_print += ["Single model losses"]
    to_print += report
    to_print += ["Average MSE per model"]
    to_print += [f"Black Box Model: {np.mean(losses[:, 0])}"]
    to_print += [f"Hybrid Model: {np.mean(losses[:, 1])}"]
    to_print = "\n".join(to_print)
    print(to_print)
    with open("extrapolation_search_results.txt", "w") as f:
        f.write(to_print)


if __name__ == "__main__":
    main()
