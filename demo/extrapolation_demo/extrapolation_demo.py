import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# Add location of HybridML to path
sys.path.append(os.getcwd())

from HybridML import Project  # noqa: E402

# Functions which we will later try to approximate using our models


def f1(i):
    return i[0] * 2 * i[1]


def f2(i):
    return -i[2] * i[3]


def f(i):
    return f1(i) + f2(i)


def generate_correlated_inputs(n=100, min=-5, max=5, noise=0.25):
    """Generates 4d inputs that are correlated. They linear combinations of two unrelated inputs plus random noise."""
    xs = random.rand(2, n) * (max - min) - max
    correlated_xs = np.array([xs[0] + xs[1], xs[0] - xs[1], -xs[0] - xs[1], -xs[0] + xs[1]])
    rand_component = np.random.rand(4, n) * (2 * noise) - noise
    noisy_correlated_ys = correlated_xs + rand_component
    return noisy_correlated_ys


def generate_unrelated_inputs(n=10, min=-5, max=5):
    """Generates 4d inputs that are unrelated."""
    return random.rand(4, n) * (max - min) - max


def generate_data(n=100, correlated=True):
    """Generate samples for training and testing.
    correlated=True -> inputs are correlated, based on 2 unrelated inputs."""
    if correlated:
        x = generate_correlated_inputs(n)
    else:
        x = generate_unrelated_inputs(n)
    y = f(x)
    # Match the input dimensions of the models
    x = [x[:2].T, x[2:].T]
    return x, y


random = np.random.RandomState(seed=4)  # chosen by fair dice roll. guaranteed to be random. xkcd.com/221
# Generate data
x_train, y_train = generate_data(n=200, correlated=True)
x_val, y_val = generate_data(n=100, correlated=False)
x_test, y_test = generate_data(n=100, correlated=False)


def show_pairplot(x, title):
    """Show a pairplot of the 2x2xn-dimensional input data."""
    # Put multidimensional data into a dataframe
    x = np.swapaxes(x, 1, 2)
    x = x.reshape((x.shape[0] ** 2, x.shape[-1]))
    x = pd.DataFrame(x.T, columns=["a", "b", "c", "d"])

    sns.set_context("paper", rc={"axes.labelsize": 15})
    sns.pairplot(x, height=1)
    plt.savefig(f"{title}.png")


show_pairplot(x_train, "correlated_inputs")
show_pairplot(x_val, "unrelated_inputs")
# show_pairplot(x_test, "Test Data")

report = []

mse_per_seed = []
number_of_seeds = 10
for seed in range(number_of_seeds):
    tf.random.set_seed(seed=seed)

    # Load project
    project_path = os.path.join(os.path.split(__file__)[0], "projects", "extrapolation_demo", "extrapolation_demo.json")
    project = Project.open(project_path)

    # Alternative way to load the data sources, that were defined in the project file extrapolation_demo.json.
    # data_sources = project.load_data_sources()
    # train = data_sources["train"]
    # val = data_sources["val"]
    # test = data_sources["test"]
    # x_train, y_train = train.xs, train.ys
    # x_val, y_val = val.xs, val.ys
    # x_test, y_test = test.xs, test.ys

    # Generate models out of the model definitions stored in the demo project
    models = project.generate_models()
    print(models)

    # Model summary can help you identify problems with the model architecture
    for model in models:
        model.summary()

    for model in models:
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)
        # Fit the model to the data
        history = model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            epochs=1000,
            callbacks=[early_stopping],
        )
    if False:  # Show further informations
        # Save models and load. They will be saved at `demo/projects/extrapolation_demo/models`
        project.save_models()
        models_loaded = project.load_models()

        # Test prediction for both models
        i = np.array([1, 2, 3, 4])
        test_point = [np.array([[1, 2]]), np.array([[3, 4]])]

        for model in models_loaded:

            # Get prediction
            prediction = model.predict(test_point, consider_additional_outputs=True)

            output_value = prediction[0]
            print(f"[{model.name}] Eval")
            print(f"[{model.name}] Expected the value: {f(i)} and got {output_value}")
            print("\n")

        # The hybrid model exposes two of its inner values (attribute "additional_outputs" defined in `hybrid.json` in
        # `demo/projects/extrapolation_demo`). These additional outputs are given as further elements of `prediction`
        hybrid_model = models[1]
        a = prediction[1]
        b = prediction[2]

        print(f"[{hybrid_model.name}] Value two black boxes: {a}, {b}")
        print(f"[{hybrid_model.name}] Values of the two original sub-functions: {f1(i)}, {f2(i)}")

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

losses = np.array(mse_per_seed)

print("Single model losses")
print("\n".join(report))

print()
print("Average MSE per model")
print("Black Box Model:", np.mean(losses[:, 0]))
print("Hybrid Model:", np.mean(losses[:, 1]))

print("oll klear.")
