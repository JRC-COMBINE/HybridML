import os
import sys

import numpy as np
import tensorflow as tf

# Add location of HybridML to path
sys.path.append(os.getcwd())

from HybridML import Project  # noqa: E402

# Functions which we will later try to approximate using our models


def f1(i):
    return i[:, 0] + 2 * i[:, 1]


def f2(i):
    return -i[:, 0] + i[:, 1]


def f(i):
    return f1(i[0]) * f2(i[0])


def generate_correlated_inputs(n=100, min=-5, max=5, noise=0.25, random=None):
    """Generates 4d inputs that are correlated. They linear combinations of two unrelated inputs plus random noise."""

    # Generate two random base inputs
    xs = random.rand(2, n) * (max - min) - max

    # Corellate inputs
    correlated_xs = np.array([xs[0] + xs[1], xs[0] - xs[1], -xs[0] - xs[1], -xs[0] + xs[1]])

    # Add noise
    rand_component = np.random.normal(size=(4, n)) * (2 * noise) - noise
    noisy_correlated_ys = correlated_xs + rand_component

    return noisy_correlated_ys


def generate_unrelated_inputs(n=10, min=-5, max=5, random=None):
    """Generates 4d inputs that are unrelated."""
    return random.rand(4, n) * (max - min) - max


def generate_data(n=100, correlated=True, random=None):
    """Generate samples for training and testing.
    correlated=True -> inputs are correlated, based on 2 unrelated inputs."""
    if random is None:
        random = np.random
    if correlated:
        x = generate_correlated_inputs(n, random=random)
    else:
        x = generate_unrelated_inputs(n, random=random)
    # Match the input dimensions of the models
    x = [x[:2].T, x[2:].T]
    y = f(x)
    return x, y


def main():
    seed = 4  # chosen by fair dice roll. guaranteed to be random. xkcd.com/221
    random = np.random.RandomState(seed=seed)
    tf.random.set_seed(seed=seed)

    # Generate data
    x_train, y_train = generate_data(n=200, correlated=True, random=random)
    x_val, y_val = generate_data(n=100, correlated=False, random=random)
    x_test, y_test = generate_data(n=100, correlated=False, random=random)

    # Alternatively: Load pregenerated input data, defined in the extrapolation_demo.json
    # data_sources = project.load_data_sources()
    # train = data_sources["train"]
    # val = data_sources["val"]
    # test = data_sources["test"]
    # x_train, y_train = train.xs, train.ys
    # x_val, y_val = val.xs, val.ys
    # x_test, y_test = test.xs, test.ys

    # Load project
    # Project is definid in project description file
    project_path = os.path.join(os.path.dirname(__file__), "projects", "extrapolation_demo", "extrapolation_demo.json")
    project = Project.open(project_path)

    # Generate models out of the model definitions stored in the demo project
    # The models are defined in hybrid.json and blackbox.json, as described in the project description file
    models = project.generate_models()

    # Model summary can help you identify problems with the model architecture
    for model in models:
        model.summary()

    for model in models:
        # Stop the training early, when the val loss has not decreased for 50 epochs
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)
        # Fit the model to the data
        model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            epochs=2000,
            callbacks=[early_stopping],
        )

    # We can also evaluate the models manually.
    for model in models:
        # `eval_result` is a list of scalar losses. There is one loss for each of metrics defined for the model.
        eval_result = model.evaluate(x_test, y_test)
        print(f"[{model.name}]")
        # Print out the loss along with the name of the metric
        for loss, metric_name in zip(eval_result, model.model.metrics_names):
            print(f"\t {metric_name} = {loss}")

    # Save models and load. They will be saved at `demo/projects/extrapolation_demo/models`
    project.save_models()
    models_loaded = project.load_models()

    # Note that the models expect a list of inputs with the right dimensionality.
    # In this case there are two inputs with N samples of size to, thus we need the dimensions [N x 2, N x 2]
    test_point = [x_test[0][0:1], x_test[1][0:1]]

    # Show further information
    print("Manual Evaluation")
    prediction = None
    for model in models_loaded:
        # Get prediction
        prediction = model.predict(test_point, consider_additional_outputs=True)

        output_value = prediction[0]
        print(f"[{model.name}] Expected the value: {f(test_point)} and got {output_value}")

    # The hybrid model exposes two of its inner values (attribute "additional_outputs" defined in `hybrid.json` in
    # `demo/projects/extrapolation_demo`). These additional outputs are given as further elements of `prediction`
    hybrid_model = models[1]
    a = prediction[1]
    b = prediction[2]

    print(f"[{hybrid_model.name}] Value two black boxes: {a}, {b}")
    print(f"[{hybrid_model.name}] Values of the two original sub-functions: {f1(test_point[0])}, {f2(test_point[1])}")

    print("ok.")


if __name__ == "__main__":
    main()
