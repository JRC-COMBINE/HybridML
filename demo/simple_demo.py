import os
import sys

import numpy as np

# Add location of HybridML to path
sys.path.append(os.getcwd())

from HybridML import Project  # noqa: E402

random = np.random.RandomState(seed=4)  # chosen by fair dice roll. guaranteed to be random. xkcd.com/221


def f(X):
    """Simple function of hybrid structure to estimate."""
    return (X[0] * 4) + (X[1] * 0.3)


def generate_data(n=100):
    X = random.uniform(size=(2, n))
    y = f(X)
    X = list(X)  # model expects list of np arrays as input.
    return X, y


def main():
    # Generate random data for two inputs
    X_train, y_train = generate_data(n=200)
    X_test, y_test = generate_data(n=20)

    # Create model from description
    model_description_path = os.path.join(os.path.split(__file__)[0], "simple_model.json")
    model = Project.create_model(model_description_path)

    print("Train the model with the generated data.")
    model.fit(X_train, y_train, validation_split=0.8, epochs=10)

    print("Evaluate the model on test data.")
    model.evaluate(X_test, y_test)

    # Save model to file
    model_path = model_description_path + ".h5"
    model.save_to_file(model_path)

    # Load model from file. The model description is needed to load the model.
    loaded_model = Project.load_model(model_description_path, model_path)

    # Predict both models and compare output
    prediction = model.predict(X_test)
    loaded_model_prediction = loaded_model.predict(X_test)
    assert np.all(np.abs(prediction - loaded_model_prediction) < 1e-5)
    print("ok.")


if __name__ == "__main__":
    main()
