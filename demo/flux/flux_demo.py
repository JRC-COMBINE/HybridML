import os
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Fluvoxamine Modules
from datasource import FluxDataLoader
from preprocessing import preprocess
from training import train
from visualization import generate_visualization_data, plot_in_one, plot_training_parameters

# Add location of HybridML to path
sys.path.append(os.getcwd())

# Import HybridML
from HybridML import Project  # noqa: E402


def set_fixed_seed(seed=None):
    """Seed the used random number generators to get reproducuable results."""
    if seed is None:
        seed = get_fixed_random_number()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_fixed_random_number():
    """For explanation concerning the algorithm: https://xkcd.com/221/"""
    return 4


def make_model(project):
    models = project.generate_models()
    assert len(models) == 1, "We are expecting only one model to be present in the project."
    model = models[0]
    return model


def main(
    project_name,
    progress_check_frequency=10,
    train_epochs=10,
    validation_split=0.8,
    time_points=15,
    early_stopping_patience=20,
    split_covariates=False,
    data_loader=None,
    plot_endlessly=False,
    plot=True,
):
    # Use double precision
    tf.keras.backend.set_floatx("float64")

    dose = 40
    # Set fixed random seed
    set_fixed_seed()

    # Load project
    base_dir = os.path.join(os.path.split(__file__)[0])
    projects_path = os.path.join(base_dir, "projects")
    project = Project.open_create(projects_path, project_name)

    # Load data
    if data_loader is None:
        data_loader = FluxDataLoader()
    X_raw, y_raw = data_loader.load()

    # Prepare data for training
    X, y = preprocess(X_raw, y_raw, dose, time_points, split_covariates)

    # Generate the model out of the model definitions stored in the demo project
    model = make_model(project)

    # Randomly choose 10 samples for visualization
    visualization_data, visualization_y = generate_visualization_data(X, y, n=10)

    # Prepare early stopping callback
    early_stopping = EarlyStopping(patience=early_stopping_patience, restore_best_weights=True, verbose=1)

    # Train
    history, log, epochs = train(
        model=model,
        input_data=X,
        y=y,
        visualization_data=visualization_data,
        progress_check_frequency=progress_check_frequency,
        train_epochs=train_epochs,
        validation_split=validation_split,
        additional_callbacks=[early_stopping],
    )
    project.save_models()

    # Save results of training (for animation plotting)
    log_prediction = np.array([item[0] for item in log])
    t_log = visualization_data[-1]

    # Plot development of parameters and animation of resulting concentration curves
    while plot:
        plot_training_parameters(history, log, progress_check_frequency)
        plot_in_one(log_prediction, visualization_y, epochs, t_log)
        if not plot_endlessly:
            break


def make_config():
    kwargs = {
        "project_name": "flux_demo",
        "progress_check_frequency": 2,
        "train_epochs": 100,
        "validation_split": 0.8,
        "time_points": 15,
    }
    return kwargs


if __name__ == "__main__":
    kwargs = make_config()

    main(**kwargs)
