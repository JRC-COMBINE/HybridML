import numpy as np


def thin_out_time_points(X, y, number_of_time_points):
    """Randomly sample time points from X and y.    """
    total_time_points = y.shape[-1]
    index = list(range(1, total_time_points))
    time_points = X[-1]

    new_time_points = []
    new_targets = []
    n_samples = X[0].shape[0]
    for sample in range(n_samples):
        rand_index = np.random.choice(index, size=number_of_time_points, replace=False)
        rand_index = list(sorted(rand_index))
        rand_index = np.insert(rand_index, 0, 0)  # We need to include time 0

        new_time_points.append(time_points[sample][rand_index])
        new_targets.append(y[sample][rand_index])

    new_time_points = np.array(new_time_points)
    new_targets = np.array(new_targets)

    return [*X[:-1], new_time_points], new_targets


def preprocess(X, y, dose, number_of_time_points=None, split_covariates=False):
    """Apply the preprocessing pipeline for Fluvoxamine user case."""

    # Reduce the number of time points, to have more realistic inputs.
    if number_of_time_points is not None:
        X, y = thin_out_time_points(X, y, number_of_time_points)

    # Insert constant initial dose into data
    dose_arr = np.array([dose] * len(X[0]))
    X = [X[0], dose_arr, *X[1:]]

    # Split the single covariate input of size [samples x n_cov] into n_cov inputs of size [samples].
    # This is necessaryfor the input of  the structured flux model.
    if split_covariates:
        covariate_inputs = [X[0][:, i] for i in range(X[0].shape[-1])]
        X = [*covariate_inputs, X[-2], X[-1]]
    return X, y
