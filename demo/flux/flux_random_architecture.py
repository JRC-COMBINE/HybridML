import argparse
import json
import logging
import os
import random
import shutil
import string
import sys
import time
from itertools import product

import humanfriendly
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)

# Add location of HybridML to path
sys.path.append(os.getcwd())

from HybridML import Project  # noqa: E402

from datasource import FluxDataLoader  # noqa: E402
from flux_demo import make_model, set_fixed_seed  # noqa: E402
from preprocessing import preprocess  # noqa: E402
from training import train_complex  # noqa: E402
from visualization import plot_in_one, plot_training_parameters  # noqa: E402

# General setup
base_dir = os.path.join(".", "demo", "flux")
projects_path = os.path.join(base_dir, "projects")
# random_projects_path = os.path.join(os.getenv("WORK"), "hybrid_platform", "random_search")
# random_projects_path = "/home/kilian/datasets/hybrid_platform/random_search"  # for testing on Kilian's local machine
random_projects_path = "/Users/younes/datasets/hybrid_platform/random_search"  # for testing on Younes' local machine


def new_random_project(combination_idx, options):
    # Unpack options
    (
        activation,
        activation_output_linear,
        num_hidden_layers,
        size_hidden_layers,
        use_random_regularization,
        arithmetic_transformation,
        optimizer,
        learning_rate,
    ) = options

    # Copy the project from a template and modify it randomly. Then run the project.
    random_id = "".join(
        [f"combination_{combination_idx}"] + ["_"] + [random.choice(string.ascii_uppercase) for _ in range(3)]
    )
    new_project_name = f"random_flux_{random_id}_unfinished"
    new_project_dir = os.path.join(random_projects_path, new_project_name)
    os.makedirs(new_project_dir, exist_ok=True)
    logging.info(f"Starting new random project with name {new_project_name}")

    # Copy model definition JSON file
    new_model_json_name = "flux_random_model"
    new_model_json_filename = f"{new_model_json_name}.json"
    new_model_json_path = os.path.join(new_project_dir, new_model_json_filename)
    copied_project_name = "structured_flux_demo"
    shutil.copy(os.path.join(projects_path, copied_project_name, "structured_flux_model.json"), new_model_json_path)

    # Copy project definition JSON file
    new_project_json_name = new_project_name
    new_project_json_path = os.path.join(new_project_dir, f"{new_project_json_name}.json")
    shutil.copy(os.path.join(projects_path, copied_project_name, f"{copied_project_name}.json"), new_project_json_path)

    # Copy additional files
    additional_files = ["parameter selection.csv"]
    for add_file_filename in additional_files:
        shutil.copy(
            os.path.join(projects_path, copied_project_name, add_file_filename),
            os.path.join(new_project_dir, add_file_filename),
        )

    # Register the model file in the project file
    proj_file = load_json(new_project_json_path)
    proj_file["name"] = new_project_name
    proj_file["models"] = [new_model_json_filename]
    write_json(proj_file, new_project_json_path)

    # Modify model file
    model_file = load_json(new_model_json_path)

    # Change black box layers and activations for all black boxes
    for bb_id in ["BB_ka", "BB_ke", "BB_k12_k21"]:
        bb = [node for node in model_file["nodes"] if node["id"] == bb_id][0]

        # Save output layer's size: It needs to stay as it was
        output_layer_size = bb["layers"][-1]["size"]

        # Create re required number of layers
        bb["layers"] = []
        num_layers = num_hidden_layers + 1  # output layer is also a layer
        for layer_idx in range(num_layers):
            bb["layers"].append({"size": size_hidden_layers, "activation": activation})

        # Restore output layer's size
        bb["layers"][-1]["size"] = output_layer_size

        # Set activation of output layer to linear
        if activation_output_linear:
            bb["layers"][-1]["activation"] = "None"

        # Add regularization to a layer
        if use_random_regularization:
            layer_with_regularization = random.choice(bb["layers"])
            regularizer_target = random.choice(["activity_regularizer", "kernel_regularizer"])
            regularizer_kind = random.choice(["L1", "L2"])
            regularizer_magnitude = random.choice([0.001, 0.01, 0.1, 1.0])
            layer_with_regularization[regularizer_target] = f"{regularizer_kind}({regularizer_magnitude})"

    # Change arithmetic nodes
    nn_output_params = ["ka", "ke", "k12", "k21"]
    for param in nn_output_params:
        arithmetic_node = [
            node for node in model_file["nodes"] if "expression" in node and node["expression"].startswith(param)
        ][0]

        # Find out what value this parameter is "primed" with. The value used should be a good estimate for the
        # parameter.
        priming_value = float(arithmetic_node["expression"].split(" + ")[-1])

        # Find out magnitude of priming value
        priming_value_mag = np.floor(np.log(priming_value) / np.log(10))

        if arithmetic_transformation.startswith("addition"):

            # Add the priming value and the neural network output
            # For this, the neural network output needs to be scaled into an appropriate magnitude
            if arithmetic_transformation == "addition_smaller_mag":
                priming_value_mag -= 1  # Go one order of magnitude smaller to not influence result too much
            if arithmetic_transformation == "addition_even_smaller_mag":
                priming_value_mag -= 2  # Go two orders of magnitude smaller

            # Set expression so that scaled NN parameter and priming value are added
            arithmetic_node["expression"] = (
                f"{param} = p_{param} * {float(1 * np.power(10, priming_value_mag))} +" f" {priming_value}"
            )

        elif arithmetic_transformation == "product":
            # Set expression so that NN parameter acts as a factor for priming value
            arithmetic_node["expression"] = f"{param} = p_{param} * {priming_value}"
        elif arithmetic_transformation == "identity":
            # Set arithmetic node to an identity node that does not do anything
            arithmetic_node["expression"] = f"{param} = p_{param}"
        else:
            assert False

    # Save the changed model
    write_json(model_file, new_model_json_path)

    # Init optimizer
    opt = optimizer(lr=learning_rate)

    # Randomly choose seed
    seed_truly_random = True
    seed = random.randint(1, 2 ** 16)
    logging.info(f"Seed: {seed}")

    # Run training
    train_project(
        project_name=new_project_name,
        optimizer=opt,
        loss="mse",
        fixed_random_seed=seed,
        seed_truly_random=seed_truly_random,
        arithmetic_transformation=arithmetic_transformation,
        model_definition=model_file,
        options=options,
    )


def load_json(path):
    with open(path, "r") as json_file:
        obj = json.load(json_file)
    return obj


def write_json(obj, path):
    with open(path, "w") as json_file:
        json.dump(obj, json_file, sort_keys=True, indent=4, separators=(",", ": "))


def train_project(
    project_name,
    optimizer,
    loss,
    fixed_random_seed,
    seed_truly_random,
    arithmetic_transformation,
    model_definition,
    options,
    number_of_time_points=15,
    split_covariates=True,
):
    # Set fixed seed
    set_fixed_seed(fixed_random_seed)

    # Load project
    project = Project.open_create(random_projects_path, project_name)

    # Load data
    loader = FluxDataLoader()
    loader.cov_path = os.path.join(
        os.getcwd(), "demo", "flux", "projects", "structured_flux_demo", "parameter selection.csv"
    )

    dose = 40

    X_raw, y_raw = loader.load()
    X, y = preprocess(
        X_raw, y_raw, dose=dose, number_of_time_points=number_of_time_points, split_covariates=split_covariates
    )
    # save original, full-resolution version
    X_full, y_full = preprocess(X_raw, y_raw, dose=dose, split_covariates=split_covariates)

    # Generate the model out of the model definitions stored in the demo project
    model = make_model(project)

    # Use custom optimizer for model
    model.compile(optimizer=optimizer, loss=loss)

    # Compose results (for later analysis)
    train_results = {
        "loss": loss,
        "optimizer": {
            "name": optimizer._name,
            "hyper": {hyper_name: float(hyper_val) for (hyper_name, hyper_val) in optimizer._hyper.items()},
        },
        "seed": fixed_random_seed,
        "seed_truly_random": seed_truly_random,
        "model_definition": model_definition,
        "options": [opt for opt in options if type(opt) in [str, bool, int, float]],
        "arithmetic_transformation": arithmetic_transformation,
    }

    # Select test subjects - these will not be used for optimization or for selecting hyperparameters
    test_indices = [0, 5, 9, 30, 44, 54, 62, 69, 81, 89]

    # Prepare validation data by selecting the indices
    val_split = 0.1
    fixed_val_set = True
    non_test_indices = np.setdiff1d(np.arange(len(y)), test_indices)
    val_num = int(np.floor(val_split * len(non_test_indices)))
    if fixed_val_set:
        # Generate random seed so randomness for later operations is preserved
        random_numpy_seed = np.random.randint(0, 2 ** 13)

        # Set fixed seed
        np.random.seed(42)

    val_indices = np.random.choice(non_test_indices, val_num, replace=False)

    if fixed_val_set:
        # Restore a random seed
        np.random.seed(random_numpy_seed)

    # Train
    begin_time = time.time()
    progress_check_frequency = 10
    try:
        # Prepare early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True, verbose=1)

        training_result = train_complex(
            trained_model=model,
            x=X,
            y=y,
            x_full=X_full,
            y_full=y_full,
            val_indices=val_indices,
            test_indices=test_indices,
            progress_check_frequency=progress_check_frequency,
            train_epochs=20000,
            additional_callbacks=[early_stopping],
        )
        (
            history,
            train_indices,
            train_indices_plot,
            val_indices_plot,
            log_train,
            epochs_train,
            log_val,
            epochs_val,
        ) = training_result
    except Exception as exception:
        train_results["crashed"] = True
        train_results["exception"] = f"{str(type(exception))}: {str(exception)}"
    else:
        train_results["crashed"] = False

    if not train_results["crashed"]:
        project.save_models()

    if not train_results["crashed"]:
        # Sanity check: Train epochs must be equal to validation epochs
        assert epochs_train == epochs_val
        epochs = epochs_train

    # Benchmark time
    time_taken = time.time() - begin_time
    if not train_results["crashed"]:
        time_per_epoch = time_taken / (max(epochs) + 1)
        logging.info(f"Time taken (at {max(epochs)} epochs): {humanfriendly.format_timespan(time_taken)}")
        logging.info(f"Time per epoch: {humanfriendly.format_timespan(time_per_epoch)}")

    train_results.update({"time_taken": time_taken})

    # Prepare data for plotting
    if not train_results["crashed"]:
        # Get train plotting data
        log_prediction_train = np.array([lo[0] for lo in log_train])
        t_train = X_full[-1][train_indices_plot]

        # Get val plotting data
        log_prediction_val = np.array([lo[0] for lo in log_val])
        t_val = X_full[-1][val_indices_plot]

        # Predict test data (for plotting)
        test_indices_plot = np.copy(test_indices)  # Plot all test indices
        visualization_data_test = [x[test_indices_plot] for x in X_full]
        log_test = [model.additional_outputs_model.predict(visualization_data_test)]

        # Bring test data prediction results into the same kind of shape as validation and training results
        log_prediction_test = np.array([lo[0] for lo in log_test])
        t_test = X_full[-1][test_indices_plot]

    # Save results of training
    if not train_results["crashed"]:
        full_history = history.history
        train_loss_final = float(full_history["loss"][-1])

        train_results.update({"loss_train": train_loss_final, "epochs": epochs, "time_taken_per_epoch": time_per_epoch})

        # Test loss
        test_input = [x[test_indices] for x in X]
        test_output = [y[test_indices]]
        test_loss = model.evaluate(test_input, test_output)
        train_results["loss_test"] = test_loss

        if "val_loss" in full_history:
            val_loss_best = float(np.min(full_history["val_loss"]))
            train_results["loss_val"] = val_loss_best

            val_weighted = len(val_indices) / len(non_test_indices) * val_loss_best
            train_weighted = len(train_indices) / len(non_test_indices) * train_loss_final
            combined_loss_final = val_weighted + train_weighted
        else:
            combined_loss_final = train_loss_final

        train_results["loss_combined"] = combined_loss_final

    # Prepare plotting directory
    if not train_results["crashed"]:
        plots_dir = os.path.join(random_projects_path, project_name, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Plot development of parameters and animation of resulting concentration curves
        plot_training_parameters(
            history, log_train, progress_check_frequency, write_plot=os.path.join(plots_dir, "params_train.png")
        )
        plt.clf()

        plot_training_parameters(
            history, log_val, progress_check_frequency, write_plot=os.path.join(plots_dir, "params_val.png")
        )
        plt.clf()

        plot_in_one(
            log_prediction_train,
            y_full[train_indices_plot],
            epochs,
            t_train,
            plot_label="Training",
            write_plot=os.path.join(plots_dir, "all_in_one_train.png"),
        )
        plt.clf()

        plot_in_one(
            log_prediction_val,
            y_full[val_indices_plot],
            epochs,
            t_val,
            plot_label="Validation",
            write_plot=os.path.join(plots_dir, "all_in_one_val.png"),
        )
        plt.clf()

        # Plot test results
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        train_results["epoch_best"] = best_epoch
        plot_in_one(
            log_prediction_test,
            y_full[test_indices_plot],
            [best_epoch],
            t_test,
            plot_label="Test",
            write_plot=os.path.join(plots_dir, "all_in_one_test.png"),
        )
        plt.clf()

    # Also add root of all losses
    for key in list(train_results.keys()):
        val = train_results[key]
        if "loss" in key and type(val) in [float, int]:
            root_val = np.sqrt(val)
            root_key = f"{key}_sqrt"
            train_results[root_key] = root_val

    # Write first few losses (train and val) into eval.
    # The purpose is that this allows us to see how quickly loss improves initially.
    if not train_results["crashed"]:
        improving_loss = train_results["improving_loss"] = {}
        for split_name, split_loss_key in [("train", "loss"), ("val", "val_loss")]:
            for epoch in range(0, 10, 2):
                if len(full_history[split_loss_key]) > epoch:
                    loss_value = full_history[split_loss_key][epoch]
                else:
                    loss_value = None
                improving_loss[f"{split_name}_epoch_{epoch + 1}"] = loss_value

    # Write results
    write_json(train_results, os.path.join(random_projects_path, project_name, "eval.json"))

    # Rename project dir with val loss for easy spotting of successful runs
    if not train_results["crashed"]:
        status_token = f"{val_loss_best:0.4E}"
    else:
        status_token = "crashed"
    project_name_tokens = project_name.split("_")
    new_project_name = "_".join(project_name_tokens[:-1] + [status_token])
    shutil.move(os.path.join(random_projects_path, project_name), os.path.join(random_projects_path, new_project_name))

    # Also rename project json file
    shutil.move(
        os.path.join(random_projects_path, new_project_name, f"{project_name}.json"),
        os.path.join(random_projects_path, new_project_name, f"{new_project_name}.json"),
    )

    logging.info(f"Run {new_project_name} complete.")


def enumerated_search():
    # Define search space
    # activation = ["tanh", "elu"]
    activation = ["relu", "tanh"]
    activation_output_linear = [True, False]
    num_hidden_layers = [1, 2, 3]
    size_hidden_layers = [2, 5, 10, 15, 20]
    use_random_regularization = [False]
    # arithmetic_transformation = ["identity", "addition_same_mag", "addition_smaller_mag", "product"]
    arithmetic_transformation = ["identity", "addition_same_mag", "addition_smaller_mag", "product"]
    # learning_rate = [0.1, 0.01, 0.001, 0.0001]
    learning_rate = [0.01, 0.001, 0.0001]
    optimizer = [
        tf.optimizers.RMSprop,
        tf.optimizers.Adam,
        # tf.optimizers.Adamax,
        tf.optimizers.Adagrad,
        # tf.optimizers.SGD,
        # tf.optimizers.Adadelta,
        # tf.optimizers.Ftrl,
        # tf.optimizers.Nadam,
    ]

    # Figure out which part of the combinatorial space this run is supposed to take over
    args = parse_args()
    total_parts = args.parts_total
    our_part_idx = args.part - 1  # e.g. part 1 -> modulo of idx with total parts number will be 0
    logging.info(
        f"Doing part {our_part_idx + 1} of {total_parts} "
        f"(performing runs with idx mod {total_parts} == {our_part_idx})."
    )

    # Enumerate all combinations of options
    opt_product = list(
        enumerate(
            product(
                activation,
                activation_output_linear,
                num_hidden_layers,
                size_hidden_layers,
                use_random_regularization,
                arithmetic_transformation,
                optimizer,
                learning_rate,
            )
        )
    )
    logging.info(f"There are {len(opt_product)} combinations of options.")

    # Duplicate the combos so that we have multiple runs for a single combo
    num_runs_of_same_combo = 20
    opt_product = num_runs_of_same_combo * opt_product
    logging.info(f"{num_runs_of_same_combo} runs per combo, so {len(opt_product)} runs in total.")

    for run_part_idx, options in enumerate(opt_product):

        # Only run this combination if it is in our part
        part_idx = run_part_idx % total_parts
        if part_idx != our_part_idx:
            logging.info(
                f"Not doing run part idx {run_part_idx} since it does not belong to our part idx {our_part_idx}."
            )
        else:
            logging.info(f"Running run part idx {run_part_idx}. It belongs to our part idx {our_part_idx}!")

            # Run the combination
            logging.info(f"Running run with idx {run_part_idx} of {len(opt_product)} runs")
            combination_idx, options = options
            new_random_project(combination_idx=combination_idx, options=options)


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search")
    parser.add_argument(
        "--part",
        type=int,
        help="part of the total search space this run should do, e.g. if " "parts_total is 3, part can be 1, 2, or 3.",
        required=True,
    )
    parser.add_argument(
        "--parts_total", type=int, help="total number of parts, e.g. for 2, there are parts 1 and 2", required=True
    )

    args = parser.parse_args()
    return args


# Run a new project with random alterations
if __name__ == "__main__":
    enumerated_search()
