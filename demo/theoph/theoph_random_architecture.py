import sys
import os
import shutil
import time
import logging
import humanfriendly
import tensorflow as tf

import random
import string
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.getLogger().setLevel(logging.INFO)

# Add location of HybridML to path
sys.path.append(os.getcwd())

from HybridML import Project  # noqa: E402


# General setup
base_dir = os.path.join(".", "demo")
projects_path = os.path.join(base_dir, "projects")
random_projects_path = os.path.join(os.getenv("WORK"), "hybrid_platform", "random_search")


def new_random_project():
    # Copy the project from a template and modify it randomly. Then run the project.
    random_id = "".join(
        [random.choice(string.ascii_uppercase) for _ in range(2)]
        + ["_"]
        + [random.choice(string.digits) for _ in range(3)]
    )
    new_project_name = f"random_project_unfinished_{random_id}"
    new_project_dir = os.path.join(random_projects_path, new_project_name)
    os.makedirs(new_project_dir, exist_ok=True)
    logging.info(f"Starting new random project with name {new_project_name}")

    new_model_json_name = "theoph_model_3_compartments_random"
    new_model_json_filename = f"{new_model_json_name}.json"
    new_model_json_path = os.path.join(new_project_dir, new_model_json_filename)
    shutil.copy(
        os.path.join(projects_path, "theoph_ode_only", "theoph_model_3_compartments_one_bb.json"), new_model_json_path
    )

    new_project_json_name = new_project_name
    new_project_json_path = os.path.join(new_project_dir, f"{new_project_json_name}.json")
    shutil.copy(os.path.join(projects_path, "theoph_ode_only", "theoph_ode_only.json"), new_project_json_path)

    shutil.copy(os.path.join(projects_path, "theoph_ode_only", "theoph_normalized.csv"), os.path.join(new_project_dir))

    # Register the model file in the project file
    proj_file = load_json(new_project_json_path)
    proj_file["name"] = new_project_name
    proj_file["models"] = [new_model_json_filename]
    write_json(proj_file, new_project_json_path)

    # Modify model file randomly
    model_file = load_json(new_model_json_path)

    # Change black box layers and activations
    bb = [node for node in model_file["nodes"] if node["id"] == "BB"][0]
    bb["layers"] = []
    for layer_idx in range(random.randint(1, 3)):
        bb["layers"].append(
            {"size": random.randint(4, 16), "activation": random.choice(["relu", "tanh", "sigmoid", "linear", "elu"])}
        )
    bb["layers"][-1]["size"] = 4  # Set last layer's size to 4 (to generate 4 parameters)

    # Randomly disable scaling of parameters
    if random.choice([True, False]) or True:
        # Remove scale nodes
        model_file["nodes"] = [node for node in model_file["nodes"] if not node["id"].endswith("_scale")]

        # Rename parameters in optimum value nodes such that it works without the scaling step
        for node in [node for node in model_file["nodes"] if node["id"].endswith("_opt")]:
            node["expression"] = node["expression"].replace("pp_", "p_")

    # Perform optimum value nodes either as addition of the optimal value or as multiplication with the optimal value
    operation = random.choice(["+", "*"])
    for node in [node for node in model_file["nodes"] if node["id"].endswith("_opt")]:
        node["expression"] = node["expression"].replace(" + ", f" {operation} ")

    # Save the changed model
    write_json(model_file, new_model_json_path)

    # Init random optimizer
    opt_classes = [
        tf.optimizers.RMSprop,
        tf.optimizers.Adam,
        tf.optimizers.Adamax,
        tf.optimizers.Adagrad,
        tf.optimizers.SGD,
        tf.optimizers.Adadelta,
        tf.optimizers.Ftrl,
        tf.optimizers.Nadam,
    ]
    opt_class = random.choice(opt_classes)
    lr = random.choice([0.1, 0.01, 0.001, 0.0001]) * random.choice([1, 2.5, 5])
    opt = opt_class(lr=lr)

    # Run training
    train_project(
        project_name=new_project_name,
        epochs=2 ** random.randint(1, 7),
        optimizer=opt,
        loss=random.choice(["mse", "mae"]),
    )


def load_json(path):
    with open(path, "r") as json_file:
        obj = json.load(json_file)
    return obj


def write_json(obj, path):
    with open(path, "w") as json_file:
        json.dump(obj, json_file, sort_keys=True, indent=4, separators=(",", ": "))


def train_project(project_name, epochs, optimizer, loss):
    # Load project
    project = Project.open_create(random_projects_path, project_name)

    # Generate the model out of the model definitions stored in the demo project
    models = project.generate_models()
    model = models[0]  # (there is only a single model for this project)

    # Use custom optimizer for model
    model.model.compile(optimizer=optimizer, loss=loss)
    train_results = {
        "loss": loss,
        "optimizer": {
            "name": optimizer._name,
            "hyper": {hyper_name: float(hyper_val) for (hyper_name, hyper_val) in optimizer._hyper.items()},
        },
    }

    # Load data
    data_sources = project.load_data_sources()
    ds = data_sources["theoph"]
    X, y = ds.get_train_data()

    # Model summary can help you identify problems with the model architecture
    model.summary()

    # Train the model
    begin_time = time.time()
    val_split = 0.1
    history = model.fit(X, y, validation_split=val_split, epochs=epochs)

    # Benchmark time
    time_taken = time.time() - begin_time
    time_per_epoch = time_taken / epochs
    logging.info(f"Time taken (at {epochs} epochs): {humanfriendly.format_timespan(time_taken)}")
    logging.info(f"Time per epoch: {humanfriendly.format_timespan(time_per_epoch)}")

    # Save results of training
    full_history = history.history.history
    train_loss_final = float(history.history.history["loss"][-1])
    val_loss_final = float(full_history["val_loss"][-1])
    combined_loss_final = val_split * val_loss_final + (1 - val_split) * train_loss_final
    train_results.update(
        {
            "loss_val": val_loss_final,
            "loss_train": train_loss_final,
            "loss_combined": combined_loss_final,
            "time_taken": time_taken,
            "epochs": epochs,
            "time_taken_per_epoch": time_per_epoch,
        }
    )

    # Ode layer benchmark

    # Plot fitted curves (concentration over time)
    plots_dir = os.path.join(random_projects_path, project_name, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    time_space = np.linspace(0, 25, 100)
    X_ = X.copy()
    X[-1] = np.array([time_space for i in range(X_[-1].shape[0])])
    print(model.name)
    prediction, par_ka, par_k12, par_k21, par_k10 = model.predict(X, consider_additional_outputs=True)
    samples = prediction.shape[0]

    # Print out final values for ODE parameters k
    # (values that Sebastian found out:
    #  ka          k10         k12         k21
    #  2.14311063  0.05806329 -0.08444969  1.15703847)
    for par_name, par_values in zip(["ka", "k12", "k21", "k10"], [par_ka, par_k12, par_k21, par_k10]):
        logging.info(f"Mean parameter {par_name}: {np.mean(par_values):0.8f}")
        train_results[par_name] = [float(v) for v in list(np.squeeze(par_values))]
        train_results[f"{par_name}_mean"] = float(np.mean(par_values))

    plot_dpi = 300

    fig, ax = plt.subplots()
    for i in range(samples):
        sns.scatterplot(X_[-1][i], y[i], ax=ax)
        sns.lineplot(X[-1][i], prediction[i], ax=ax)
    ax.set_ylabel("Concentration")
    ax.set_xlabel("Time")
    ax.set_title(model.name)
    plt.savefig(os.path.join(plots_dir, f"concentration_all_comb_loss_{combined_loss_final}.png"), dpi=plot_dpi)
    plt.clf()

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

    plt.savefig(os.path.join(plots_dir, f"concentration_separately_comb_loss_{combined_loss_final}.png"), dpi=plot_dpi)
    plt.clf()

    # Write results
    write_json(train_results, os.path.join(random_projects_path, project_name, "eval.json"))

    # Rename project dir with val loss for easy spotting of successful runs
    project_name_tokens = project_name.split("_")
    new_project_name = "_".join(project_name_tokens[:2] + [str(combined_loss_final)] + project_name_tokens[-2:])
    shutil.move(os.path.join(random_projects_path, project_name), os.path.join(random_projects_path, new_project_name))

    # Also rename project json file
    shutil.move(
        os.path.join(random_projects_path, new_project_name, f"{project_name}.json"),
        os.path.join(random_projects_path, new_project_name, f"{new_project_name}.json"),
    )

    logging.info(f"Random run {project_name} complete.")


# Run a new project with random alterations
new_random_project()
