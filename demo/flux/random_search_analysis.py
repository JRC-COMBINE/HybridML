import glob
import logging
import os
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add location of HybridML to path
sys.path.append(os.getcwd())

from flux_random_architecture import load_json, random_projects_path  # noqa: E402

logging.getLogger().setLevel(logging.INFO)


Result = namedtuple("Result", ["loss", "path", "eval"])


def export_search_results():
    # Open search results
    logging.info("Opening results...")
    results = open_search_results()

    # Extract options used in each run
    logging.info("Extracting options and losses...")
    opts = [extract_opts_and_eval(res.eval) for res in results]

    # Add path to opts
    for res, opt in zip(results, opts):
        opt["path"] = os.path.split(res.path)[-1]

    # Add combination to opts (each possible combination of options has a unique combination number)
    for opt in opts:
        combination_number = int(opt["path"].split("_")[-3])
        opt["combo"] = combination_number

    # Compute statistics for opts with the same combination, i.e. different trials of the same option combination
    combination_stats(opts)

    # Build pandas table of results
    res_df = pd.DataFrame(data=opts)

    # Write results to CSV file
    csv_file_path = os.path.join(random_projects_path, "search_results.csv")
    logging.info(f"Writing CSV file with {len(res_df)} results to {csv_file_path}...")
    res_df.to_csv(csv_file_path)
    logging.info("Done!")


def combination_stats(opts):
    # Group opts by combination
    opts_by_combo = {}
    for opt in opts:
        c = opt["combo"]
        if c not in opts_by_combo:
            opts_by_combo[c] = []
        opts_by_combo[c].append(opt)

    # Determine crash rate for each combination
    for combination, combo_opts in opts_by_combo.items():
        num_crashed_opts = len([o for o in combo_opts if o["crashed"]])
        crash_rate = num_crashed_opts / len(combo_opts)
        for opt in combo_opts:
            opt["combo_crash_rate"] = crash_rate

    # Determine statistics of loss within each combination
    for combination, combo_opts in opts_by_combo.items():
        successful_opts = [o for o in combo_opts if not o["crashed"] and not np.isnan(o["loss_val"])]
        val_losses = [o["loss_val"] for o in successful_opts]
        if len(val_losses) > 0:
            loss_mean = np.nanmean(val_losses)
            loss_coefficient_of_variation = np.std(val_losses) / loss_mean
            for opt in successful_opts:
                opt["combo_val_loss_mean"] = loss_mean
                opt["combo_val_loss_coeff_var"] = loss_coefficient_of_variation

    # Print some statistics *about* the statistics
    crash_rates = [o["combo_crash_rate"] for o in opts]
    logging.info(f"Overall crash rate: {100 * np.mean(crash_rates):0.2f}%")

    # Crash confusion: How likely is it that a single combination sometimes leads to crashes and sometimes not? In
    # other words, how strongly does the combination determine the "willingness to crash" and how much of
    # it is up to randomness?
    crash_confusions = np.array([c if c <= 0.5 else c - 1 for c in crash_rates])  # e.g. [0.9, 0.2] -> [-0.1, 0.2]
    crash_confusions = np.abs(crash_confusions)  # the higher the number, the more confused is the combination
    crash_confusion = np.mean(crash_confusions)  # between 0 and 0.5
    logging.info(f"Mean crashing confusion: {crash_confusion:0.6f}")

    losses_mean = [o["combo_val_loss_mean"] for o in opts if "combo_val_loss_mean" in o]
    logging.info(f"Mean val loss: {np.mean(losses_mean)}")

    losses_cv = [o["combo_val_loss_coeff_var"] for o in opts if "combo_val_loss_coeff_var" in o]
    logging.info(f"Mean coefficient of variation: {np.mean(losses_cv)}")


def extract_opts_and_eval(eval_info):
    # Extract option values from eval: This flattens eval, e.g. making optimizer and its learning rate two distinct
    # "options" with values

    # Shorthands
    e = eval_info

    # Extract options
    activation, activation_output_linear, num_hidden_layers, size_hidden_layers = e["options"][:4]
    use_random_regularization, arithmetic_transformation, learning_rate = e["options"][4:]
    optimizer = e["optimizer"]["name"]

    # Extract info about possible crash
    crashed = e["crashed"]
    if crashed:
        crash_message = e["exception"]
    else:
        crash_message = ""

    # Seed
    seed_truly_random = e["seed_truly_random"]
    if not seed_truly_random:
        fixed_seed_used = e["seed"]
    else:
        fixed_seed_used = False

    # Compose opts
    opts = {
        "crashed": crashed,
        "crash_message": crash_message,
        "optimizer_name": optimizer,
        "fixed_seed_used": not seed_truly_random,
        "fixed_seed": fixed_seed_used,
        "activation": activation,
        "activation_output_linear": activation_output_linear,
        "num_hidden_layers": num_hidden_layers,
        "size_hidden_layers": size_hidden_layers,
        "use_random_regularization": use_random_regularization,
        "arithmetic_transformation": arithmetic_transformation,
        "learning_rate": learning_rate,
    }

    # Best epoch (if using early stopping: the epoch to which we revert after stopping)
    if not crashed and "epoch_best" in e:
        opts["epoch_best"] = e["epoch_best"]

    # Add information about loss
    opts.update({key: val for (key, val) in e.items() if key.startswith("loss")})

    # Add information about how loss improves when training
    if not crashed and "improving_loss" in e:
        opts.update({f"loss_{key}": val for (key, val) in e["improving_loss"].items()})

    return opts


def open_search_results():
    # List search results
    results = glob.glob(os.path.join(random_projects_path, "*"))
    results = [path for path in results if os.path.isdir(path)]

    # Filter out unfinished results
    results = [path for path in results if "unfinished" not in path]

    # Sort by loss (from low to high)
    losses = []
    for path in results:
        loss = os.path.split(path)[1].split("_")[-1]
        if loss == "crashed":
            loss = np.nan
        losses.append(loss)
    sorted_order = np.argsort(losses)  # Sort using numpy, built-in sort does not work well with inf and nan
    losses = [losses[idx] for idx in sorted_order]
    results = [results[idx] for idx in sorted_order]

    # Open eval files for all results: They will tell us about the options used for the runs
    results_infos = []
    for path, loss in tqdm(zip(results, losses), desc="Opening eval files...", total=len(results)):
        # Find eval
        eval_paths = glob.glob(os.path.join(path, "eval.json"))
        if len(eval_paths) != 1:
            continue  # Skip this run: It doesn't have eval info
        eval_path = eval_paths[0]

        # Open eval
        result_eval = load_json(eval_path)

        # Save result
        result = Result(path=path, loss=loss, eval=result_eval)
        results_infos.append(result)

    return results_infos


if __name__ == "__main__":
    export_search_results()
