import os
import sys

# Add location of HybridML to path
sys.path.append(os.getcwd())

from datasource import FluxDataLoader  # noqa: E402
from flux_demo import main  # noqa: E402


def make_config(flux_path=None):
    if not flux_path:
        flux_path = os.path.join(os.getcwd(), "demo", "flux")
    cov_path = os.path.join(flux_path, "projects", "structured_flux_demo", "parameter selection.csv")

    data_loader = FluxDataLoader()
    data_loader.cov_path = cov_path
    kwargs = {
        "project_name": "structured_flux_demo",
        "data_loader": data_loader,
        "progress_check_frequency": 10,
        "train_epochs": 100,
        "validation_split": 0.8,
        "time_points": 15,
        "split_covariates": True,
        "plot_endlessly": False,
    }
    return kwargs


if __name__ == "__main__":
    kwargs = make_config()
    main(**kwargs)
