import os
import re

import numpy as np
import pandas as pd

id_col = "IndividualId"
time_col = "Time"
conc_col = "plasmaconc"
gender_col = "Gender"


class FluxDataLoader:
    def __init__(self):
        # Paths to dataset
        self.base_path = os.path.join(os.path.dirname(__file__), "projects", "data", "Fluvoxamine")
        self.pop_path = os.path.join(self.base_path, "Population based on de Vries-Population.csv")
        self.pk_path = os.path.join(self.base_path, "Population based on de Vries 1993_50mgSD_NA_NA-Results.csv")
        self.cov_path = os.path.join(self.base_path, "covariateToUseTable.xlsx")

    def load(self):
        # Load Datasets
        pop = pd.read_csv(self.pop_path)
        pk = pd.read_csv(self.pk_path)

        # Load meta information: covariates selection
        if self.cov_path.endswith(".xlsx"):
            covs = pd.read_excel(self.cov_path, engine="openpyxl")
        else:
            covs = pd.read_csv(self.cov_path)

        # Remove unimportant columns from Constrain results table (which contains the measured drug concentrations)
        pk.columns = [id_col, time_col, conc_col]

        # Find out from meta data which covariates are to be used
        covariates_used = covs[covs["use"] == "yes"]["x"].to_list()

        # Limit pop to the covariates to be used (steps 1 to 3)

        # 1) The columns in the covariateToUseTable do not contain special characters, due to encoding issues.
        # The first step is to reconstruct the malformed column names from the pop columns.
        pop_cols_mangled_index = {}
        for col in pop.columns:
            col_mangled = re.sub("[\[\]\(\)\-| /]", ".", col)
            col_mangled = re.sub("²", "Â²", col_mangled)
            pop_cols_mangled_index[col_mangled] = col

        # 2) Find out normally-encoded covariate names corresponding to chosen covariates
        cols_to_use = [pop_cols_mangled_index[cov] for cov in covariates_used]
        if id_col not in cols_to_use:
            cols_to_use = [id_col] + cols_to_use

        # 3) Limit to selected columns
        pop_filtered = pop.filter(cols_to_use)

        # Normalize data in pop_filtered
        for col in pop_filtered.columns:
            if col in [id_col]:
                continue
            elif col == gender_col:
                pop_filtered[col] -= 1.5  # Gender is either 1 or 2; subtracting 1.5 moves this to -0.5 and 0.5
            else:
                pop_filtered[col] -= pop_filtered[col].mean()
                pop_filtered[col] /= pop_filtered[col].std()

        # Assemble all data for the different subjects in the data

        # Group concentration information by subject
        subjects = pk.groupby(pk[id_col])
        subject_ids = sorted(set(pop_filtered[id_col]))

        # Process concentrations and times of measurement for each of the subjects
        times = []
        concentrations = []
        for subject_id in subject_ids:
            subject_concentrations = subjects.get_group(subject_id)
            times.append(subject_concentrations[time_col].values)
            concentrations.append(subject_concentrations[conc_col].values)

        # Identify the line indices for the subjects
        subject_line_indices = [pop.index[pop[id_col] == subj_id] for subj_id in subject_ids]
        subject_line_indices = [col_index[0] for col_index in subject_line_indices if len(col_index) > 0]

        # Build list: each element is array with subject ID as first element, followed by the covariates
        covariates_for_subjects = [pop_filtered.loc[i].values for i in subject_line_indices]

        # Remove subject ID
        assert cols_to_use[0] == id_col, f"Expecting the first covariate to be subject ID ({id_col})!"
        covariates_for_subjects = [sub_covariates[1:] for sub_covariates in covariates_for_subjects]

        X = [np.array(covariates_for_subjects), np.array(times)]
        y = np.array(concentrations)
        return X, y
