#%%
import pandas as pd

import os

"""
Normalization of the theoph dataset.
The time is scaled down to std=1 in order for the ode-solver to work faster
Concentration is also kept above 0, as the ode-model would not work otherwise.
The other parameters are normalized to std=1 and mean=0.
"""


def make_relative_path(file_name):
    return os.path.join(os.path.split(__file__)[0], file_name)


theoph = pd.read_csv(make_relative_path("theoph.csv"))
#%%
subj_col, wt_col, dose_col, time_col, conc_col = theoph.columns


for col in [wt_col, dose_col, conc_col]:
    theoph[col] /= theoph[col].max()

print(theoph.describe())

# %%
theoph.to_csv(make_relative_path("theoph_normalized.csv"), index=False)
