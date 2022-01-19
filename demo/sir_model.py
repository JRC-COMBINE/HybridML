import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())

from HybridML import Project  # noqa: E402

model_description = {
    "name": "sir_model",
    "comment": "",
    "inputs": [
        {"id": "beta", "size": 1},
        {"id": "gamma", "size": 1},
        {"id": "x_init", "size": 3},
        {"id": "t", "size": None}
    ],
    "nodes": [
        {"id": "arithmetic", "type": "arithmetic", "expression": "n = x_init[:,0]+x_init[:,1]+x_init[:,2]"},
        {
            "id": "linear_ode",
            "type": "ode",
            "rhs": "[-beta * x[0] * x[1] / n, beta * x[0] * x[1] / n -gamma * x[1], gamma * x[1]]",
            "time_series_input": "t",
            "initial_value_input": "x_init",
            "output": "ode_output"
        }
    ],
    "outputs": ["ode_output"],
    "additional_outputs": [],
    "loss": "mse",
    "optimizer": "adam",
    "metrics": []
}
model = Project.create_model_from_description(model_description)

# Input Data
x0 = np.array([[997, 3, 0]])
t = np.arange(100)[np.newaxis]
beta = np.ones((1, 1)) * 0.4
gamma = np.ones((1, 1)) * 0.04

# Predict and Plot
result = model.predict([beta, gamma, x0, t])

plt.plot(result[0])
plt.legend("SIR")
plt.xlabel("Days")
plt.ylabel("Cases")
plt.show()
