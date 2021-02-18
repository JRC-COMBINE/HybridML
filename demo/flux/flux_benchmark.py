import sys
import os
import time
from flux_benchmark_data import get_data


# Add location of HybridML to path
sys.path.append(os.getcwd())
from HybridML.keras.layers.LinearOde import LinearOdeLayer  # noqa: E402
from HybridML.keras.layers.CasadiLinearOde import CasadiLinearOdeLayer  # noqa: E402
from HybridML.keras.layers.TensorflowLinearOde import TensorflowLinearOdeLayer  # noqa: E402


layers = [LinearOdeLayer(), CasadiLinearOdeLayer(), TensorflowLinearOdeLayer()]
names = [layer.__class__.__name__ for layer in layers]


# Prepare input data
# 19 samples taken from the fluvoxamine use case
A_list, x_init_list, t_list = get_data()

durations = []
for layer in layers:
    start = time.time()
    result = layer([A_list, x_init_list, t_list])
    end = time.time()
    durations.append(end - start)

print(f"Integrating: {A_list.shape[0]} samples with each {t_list.shape[-1]} time points.")
print(f"Maximum time point: {t_list.max():.0f}")
for name, duration in zip(names, durations):
    print(name, f"took {duration:.4f}s.")
print("ok.")
