#%%
import sys
import os
import numpy as np

# Add location of HybridML to path
sys.path.append(os.getcwd())

from HybridML import Project  # noqa: E402
from HybridML.utils import FileUtil  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# Load project
base_dir = os.path.join(".", "demo")
projects_path = os.path.join(base_dir, "projects")
project = Project.open_create(projects_path, "theoph_demo")
#%%


def load_checkpoint_models():
    model_descriptions = project.project_file.model_descriptions
    datas = []
    for model_desc in model_descriptions:
        data = FileUtil.load_json(model_desc)
        datas.append(data)
    datas = datas[:1]
    models = []
    for desc, data in zip(["theoph_demo_2_compartments"], datas):
        # for desc, data in zip(["theoph_demo_2_compartments", "theoph_demo_3_compartments"], datas):
        for i in range(50):
            if i % 10 != 0:
                continue
            path = os.path.join(projects_path, "theoph_demo", "models", desc + f"_epoch-{i}.h5")
            model = project.model_creator.load_model_from_file(data, path)
            model.name = f"{desc}-epoch-{i}"
            models.append(model)
    return models


# models = load_checkpoint_models()
models = project.load_models()[:1]


# Date should be manually split into respective splits (train, val, test) by you. It is loaded here from csv files in
# `demo/projects/extrapolation_demo/`
data_sources = project.load_data_sources()
ds = data_sources["theoph"]

X_, y = ds.get_train_data()


def transform_data(X, y, i=2):
    return [X[0][:i], X[1][:i], X[2][:i]], y[:i]


X_, y = transform_data(X_, y)

time_space = np.linspace(0, 25, 30)
X = X_.copy()
X[-1] = np.array([time_space for i in range(X_[-1].shape[0])])
plot_data = []
# models_to_predict = [models[0], models[9], models[10], models[19]]
models_to_predict = models

for model in models_to_predict:
    prediciton = model.predict(X)
    plot_data.append((prediciton, model.name))


def plot_lot(plot_data, _axs=None):
    if _axs is None:
        fig, axs = plt.subplots(1, len(plot_data), sharex=True, sharey=True)
        fig.suptitle(plot_data[0][1][:-8])

    else:
        axs = _axs
    for i, (prediction, name) in enumerate(plot_data):
        ax = axs[i] if len(plot_data) > 1 else axs

        if i == 0:
            ax.legend(["True", "Prediction"])
            ax.set_ylabel("Concentration")
            ax.set_xlabel("Time")

        samples = prediction.shape[0]
        for i in range(samples):
            sns.scatterplot(X_[-1][i], y[i], ax=ax)
            sns.lineplot(X[-1][i], prediction[i], ax=ax)
    if _axs is not None:
        plt.show()


plot_lot(plot_data)
# axs = [None, None]
# # fig, axs = plt.subplots(2, len(plot_data) // 2, sharex=True, sharey=True, figsize=(20, 10))
# plot_lot(plot_data[: len(plot_data) // 2], axs[0])
# plot_lot(plot_data[len(plot_data) // 2 :], axs[1])
# # plt.show()
# %%
for model in models:
    print(model.name)
    model.evaluate(X_, y)


# %%
verbose_prediction = models[1].predict(X_, consider_additional_outputs=True)


# %%
len(verbose_prediction)


pred_dict = dict(zip(["prediction", "ka", "k12", "k21", "k10"], verbose_prediction))
fig, ax = plt.subplots(1, 4, sharex=True, sharey=True)
for name, values in pred_dict.items():
    if name == "prediction":
        continue
    ax.plot()


# %%

for model in models:
    verbose_prediction = model.predict(X_, consider_additional_outputs=True)
    abc = enumerate(zip(["prediction", "ka", "k12", "k21", "k10"], verbose_prediction))
    values = [(name, value) for i, (name, value) in abc if i != 0]
    weights = X[0]
    #%%
    print(model.name)
    fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(10, 10))
    for i in range(len(values)):
        ax = axs[i]
        ax.set_title(values[i][0])
        ax.scatter(weights, values[i][1])
        ax.plot(weights, values[i][1])
        ax.set_xlabel("Weight")
    fig.suptitle(model.name)
    plt.show()


# %%
from tensorflow.keras.layers import Input  # noqa E402
from tensorflow.keras.models import Model  # noqa E402

X = X_
ode_layer = models[1].model.layers[-1]
ode_layer.input
I_k = Input(shape=(4,))
I_x_init = Input(shape=(3,))
I_t = Input(shape=(None,))
out = ode_layer([I_k, I_x_init, I_t])
new_model = Model(inputs=[I_k, I_x_init, I_t], outputs=[out])

k = np.array([[0.05775208, -0.09672820, 1.16089870, 2.13519472]])

x_init = np.array([[np.mean(X[1]), 0, 0]])
t = np.array([np.linspace(0, 25, 30)])

manual_prediction = new_model.predict([k, x_init, t])
#%%
manual_prediction.shape
#%%
samples = 11
fig, ax = plt.subplots()
for i in range(samples):
    sns.scatterplot(X_[-1][i], y[i], ax=ax)
sns.lineplot(t[0], manual_prediction[0], ax=ax)
ax.set_ylabel("Concentration")
ax.set_xlabel("Time")
ax.set_title(new_model.name)
plt.show()

# %%
