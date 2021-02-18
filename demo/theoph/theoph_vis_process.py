import sys
import os

# Add location of HybridML to path
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt  # noqa: E402
from HybridML.utils import FileUtil  # noqa: E402

from theoph_vis_results import get_data, plot_lot, projects_path  # noqa: E402


def load_checkpoint_models(project, num_checkpoints, freq=10):
    model_descriptions = project.project_file.model_descriptions
    datas = []
    for model_desc in model_descriptions:
        data = FileUtil.load_json(model_desc)
        datas.append(data)
    datas = datas[:1]
    models = []
    for desc, data in zip(["theoph_demo_2_compartments"], datas):
        # for desc, data in zip(["theoph_demo_2_compartments", "theoph_demo_3_compartments"], datas):
        for i in range(num_checkpoints):
            if i % freq != 0:
                continue
            path = os.path.join(projects_path, "theoph_demo", "models", desc + f"_epoch-{i}.h5")
            model = project.model_creator.load_model_from_file(data, path)
            model.name = f"{desc}-epoch-{i}"
            models.append(model)
    return models


if __name__ == "__main__":

    project, X_predict, X_real, y_real = get_data()

    models = load_checkpoint_models(project, 5, freq=1)

    models_to_predict = models

    plot_data = []
    for model in models_to_predict:
        prediciton = model.predict(X_predict)
        plot_data.append((X_predict, prediciton, model.name))

    fig, axs = plt.subplots(1, len(plot_data), sharex=True, sharey=True, figsize=(20, 10))
    plot_lot(plot_data, (X_real, y_real), axs)
    plt.show()
