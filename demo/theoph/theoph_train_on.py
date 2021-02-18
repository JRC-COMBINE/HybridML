import sys
import os

# Add location of HybridML to path
sys.path.append(os.getcwd())

from HybridML import Project  # noqa: E402
from HybridML.Visualization import show_training  # noqa: E402
from HybridML.utils.TensorBoard import prepare_tensorboard_callback  # noqa: E402

# Load project
base_dir = os.path.join(".", "demo")
projects_path = os.path.join(base_dir, "projects")
project = Project.open_create(projects_path, "theoph_demo")

models = project.load_models()


data_sources = project.load_data_sources()
ds = data_sources["theoph"]

X, y = ds.get_train_data()

log_dir = os.path.join(base_dir, "projects", "theoph_demo", "log")


for model in models:
    model.summary()

    tensorboard_callback = prepare_tensorboard_callback(log_dir, model.name)

    callbacks = [tensorboard_callback]
    validation_split = 0.2
    epochs = 5

    for i in range(epochs):
        # Fit the model to the data
        print(f"Epoch: {i}/{epochs}")
        history = model.fit(X, y, validation_split=validation_split, shuffle=False, epochs=1, callbacks=callbacks)
        model.save(os.path.join(project.project_file.model_dir), model.model.name + f"_epoch-{i}")

    model.save(project.project_file.model_dir)

    show_training(history, model_name=model.name)

project.save_models()
