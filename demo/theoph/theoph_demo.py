import sys
import os

# Add location of HybridML to path
sys.path.append(os.getcwd())

from HybridML import Project  # noqa: E402
import tensorflow as tf  # noqa: E402
import random  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


class HolzhammerOptimizer(tf.keras.optimizers.Optimizer):
    """
    This optimizer uses the Holzhammer Method [Mueller 2020] to drastically change the variables.
    The HolzhammerOptimizer now features intelligent adaptive learning_rate correction.
    """

    def __init__(self, learning_rate=100, decay=True, **kwargs):
        super(HolzhammerOptimizer, self).__init__(**kwargs, name="Holzhammer!")
        self.learning_rate = learning_rate
        self.decay = decay
        self.reset_trackers

    def reset_trackers(self):
        self.var_tracker = []
        self.grad_tracker = []
        self.name_tracker = []

    # def _resource_apply_dense(self, grad, var, apply_state=None):
    #     var.assign_sub(self.learning_rate * grad)
    #     self.learning_rate *= 0.8

    def get_config(self):
        pass

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        pass

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var.assign_sub(self.learning_rate * grad)
        self.var_tracker.append(var.numpy())
        self.grad_tracker.append(grad.numpy())
        self.name_tracker.append(var.name)
        if self.decay:
            self._apply_lr_decay()

    def _apply_lr_decay(self):
        r = random.random() * 0.18 + 0.91
        self.learning_rate *= 0.96 * r

    def _create_slots(self, var_list):
        pass


def main(plot=True):

    # Load project
    base_dir = os.path.join(".", "demo", "theoph")
    projects_path = os.path.join(base_dir, "projects")
    project = Project.open_create(projects_path, "theoph_demo")

    # Generate the model out of the model definitions stored in the demo project
    models = project.generate_models()

    # models = project.load_models()

    # Date should be manually split into respective splits (train, val, test) by you. It is loaded here from csv files in
    # `demo/projects/theoph_demo/`
    data_sources = project.load_data_sources()
    ds = data_sources["theoph"]

    X, y = ds.get_train_data()

    def transform_data(X, y, i=2):
        return [X[0][:i], X[1][:i], X[2][:i]], y[:i]

    X, y = transform_data(X, y)

    # Write logs to project directory
    # log_dir = os.path.join(base_dir, "projects", "theoph_demo", "log")

    # Model summary can help you identify problems with the model architecture
    for model in models[:1]:
        my_holzhammer = HolzhammerOptimizer(learning_rate=2)
        # model.model.compile(loss="mse", optimizer="adam")
        model.model.compile(loss="mse", optimizer=my_holzhammer)
        model.summary()

        # Tensorboard allows you to look into training progress even during training and may help you identify training
        # problems such as overfitting. We set up a `callback` for this here. Keras will deliver the most recent
        # information about the training progress to each of the callbacks.
        # Another useful callback (which is not shown here) is early stopping, which makes the training automatically abort
        # once it stops making progress in reducing the loss.
        # To read more about callbacks, refer to https://keras.io/callbacks/
        # tensorboard_callback = prepare_tensorboard_callback(log_dir, model.name)
        # callbacks = [tensorboard_callback]

        callbacks = []
        validation_split = 0.5
        epochs = 5

        grad_trackers = []
        var_trackers = []
        name_trackers = []

        for i in range(epochs):
            my_holzhammer.reset_trackers()
            # Fit the model to the data
            print(f"Epoch: {i}/{epochs}")
            model.fit(X, y, validation_split=validation_split, shuffle=False, epochs=1, callbacks=callbacks)
            model.save(os.path.join(project.project_file.model_dir), model.model.name + f"_epoch-{i}")

            grad_trackers.append(my_holzhammer.grad_tracker)
            var_trackers.append(my_holzhammer.var_tracker)
            name_trackers.append(my_holzhammer.name_tracker)
        if plot:
            val_dict = {}
            grad_dict = {}
            for var_index in range(len(grad_trackers[0])):
                var_name = name_trackers[0][var_index]
                var_values, grad_values = [], []
                for epoch in range(epochs):
                    var_values.append(var_trackers[epoch][var_index])
                    grad_values.append(grad_trackers[epoch][var_index])
                val_dict[var_name] = var_values
                grad_dict[var_name] = grad_values

            val_rows = [[val[0] for val in row] for row in val_dict.values()]
            grad_rows = [[val[0] for val in row] for row in grad_dict.values()]

            fig, axs = plt.subplots(2, len(val_rows), sharex=True)

            axs[0][0].set_title("Values by epochs")

            for i, (row1, row2) in enumerate(zip(val_rows, grad_rows)):
                axs[0][i].plot(row1)
                axs[1][i].plot(row2)
                axs[0][i].set_title(list(val_dict.keys())[i] + " Val")
                axs[1][i].set_title(list(val_dict.keys())[i] + " Grad")

            plt.show()
            model.save(project.project_file.model_dir)

    # Save models and load. They will be saved at `demo/projects/theoph_demo/models`
    project.save_models()


if __name__ == "__main__":
    main(plot=False)
