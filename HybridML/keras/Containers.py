import os
from HybridML.building.DataModel import ModelContainer


class KerasHistoryContainer:
    def __init__(self, history):
        self.history = history


class KerasModelContainer(ModelContainer):
    """Contains a Tensorflow model and wraps its methods."""

    def __init__(self, model, network, additional_outputs_model=None):
        super().__init__(model, network)
        self.trainable = model._is_compiled
        # Additional outputs don't have training data for them.
        # So we have two models, sharing the same weights. One for training and the additional outputs model
        self.additional_outputs_model = additional_outputs_model if additional_outputs_model else model

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def save(self, model_dir, name=None):
        """Create directory and save model to file."""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        name = name or self.name
        path = os.path.join(model_dir, name + ".h5")
        self.save_to_file(path)

    def save_to_file(self, file):
        self.additional_outputs_model.save(file)

    def summary(self):
        self.model.summary()

    def fit(self, x, y, validation_data=None, shuffle=None, validation_split=None, epochs=1, callbacks=None, **kwargs):
        history = self.model.fit(
            x=x,
            y=y,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_split=validation_split,
            shuffle=shuffle,
        )
        return KerasHistoryContainer(history)

    def evaluate(self, x, y, **kwargs):
        return self.model.evaluate(x=x, y=y, **kwargs)

    def predict(self, data, consider_additional_outputs=False):
        if consider_additional_outputs:
            return self.additional_outputs_model.predict(x=data)
        else:
            return self.model.predict(x=data)
