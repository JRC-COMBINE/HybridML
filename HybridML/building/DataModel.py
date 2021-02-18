from HybridML.parsing.DataModel import ParsedNode, ParsedNodeConnector
from typing import List, Dict

"""Abstract base class. ModelContainer contains eg. a tensorflow model."""


class ModelContainer:
    def __init__(self, model, network):
        self.model = model
        self.network: NetworkContainer = network
        self.name: str = model.name
        self.trainable = False

    def summary(self):
        self.model.summary()

    def fit(self, x, y, validation_data=None, shuffle=None, validation_split=None, epochs=1, callbacks=None, **kwargs):
        raise NotImplementedError()

    def evaluate(self, x, y, **kwargs):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()

    def save(self, model_dir):
        raise NotImplementedError()


class NetworkContainer:
    """Contains the network of a model, that is created while parsing the model-json."""

    def __init__(
        self,
        inputs: List[ParsedNodeConnector] = [],
        data_points: Dict[str, ParsedNodeConnector] = {},
        nodes: List[ParsedNode] = [],
    ):
        self.data_points: Dict[str, BuiltNodeConnectorContainer] = data_points
        self.inputs: List[BuiltNodeConnectorContainer] = inputs
        self._nodes: BuiltNodeContainer = nodes


class BuiltNodeContainer:
    """Contains a single node of a model network."""

    def __init__(self, content, outputs, layer=None):
        self.content = content
        self.outputs = outputs
        self.id = content.id if content else None
        self.layer = layer


class BuiltNodeConnectorContainer:
    """Contains a NodeConnector (Named Variable) from the model network."""

    def __init__(self, id, content):
        self.id = id
        self.content = content
