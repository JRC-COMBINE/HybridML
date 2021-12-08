import tensorflow as tf
from HybridML.parsing.BaseParsers import NodeParser
from HybridML.parsing.DataModel import ParsedNode


class JSON_DICT:
    model_path = "model_path"


class ExistingModelNode(ParsedNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_loaded = False
        self.model = None
        self.model_path = self.data[JSON_DICT.model_path]

    def __str__(self):
        return f"ExistingModel {self.id} type: {self.type}"

    def load_model(self):
        if self.is_loaded:
            return
        self.model = tf.keras.models.load_model(self.model_path)
        self.model.trainable = False

    def determine_output_sizes(self):
        self.load_model()
        outputs = self.model.output
        if not isinstance(outputs, list):
            outputs = [outputs]
        output_sizes = [output.shape[1:] for output in outputs]
        return output_sizes


class ExistingModelNodeParser(NodeParser):
    def __init__(self):
        super().__init__("existing_model")

    def parse(self, data) -> ParsedNode:
        return ExistingModelNode(data)
