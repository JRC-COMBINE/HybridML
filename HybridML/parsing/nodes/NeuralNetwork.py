from ..DataModel import ParsedNode
from ..BaseParsers import Parser, NodeParser


class NeuralNet_JSON_DICT:
    activation = "activation"
    layer = "layer"
    layers = "layers"
    size = "size"
    outputs = "outputs"
    kernel_regularizer = "kernel_regularizer"
    activity_regularizer = "activity_regularizer"


class Layer:
    """Represents a single layer."""

    def __init__(self):
        self.size: int = 0
        self.activation: str = ""
        self.type: str = ""


class NNNode(ParsedNode):
    def __init__(self, data):
        super().__init__(data)
        self.layers: [Layer] = []

    def __str__(self):
        return f"NNNode: {self.id}"

    def determine_output_sizes(self):
        inp_sizes = []
        for inp in self.inputs:
            if inp.size is None:
                raise Exception("Cannot calc output size. Not all input sizes are defined.")
            inp_sizes.append(inp.size)
        self.input_sizes = inp_sizes

        assert len(self.layers) > 0

        # Unpack the last layer's output into individual outputs (if possible)
        last_layer = self.layers[-1]
        output_size = last_layer.size

        # Is there a 1-to-1 match up of outputs to output dimensionality?
        if output_size == len(self.outputs):
            for output in self.outputs:
                output.size = 1
            return [1] * len(self.outputs)
        else:
            assert len(self.outputs) == 1, (
                "Can't have more than a single output if dimensionality of last layer is"
                " not equal to number of outputs!"
            )
            return [output_size]


class LayerParser(Parser):
    def __init__(self):
        super().__init__()

    def create_empty_layer(self):
        layer = Layer()
        layer.type = "dense"
        return layer

    def create_layer(self, layer_data):
        if isinstance(layer_data, int):
            raise Exception("Single number layer definition is deactivated. Please use expanded layer definition.")
        elif "outputs" in layer_data:
            raise Exception("Multiple output layers are deprecated.")
        else:
            return self.parse_layer_definition(layer_data)

    def parse(self, layer_data):
        layer = self.create_layer(layer_data)
        return layer

    def parse_layer_definition(self, layer_data):
        assert "activation" in layer_data
        assert "size" in layer_data
        layer = self.create_empty_layer()
        layer.activation = layer_data[NeuralNet_JSON_DICT.activation]
        layer.size = layer_data[NeuralNet_JSON_DICT.size]

        layer.kernel_regularizer = layer_data.get(NeuralNet_JSON_DICT.kernel_regularizer)
        layer.activity_regularizer = layer_data.get(NeuralNet_JSON_DICT.activity_regularizer)

        return layer


class NNNodeParser(NodeParser):
    def __init__(self, layer_parser=LayerParser()):
        super().__init__("nn")
        self.layer_parser = layer_parser

    def parse(self, data) -> ParsedNode:
        node = NNNode(data)

        layer_data = data[NeuralNet_JSON_DICT.layers]
        layers = []
        for layer_datum in layer_data:
            layers.append(self.layer_parser.parse(layer_datum))
        node.layers = layers

        return node
