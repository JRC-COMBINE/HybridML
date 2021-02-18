import tensorflow.keras as keras
import numpy as np


from HybridML.parsing.DataModel import ParsedNode
from HybridML.parsing.BaseParsers import NodeParser

basic_function_layers = {
    "addition": lambda: keras.layers.Add(),
    "multiplication": lambda: keras.layers.Multiply(),
    "substraction": lambda: keras.layers.Subtract(),
    "average": lambda: keras.layers.Average(),
    "maximum": lambda: keras.layers.Maximum(),
    "minimum": lambda: keras.layers.Minimum(),
    "concatenate": lambda: keras.layers.Concatenate(),
}

basic_function_names = list(basic_function_layers.keys())


class BasicFunctionNode(ParsedNode):
    def __str__(self):
        return f"BasicFunctionNode {self.id} type: {self.type}"

    def determine_output_sizes(self):
        sizes = [inp.size for inp in self.inputs]
        output_size = np.max(sizes)
        return [output_size] * len(self.outputs)


class BasicFunctionNodeParser(NodeParser):
    def __init__(self):
        super().__init__(basic_function_names)

    def parse(self, data) -> ParsedNode:
        return BasicFunctionNode(data)
