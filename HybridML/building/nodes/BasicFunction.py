from ..BaseBuilders import NodeBuilder
from ..DataModel import BuiltNodeContainer
from HybridML.parsing.nodes.BasicFunction import basic_function_names, basic_function_layers, BasicFunctionNode


class BasicFunctionNodeBuilder(NodeBuilder):
    def __init__(self):
        super().__init__(basic_function_names)

    def build(self, node: BasicFunctionNode, inputs) -> BuiltNodeContainer:
        keras_layer = basic_function_layers[node.type]()
        output = keras_layer(inputs)
        return BuiltNodeContainer(node, output, keras_layer)
