from HybridML.parsing.nodes.ExistingModel import ExistingModelNode

from ..BaseBuilders import NodeBuilder
from ..DataModel import BuiltNodeContainer


class ExistingModelNodeBuilder(NodeBuilder):
    def __init__(self):
        super().__init__("existing_model")

    def build(self, node: ExistingModelNode, inputs) -> BuiltNodeContainer:
        node.load_model()
        output = node.model(inputs)
        return BuiltNodeContainer(node, output, node.model)
