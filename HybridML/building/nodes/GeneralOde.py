from HybridML.building.BaseBuilders import NodeBuilder
from HybridML.building.DataModel import BuiltNodeContainer
from HybridML.keras.layers.GeneralOde import GeneralOdeLayer
from HybridML.parsing.nodes.GeneralOde import GeneralOdeNode, json_dict


class GeneralOdeNodeBuilder(NodeBuilder):
    def __init__(self):
        super().__init__(json_dict.ids)

    def build(self, node: GeneralOdeNode, inputs) -> BuiltNodeContainer:
        layer = GeneralOdeLayer(node.rhs)
        outputs = layer(inputs)
        return BuiltNodeContainer(node, outputs, layer)
