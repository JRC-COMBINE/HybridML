from ..BaseBuilders import NodeBuilder
from ..DataModel import BuiltNodeContainer
from HybridML.keras.layers.ArithmeticExpression import ArithmeticExpressionLayer
from HybridML.parsing.nodes.ArithmeticExpression import ArithmeticExpressionNode, ArithmeticExpression_JSON_DICT


class ArithmeticExpressionNodeBuilder(NodeBuilder):
    def __init__(self):
        super().__init__(ArithmeticExpression_JSON_DICT.arithmetic)

    def build(self, node: ArithmeticExpressionNode, inputs) -> BuiltNodeContainer:
        layer = ArithmeticExpressionLayer(node.expression, name=node.id)
        outputs = layer(inputs)
        return BuiltNodeContainer(node, outputs, layer)
