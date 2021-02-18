from ..DataModel import ParsedNode
from ..BaseParsers import NodeParser
from HybridML.keras.layers.ArithmeticExpression import find_variables_in_assignment


class ArithmeticExpression_JSON_DICT:
    expression = "expression"
    arithmetic = "arithmetic"


class ArithmeticExpressionNode(ParsedNode):
    """Node, containing an ArithmeticExpressionLayer."""

    def __init__(self, data):
        super().__init__(data)
        self.expression = data[ArithmeticExpression_JSON_DICT.expression]
        self.input_vars, self.output_vars = find_variables_in_assignment(self.expression)

    def __str__(self):
        return f"ExpressionNode {self.id}"

    def determine_output_sizes(self):
        output_size = len(self.output_vars)
        return [output_size] * len(self.outputs)

    def get_input_ids(self):
        return self.input_vars

    def get_output_ids(self):
        return self.output_vars


class ArithmeticExpressionNodeParser(NodeParser):
    def __init__(self):
        super().__init__(ArithmeticExpression_JSON_DICT.arithmetic)

    def parse(self, data) -> ParsedNode:
        return ArithmeticExpressionNode(data)
