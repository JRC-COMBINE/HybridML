from HybridML.keras.layers.ArithmeticExpression import find_vars_in_expression

from ..BaseParsers import NodeParser
from ..DataModel import ParsedNode
from .GeneralOde import BaseOdeNode


class json_dict:
    ids = ["linear_ode", "casadi_linear_ode", "tf_linear_ode"]
    system_matrix = "system_matrix"


class LinearOdeNode(BaseOdeNode):
    def __init__(self, data):
        super().__init__(data)
        self.system_matrix_str = data[json_dict.system_matrix]
        self.system_matrix_ids = find_vars_in_expression(self.system_matrix_str)

    def __str__(self):
        return f"LinearOdeNode: {self.id}"

    def get_parameter_ids(self):
        return sorted(self.system_matrix_ids)


class LinearOdeNodeParser(NodeParser):
    def __init__(self):
        super().__init__(json_dict.ids)

    def parse(self, data) -> ParsedNode:

        node = LinearOdeNode(
            data,
        )
        return node
