from HybridML.keras.layers.ArithmeticExpression import find_vars_in_expression

from ..BaseParsers import NodeParser
from ..DataModel import ParsedNode


class json_dict:
    ids = ["linear_ode", "casadi_linear_ode", "tf_linear_ode"]
    initial_value_input = "initial_value_input"
    system_matrix = "system_matrix"
    time_series_input = "time_series_input"
    output = "output"


class LinearOdeNode(ParsedNode):
    def __init__(
        self,
        data,
        time_series_id=None,
        init_value_id=None,
        output_id=None,
        system_matrix_str=None,
        system_matrix_ids=None,
    ):
        super().__init__(data)
        self.time_series_id = data[json_dict.time_series_input]
        self.init_value_id = data[json_dict.initial_value_input]
        self.output_id = data[json_dict.output]
        self.system_matrix_str = data[json_dict.system_matrix]
        # analyse expression to find variables
        self.system_matrix_ids = find_vars_in_expression(self.system_matrix_str)

    def __str__(self):
        return f"LinearOdeNode: {self.id}"

    def get_parameter_ids(self):
        return sorted(self.system_matrix_ids)

    def get_input_ids(self):
        return self.get_parameter_ids() + [self.init_value_id, self.time_series_id]

    def get_output_ids(self):
        return [self.output_id]

    def determine_output_sizes(self):
        # samples x time x compartements
        time_input = self.inputs[-1]
        init_input = self.inputs[-2]
        return [(time_input, init_input)]


class LinearOdeNodeParser(NodeParser):
    def __init__(self):
        super().__init__(json_dict.ids)

    def parse(self, data) -> ParsedNode:

        node = LinearOdeNode(
            data,
        )
        return node
