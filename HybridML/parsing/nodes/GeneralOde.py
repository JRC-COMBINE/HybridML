from HybridML.keras.layers.ArithmeticExpression import find_vars_in_expression

from ..BaseParsers import NodeParser
from ..DataModel import ParsedNode


class json_dict:
    ids = ["ode"]
    initial_value_input = "initial_value_input"
    time_series_input = "time_series_input"
    output = "output"
    rhs = "rhs"


class BaseOdeNode(ParsedNode):
    def __init__(self, data):
        super().__init__(data)
        self.time_series_id = data[json_dict.time_series_input]
        self.init_value_id = data[json_dict.initial_value_input]
        self.output_id = data[json_dict.output]

    def get_input_ids(self):
        return self.get_parameter_ids() + [self.init_value_id, self.time_series_id]

    def get_output_ids(self):
        return [self.output_id]

    def determine_output_sizes(self):
        # samples x time x compartements
        time_input = self.inputs[-1]
        init_input = self.inputs[-2]
        return [(time_input, init_input)]


class GeneralOdeNode(BaseOdeNode):
    def __init__(self, data):
        super().__init__(data)
        self.parameter_ids = None
        self.rhs = data[json_dict.rhs]

    def __str__(self):
        return f"GeneralOdeNode: {self.id}"

    def get_parameter_ids(self):
        if self.parameter_ids is None:
            self.parameter_ids = sorted(find_vars_in_expression(self.rhs))            
            if "x" in self.parameter_ids:
                self.parameter_ids.remove("x")
        return self.parameter_ids


class GeneralOdeNodeParser(NodeParser):
    def __init__(self):
        super().__init__(json_dict.ids)

    def parse(self, data) -> ParsedNode:

        node = GeneralOdeNode(
            data,
        )
        return node
