class JSON_DICT:
    class Node:
        inputs = "inputs"
        outputs = "outputs"
        type = "type"
        id = "id"

    class NodeConnector:
        id = "id"
        size = "size"

    class Model:
        nodes = "nodes"
        inputs = "inputs"
        metrics = "metrics"
        optimizer = "optimizer"
        loss = "loss"
        outputs = "outputs"
        name = "name"
        comment = "comment"
        additional_outputs = "additional_outputs"


class ParsedNode:
    """Abstract representation of a node from a model-json"""

    def __init__(self, data, id=None, type=None):
        """
        :param id: str, unique id.
        :param type: str, type that identifies the node in the model-jso.n"""
        self.id: str = id if id else data[JSON_DICT.Node.id]
        self.type: str = type if type else data[JSON_DICT.Node.type]
        self.data = data
        self.inputs: [ParsedNodeConnector] = []
        self.outputs: [ParsedNodeConnector] = None
        self.output_sizes: [int] = []
        self.input_sizes: [int] = []

    def __str__(self):
        """String representation of the node."""
        return "ParsedNode: " + self.id

    def get_input_ids(self):
        """Returns the ids of the ParsedNodeConnector inputs."""
        return self.data[JSON_DICT.Node.inputs]

    def get_output_ids(self):
        """Returns the ids of the ParsedNodeConnector outputs."""
        return self.data[JSON_DICT.Node.outputs]

    def determine_output_sizes(self):
        """Calculates the sizes of the node outputs.
        :returns: List of integers."""
        raise NotImplementedError()


class ParsedNodeConnector:
    """Contains a named variable that is used as an input or output for a node in the model-json."""

    def __init__(self, id, size=None, input_node=None):
        self.id = id
        self.size = size
        self.input_node: ParsedNode = input_node
        self.output_nodes: [ParsedNode] = []
        self.model = None

    def __str__(self):
        return "ParsedNodeConnector: " + self.id


class ParsedNetwork:
    """Representation of the network."""

    def __init__(self):
        self.inputs = [ParsedNodeConnector]
        self.nodes: [ParsedNode] = []
        self.data_points = {}

    def extend_nodes(self, nodes):
        for node in nodes:
            node.net = self
            self.nodes.append(node)

    def extend_data_points(self, data_points):
        for k, v in data_points.items():
            if k in self.data_points:
                raise Exception(f"The NodeConnector-Id {k} is used multiple times")
            self.data_points[k] = v

    def set_inputs(self, inputs):
        self.extend_data_points({input.id: input for input in inputs})
        self.inputs = inputs


class ParsedModel:
    """Representation of a model."""

    def __init__(self):
        self.optimizer = ""
        self.metrics = []
        self.loss = ""
        self.network = None
        self.name = ""

    def is_compilable(self):
        return self.loss is not None and self.optimizer is not None
