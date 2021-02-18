from .BaseParsers import Parser, NodeParser
from .DataModel import ParsedNode, ParsedModel, ParsedNetwork, ParsedNodeConnector, JSON_DICT


class NodeParserSelector(Parser):
    """Has a list of node parsers. Selects the right parsers, that can build a node out of the incoming data."""

    def __init__(self):
        super().__init__()
        self.parser_dict = {}

    def append_parser(self, parser: Parser):
        for n_type in parser.parses_types:
            self.parser_dict[n_type] = parser

    def parse(self, node_data) -> ParsedNode:
        """Select the right parser for the job and assign it to him."""
        if "type" not in node_data:
            raise Exception("Node data contains no type information")
        type = node_data[JSON_DICT.Node.type]
        if type not in self.parser_dict:
            raise Exception("Network node type unknown: " + type)
        res = self.parser_dict[type].parse(node_data)
        return res


class NetParser(Parser):
    """Parses the net from a model-json and saves it in an abstract representation."""

    def __init__(self, node_parser: NodeParser):
        super().__init__()
        self.node_parser: NodeParser = node_parser

    def _parse_nodes(self, nodes_data):
        nodes = []
        used_node_ids = []
        for node_data in nodes_data:
            node_id = node_data[JSON_DICT.Node.id]

            if node_id in used_node_ids:
                raise Exception(f"The node id '{node_id}' was is used multiple times")

            node = self.node_parser.parse(node_data)

            nodes.append(node)
        return nodes

    def _parse_node_connectors(self, nodes):
        data_points = {}
        # Read NodeConnector
        for node in nodes:

            # Create outputs for the node
            node_outputs = [ParsedNodeConnector(id, input_node=node) for id in node.get_output_ids()]
            node.outputs = node_outputs

            for output in node_outputs:
                if output.id in data_points:
                    raise Exception("ParsedNodeConnector ID not unique: " + output.id)
                data_points[output.id] = output
        return data_points

    def _parse_net_inputs(self, inputs_data):
        inputs = [
            ParsedNodeConnector(input_data[JSON_DICT.NodeConnector.id], input_data[JSON_DICT.NodeConnector.size])
            for input_data in inputs_data
        ]
        return inputs

    def _parse_network_structure(self, nodes_data, inputs_data):
        """Parses the nodes and connnects them."""
        net = ParsedNetwork()

        # Read Nodes
        nodes = self._parse_nodes(nodes_data)
        net.extend_nodes(nodes)

        # Read NodeConnectors
        data_points = self._parse_node_connectors(net.nodes)
        net.extend_data_points(data_points)

        # Read Network InputPoints
        inputs = self._parse_net_inputs(inputs_data)
        net.set_inputs(inputs)

        # Connect ParsedNodeConnectors to Inputs of the Nodes
        for node in net.nodes:
            self._connect_node_with_inputs(net.data_points, node)

        return net

    def _connect_node_with_inputs(self, data_points, node):
        """Finds the inputs of a node and connects it with them."""
        input_ids = node.get_input_ids()
        node_inputs = [data_points[iid] for iid in input_ids]

        node.inputs = node_inputs
        for inp in node_inputs:
            inp.output_nodes.append(node)

    def _propagate_sizes(self, nodes, inputs):
        """Walk through the network and propagate the sizes of the node connectors."""

        index = 0
        idle_loops = 0
        nodes_todo = nodes.copy()
        known_data_points = inputs.copy()
        while len(nodes_todo) > 0:
            index %= len(nodes_todo)
            if idle_loops > len(nodes_todo):
                raise Exception("Cant walk through network, probably due to a cycle.")
            node: ParsedNode = nodes_todo[index]

            all_inputs_known = all(inp in known_data_points for inp in node.inputs)

            if all_inputs_known:
                # Determine Output_Sizes for Node
                sizes = node.determine_output_sizes()
                for size, output in zip(sizes, node.outputs):
                    output.size = size

            # Book keeping for the loop
            if all_inputs_known:
                nodes_todo.remove(node)
                idle_loops = 0
            else:
                index += 1
                idle_loops += 1
                continue
            known_data_points.extend(node.outputs)

    def parse(self, nodes_data, inputs_data):
        """Parse a model-json and return a ParsedNetwork, ready for building."""
        net = self._parse_network_structure(nodes_data, inputs_data)
        self._propagate_sizes(net.nodes, net.inputs)
        return net


class ModelParser(Parser):
    """Parse a model-json and return a ModelContainer."""

    def __init__(self, net_parser=None):
        self.net_parser = net_parser

    def _parse_set_simple_parameters(self, data, model):
        """Parse the basic parameters of the model."""
        # Mandatory:
        model.outputs = data[JSON_DICT.Model.outputs]
        model.optimizer = data.get(JSON_DICT.Model.optimizer)
        model.loss = data.get(JSON_DICT.Model.loss)

        # Optional
        model.name = data.get(JSON_DICT.Model.name)
        model.comment = data.get(JSON_DICT.Model.comment)
        model.metrics = data.get(JSON_DICT.Model.metrics)
        model.additional_outputs = data.get(JSON_DICT.Model.additional_outputs)

    def _parse_network(self, data, model):
        nodes_data = data[JSON_DICT.Model.nodes]
        inputs_data = data[JSON_DICT.Model.inputs]

        return self.net_parser.parse(nodes_data, inputs_data)

    def parse(self, data):
        model = ParsedModel()

        self._parse_set_simple_parameters(data, model)
        if self.net_parser is None:
            return model
        else:
            net = self._parse_network(data, model)
            model.network = net
            model.network.model = model
            return model
