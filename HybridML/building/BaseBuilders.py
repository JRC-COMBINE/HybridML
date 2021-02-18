import collections
import numpy as np

from HybridML.building.DataModel import ModelContainer, NetworkContainer, BuiltNodeContainer


class NodeBuilder:
    """Abstract base class for all NodeBuilders"""

    def __init__(self, builds_types):
        if isinstance(builds_types, (collections.Sequence, np.ndarray)) and not isinstance(builds_types, str):
            self.builds_types = builds_types
        else:
            self.builds_types = [builds_types]

    def build(self, node, inputs) -> BuiltNodeContainer:
        raise NotImplementedError()


class NodeBuilderSelector(NodeBuilder):
    """Has a list of node builders. Selects the right builder, that can build a node out of the incoming data, when build is called."""

    def __init__(self):
        super().__init__(None)
        self._node_builders = {}

    def append_builder(self, builder: NodeBuilder):
        for n_type in builder.builds_types:
            self._node_builders[n_type] = builder

    def build(self, node, inputs) -> BuiltNodeContainer:
        if node.type not in self._node_builders:
            raise Exception("Node type unknown: " + node.type)
        return self._node_builders[node.type].build(node, inputs)


class ModelBuilder:
    def build(self, parsed_input) -> ModelContainer:
        raise NotImplementedError()


class NetBuilder:
    def __init__(self, node_builder: NodeBuilder):
        self.node_builder = node_builder

    def build_net(self, net) -> NetworkContainer:
        raise NotImplementedError()
