from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from HybridML.building.BaseBuilders import NodeBuilder, ModelBuilder, NetBuilder
from HybridML.building.DataModel import NetworkContainer, BuiltNodeConnectorContainer
from HybridML.parsing.DataModel import ParsedNode
from .Containers import KerasModelContainer
import re
import tensorflow as tf


class KerasModelBuilder(ModelBuilder):
    """Creates a tensorflow/keras model out of the parsed model, emerging from parsing the json-file."""

    def __init__(self, keras_net_builder=None, custom_losses=None):
        self.keras_net_builder = keras_net_builder
        self.custom_losses = custom_losses or {}

    def build(self, parsed_model):
        net = parsed_model.network
        built_network = self.keras_net_builder.build_net(net)

        keras_model = self._build_keras_model(parsed_model, built_network)
        additional_outputs_model = self._build_additional_outputs_model(parsed_model, built_network)
        container = KerasModelContainer(keras_model, built_network, additional_outputs_model)

        return container

    def _compile_model(self, keras_model, parsed_model):
        if parsed_model.is_compilable():
            loss = self.custom_losses.get(parsed_model.loss) or parsed_model.loss
            keras_model.compile(loss=loss, metrics=parsed_model.metrics, optimizer=parsed_model.optimizer)

    def _build_keras_model(self, model, built_network, output_ids=None):
        inputs = [input.content for input in built_network.inputs]
        output_ids = output_ids if output_ids else model.outputs
        outputs = [built_network.data_points[id].content for id in output_ids]
        keras_model = self._create_keras_model(inputs=inputs, outputs=outputs, name=model.name)
        self._compile_model(keras_model, model)
        return keras_model

    def _create_keras_model(self, inputs, outputs, name=None):
        keras_model = Model(inputs=inputs, outputs=outputs, name=name)
        return keras_model

    def _build_additional_outputs_model(self, parsed_model, built_network):
        """The additional outputs model has as outputs the regular outputs plus the additional outputs, specified in the model-json"""
        if not parsed_model.additional_outputs:
            return None
        output_ids = [*(parsed_model.outputs), *(parsed_model.additional_outputs)]
        return self._build_keras_model(parsed_model, built_network, output_ids)


class KerasModelLoader(KerasModelBuilder):
    """Loads a saved Keras Model from file."""

    def __init__(self, custom_objects=None, *args, **kwargs):
        super(KerasModelLoader, self).__init__(*args, **kwargs)
        self.custom_objects = custom_objects

    def build(self, parsed_model, model_path):
        # load the additional outputs model and create the main model by selecting the standart outputs.
        # setting compile=False, so that the losses can be set afterwards
        custom_objects = self.custom_objects
        additional_outputs_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

        model_outputs = [additional_outputs_model.outputs[i] for i in range(len(parsed_model.outputs))]
        model_inputs = additional_outputs_model.inputs

        keras_model = self._create_keras_model(model_inputs, model_outputs, additional_outputs_model.name)
        self._compile_model(keras_model, parsed_model)

        container = KerasModelContainer(keras_model, None, additional_outputs_model)
        return container


class KerasNetBuilder(NetBuilder):
    """Recieves a parsed network and builds a tensorflow netork out of it."""

    def __init__(self, node_builder: NodeBuilder):
        super().__init__(node_builder)
        self.node_builder = node_builder

    def build_net(self, net):
        self.data_points = {}
        # build inputs
        for inp in net.inputs:
            name = inp.id
            name = re.sub("[\[\]\(\)\-| /]", ".", name)
            name = re.sub("²", "Â²", name)
            self.data_points[inp.id] = BuiltNodeConnectorContainer(inp.id, Input(shape=(inp.size,), name=name))

        # build and connect nodes
        built_nodes = []
        index = 0
        idle_loops = 0
        nodes_todo = net.nodes.copy()
        while len(nodes_todo) > 0:
            index %= len(nodes_todo)
            if idle_loops > len(nodes_todo):
                raise Exception("Problem with network, maybe cycle?")
            node: ParsedNode = nodes_todo[index]

            if not self.__are_inputs_for_current_node_known(node):
                index += 1
                idle_loops += 1
            else:
                nodes_todo.remove(node)
                idle_loops = 0

                built_node = self.__build_and_connect_node(node)
                built_nodes.append(built_node)

        inputs = [self.data_points[input.id] for input in net.inputs]
        return NetworkContainer(inputs=inputs, data_points=self.data_points, nodes=built_nodes)

    def __build_and_connect_node(self, node):
        input_tensors = [self.data_points[inp.id].content for inp in node.inputs]

        built_node = self.node_builder.build(node, input_tensors)

        self.__add_output_tensors_to_node_connector_dict(built_node.outputs, node.outputs)
        return built_node

    def __add_output_tensors_to_node_connector_dict(self, tensors, data_points):
        if len(data_points) == 1:
            tensors = [tensors]
        elif len(data_points) == tensors.shape[-1]:
            tensors = self.__spilt_tensor_into_data_points(tensors)
        for ten, dp in zip(tensors, data_points):
            self.data_points[dp.id] = BuiltNodeConnectorContainer(dp.id, ten)

    def __spilt_tensor_into_data_points(self, tensors):
        """Split up the tensor into individual tensors, corresponding to the number of data points."""
        out_tensors = []
        for point_idx in range(tensors.shape[-1]):
            out_tensors.append(tensors[:, point_idx : point_idx + 1])
        return out_tensors

    def __are_inputs_for_current_node_known(self, node):
        for inp in node.inputs:
            if inp.id not in self.data_points:
                return False
        return True
