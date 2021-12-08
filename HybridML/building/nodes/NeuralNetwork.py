import tensorflow as tf
from HybridML.parsing.nodes.NeuralNetwork import Layer, NNNode
from tensorflow.keras.layers import Concatenate, Dense

from ..BaseBuilders import NodeBuilder
from ..DataModel import BuiltNodeContainer


class NNNodeBuilder(NodeBuilder):
    def __init__(self):
        super().__init__("nn")

    def create_single_layer(self, layer):
        assert isinstance(layer, Layer)
        assert layer.type == "dense"
        if layer.activation is None or layer.activation.lower() == "none":
            activation = None
        else:
            activation = layer.activation

        activity_regularizer = self.parse_regularizer(layer.activity_regularizer)
        kernel_regularizer = self.parse_regularizer(layer.kernel_regularizer)

        return Dense(
            layer.size,
            activation=activation,
            activity_regularizer=activity_regularizer,
            kernel_regularizer=kernel_regularizer,
        )

    def build(self, node: NNNode, inputs) -> BuiltNodeContainer:
        # inp.shape[1:]: cut off the sample dimension for the input shapes
        model_inputs = [tf.keras.layers.Input(inp.shape[1:]) for inp in inputs]

        if len(model_inputs) == 1:
            x = model_inputs[0]
        else:
            x = Concatenate()(model_inputs)

        for layer in node.layers:
            tf_layer = self.create_single_layer(layer)
            x = tf_layer(x)
        model = tf.keras.models.Model(inputs=model_inputs, outputs=[x], name=f"BlackBox_{node.id}")

        if len(inputs) == 1:
            inputs = inputs[0]
        outputs = model(inputs)
        return BuiltNodeContainer(node, outputs, model)

    def parse_regularizer(self, regularizer_str):
        """Parse a regularizer string into a keras regularizer.

        "L1(0.1)" and "L2(0.3)" are examples for regularizer strings."""
        if regularizer_str is None:
            return None
        reg = regularizer_str.strip()
        if not reg:
            return None
        reg_parameter = reg[2:].strip()[1:-1].strip()
        reg_par = float(reg_parameter)

        reg_type = reg[:2].lower()
        if reg_type == "l1":
            return tf.keras.regularizers.L1L2(l1=reg_par)
        elif reg_type == "l2":
            return tf.keras.regularizers.L1L2(l2=reg_par)
        else:
            raise Exception(f"Unknown regularizer: {reg_type}")
