import os
import unittest
import tensorflow as tf
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.split(os.path.dirname(__file__))[0])

import test_utility  # noqa: E402
from HybridML.ModelCreator import KerasModelCreator  # noqa: E402
from HybridML.NodeRegistry import DefaultNodeRegistry  # noqa: E402


class test_neural_net_node(test_utility.TestCaseTimer):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.data = test_utility.load_relative_json(__file__)
        self.model_creator = KerasModelCreator(DefaultNodeRegistry)

    def build_model_by_id(self, _id):
        json = self.data[_id]
        model = self.model_creator.generate_models([json])[0]
        real_model = model.model

        return real_model

    def build_node_by_id(self, _id):
        model = self.build_model_by_id(_id)
        node = model.layers[-1]
        return node

    def test_layers(self):
        node = self.build_node_by_id("test_layers")
        layers = node.layers[1:]  # cut of input layer
        self.assertEqual(len(node.outputs), 1, "Should only have one output.")
        self.assertEqual(len(node.inputs), 1, "Should only have one input.")
        self.assertEqual(len(layers), 4, "Wrong number of layers in Neural Net creation. Expected 4 = 4 layers ")
        layer_sizes = [la.output.shape[-1] for la in layers]
        self.assertEqual(layer_sizes, [4, 3, 2, 1], "The layers should have the same size, as defined in json.")
        activations = ["Elu", "Relu", "Tanh", "Sigmoid"]
        self.assertTrue(
            all(act_name in la.output.name for la, act_name in zip(layers, activations)),
            "Activations of the layers should match.",
        )

    def test_regularizer_parsing(self):
        model = self.build_node_by_id("test_layers")

        reg_layer = model.layers[-1]
        kr = reg_layer.kernel_regularizer
        self.assertTrue(kr is not None)
        self.assertTrue(isinstance(kr, tf.python.keras.regularizers.L1L2))
        self.assertAlmostEqual(float(kr.l1), 0.1)
        self.assertAlmostEqual(float(kr.l2), 0.0)

        ar = reg_layer.activity_regularizer
        self.assertTrue(ar is not None)
        self.assertTrue(isinstance(ar, tf.python.keras.regularizers.L1L2))
        self.assertAlmostEqual(float(ar.l1), 0.0)
        self.assertAlmostEqual(float(ar.l2), 0.2)

    def test_multiple_inputs(self):
        node = self.build_node_by_id("multiple_inputs")
        layers = node.layers
        self.assertEqual(len(layers), 6)  # 4 inputs, 1 concatenate, 1 dense
        self.assertEqual(len(node.inputs), 4)
        self.assertEqual(len(node.outputs), 1)
        self.assertTrue("concat" in layers[-2].name)

    def test_multiple_outputs(self):
        model = self.build_model_by_id("multiple_outputs")
        node = model.layers[1]
        layers = node.layers
        self.assertEqual(len(layers), 2)  # 1 input, 1 dense
        self.assertEqual(len(node.inputs), 1)

        # Slicing of the outputs happens in the network, not in the node.
        self.assertEqual(len(model.outputs), 4)
        self.assertTrue(all(outp.shape[-1] == 1 for outp in model.outputs))


if __name__ == "__main__":
    t = test_neural_net_node()
    t.test_multiple_outputs()
    t.test_multiple_inputs()
    t.test_regularizer_parsing()

    unittest.main()
