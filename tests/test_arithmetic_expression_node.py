import os
import sys
import unittest

import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.split(os.path.dirname(__file__))[0])
from HybridML.building.nodes.ArithmeticExpression import ArithmeticExpressionNodeBuilder  # noqa: E402
from HybridML.ModelCreator import KerasModelCreator  # noqa: E402
from HybridML.NodeRegistry import DefaultNodeRegistry  # noqa: E402
from HybridML.parsing.nodes.ArithmeticExpression import ArithmeticExpressionNodeParser  # noqa: E402

import test_utility  # noqa: E402


class arithmetic_expression_node_registry:
    node_parsers = [ArithmeticExpressionNodeParser()]
    node_builders = [ArithmeticExpressionNodeBuilder()]
    custom_losses = {}


class arithmetic_tester:
    def __init__(self, file, node_registry):
        self.file = file
        self.model_creator = KerasModelCreator(node_registry)
        self.path = os.path.join(os.path.dirname(__file__), self.file)

    def fill_expressions(self, x1_val, x2_val, x1_s="x1", x2_s="x2"):
        x1_val = float(x1_val)
        x2_val = float(x2_val)
        operators = {
            "+": lambda x, y: x + y,
            "-": lambda x, y: x - y,
            "*": lambda x, y: x * y,
            "/": lambda x, y: x / y,
            "**": lambda x, y: x ** y,
            "%": lambda x, y: x % y,
        }
        vals = []

        for op, fct in operators.items():
            vals.append(
                {"expression": "out1 = " + x1_s + op + x2_s, "x1": x1_val, "x2": x2_val, "result": fct(x1_val, x2_val)}
            )
            vals.append(
                {"expression": "out1 = " + x2_s + op + x1_s, "x1": x1_val, "x2": x2_val, "result": fct(x2_val, x1_val)}
            )
        return vals

    def create_model_w_expression(self, expression, model_name="standard_test"):
        model_json = test_utility.read_json_and_replace(self.path, "#expression#", expression)
        model_json = model_json[model_name]
        return self.model_creator.generate_models([model_json])[0]

    def predict_model(self, model, x1, x2):
        return model.predict([np.array([[x1]]), np.array([[x2]])])

    def create_predict_model(self, expression, x1, x2, model_name="standard_test"):
        model = self.create_model_w_expression(expression, "standard_test")
        return self.predict_model(model, x1, x2)

    def create_model(self, model_name, to_replace={}):
        model_json = test_utility.read_json_replace_dict(self.path, to_replace)
        model_json = model_json[model_name]
        return self.model_creator.generate_models([model_json])[0]


class test_arithmetric_expression_node(test_utility.TestCaseTimer):
    """Integration Test for ArithmeticExpressionLayer and ArithmeticExpressionNode inside HybridML's structure"""

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.tester = arithmetic_tester(os.path.split(__file__)[1] + ".json", arithmetic_expression_node_registry)

    def test_simple_building(self):
        res = self.tester.create_predict_model("out1 = x1+x2", 1, 1)[0]
        test_utility.assertClose(self, res, 2)

    def test_binary_expressions(self):
        vals = []
        vals.extend(self.tester.fill_expressions(1, 10))
        vals.extend(self.tester.fill_expressions(-10, 1))
        for val in tqdm(vals):
            prediction = self.tester.create_predict_model(val["expression"], val["x1"], val["x2"])
            test_utility.assertClose(self, prediction, val["result"])

    def test_w_const(self):
        vals = self.tester.fill_expressions(10, 2, "x1", "2")
        for val in tqdm(vals):
            prediction = self.tester.create_predict_model(val["expression"], val["x1"], val["x2"])
            test_utility.assertClose(self, prediction[0], val["result"])

    def test_two_constants(self):
        self.assertRaises(Exception, lambda: self.tester.create_predict_model("1+2", 0, 0))
        self.assertRaises(Exception, lambda: self.tester.create_predict_model("1*2", 0, 0))
        self.assertRaises(Exception, lambda: self.tester.create_predict_model("1^2", 0, 0))

    def test_unknown_variables(self):
        self.assertRaises(Exception, lambda: self.tester.create_predict_model("x1+y2", 0, 0))

    def test_wrong_syntax(self):
        self.assertRaises(Exception, lambda: self.tester.create_predict_model("++", 0, 0))
        self.assertRaises(Exception, lambda: self.tester.create_predict_model("x1+*x2", 0, 0))
        self.assertRaises(Exception, lambda: self.tester.create_predict_model("2+-*/x1", 0, 0))
        self.assertRaises(Exception, lambda: self.tester.create_predict_model("2^^^^^", 0, 0))

    def test_matrix_input(self):
        random = np.random.RandomState(seed=42)
        for _ in tqdm(range(10)):
            x1 = random.rand() * 10 - 5
            x2 = random.rand() * 10 - 5
            res = np.array([[x1, 2 * x2], [x1 * x2, -x2]])
            prediction = self.tester.create_predict_model("out1 = [[x1, 2 * x2],[x1 * x2, -x2]]", x1, x2)[0]
            self.assertTrue(np.all(np.abs(prediction - res) < 1e-5))

    def test_slicing(self):
        random = np.random.RandomState(seed=42)
        for _ in tqdm(range(10)):
            x1 = random.randn(2)
            x2 = 0
            replace_dict = {"#expression#": "out1 = x1[:, 0]", "#size1#": 2, "#size2#": 1}
            model = self.tester.create_model(model_name="other_dimensions", to_replace=replace_dict)
            result = model.predict([np.array([x1]), np.array([x2])])

            assert np.all(result == x1[0])

    def test_non_ascending_variable_ordering(self):
        a, b, c = (1, 2, 4)
        model = self.tester.create_model(model_name="non_asceding_variable_order")
        result = model.predict([np.array([a]), np.array([b]), np.array([c])])
        assert np.all(result == np.array([[4, 2, 1]]))

    def test_saving_and_loading(self):
        vals = self.tester.fill_expressions(1, 5)
        for val in tqdm(vals):
            model1 = self.tester.create_model_w_expression(val["expression"], "standard_test")
            file = os.path.join(os.path.dirname(__file__), "test_arithmetic_save.h5")
            model1.save_to_file(file)
            model2 = tf.keras.models.load_model(file, custom_objects=DefaultNodeRegistry.custom_objects)
            prediction1 = self.tester.predict_model(model1, val["x1"], val["x2"])
            prediction2 = self.tester.predict_model(model2, val["x1"], val["x2"])
            test_utility.assertClose(self, prediction1, prediction2)
            os.remove(file)


if __name__ == "__main__":
    t = test_arithmetric_expression_node()
    t.test_non_ascending_variable_ordering()
    unittest.main()
