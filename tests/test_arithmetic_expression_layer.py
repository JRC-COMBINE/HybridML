import os
import sys
import unittest
from collections import namedtuple

import numpy as np
import tensorflow as tf

import test_utility

sys.path.append(os.path.split(os.path.dirname(__file__))[0])
sys.path.append(os.path.split(os.path.dirname(__file__))[0])

from HybridML.keras.layers.ArithmeticExpression import (  # noqa: E402
    ArithmeticExpressionLayer, find_variables_in_assignment,
    find_vars_in_expression)

TestItem = namedtuple("TestItem", ["expression", "input", "target"])
GradTestItem = namedtuple("GradTestItem", ["expression", "input", "target", "gradient"])


def np_arr(*x):
    return np.array([*x])


def tf_arr(*x):
    return tf.constant([*x], dtype=tf.keras.backend.floatx())


class test_arithmetic_expression_layer(test_utility.TestCase):
    def _create_layer(self, expression, *args, **kwargs):
        layer = ArithmeticExpressionLayer(expression, *args, **kwargs)
        return layer

    def assertNpEqual(self, a, b, *args, **kwargs):
        self.assertTrue(np.all(a == b), *args, **kwargs)

    def test_find_variables_in_expression(self):
        expressions = [("x1", ["x1"]), ("x1+x2", ["x1", "x2"]), ("[a,b]*c", ["a", "b", "c"])]
        for expression, expected in expressions:
            result = find_vars_in_expression(expression)
            self.assertEqual(expected, result)

    def test_find_variables_in_assignment(self):
        expressions = ["abc", "=abc", "abc="]
        for expression in expressions:
            fun = lambda: find_variables_in_assignment(expression)  # noqa E731
            msg = "An Assignment needs to have an equals sign and expressions on both sides."
            self.assertRaises(Exception, fun, msg)

        expressions = [("a=b", (["b"], ["a"])), ("x1,x2 = a*n+x", (["a", "n", "x"], ["x1", "x2"]))]
        for expression, target in expressions:
            res = find_variables_in_assignment(expression)
            self.assertEqual(res, target)

    def test_basic_creation(self):
        name = "arith_layer"
        layer = self._create_layer("a", name=name)
        self.assertIsNotNone(layer)
        self.assertEqual(layer.name, name)
        self.assertEqual(layer.stack_fn(10), 10, "Stack function defaults to identity.")
        self.assertEqual(layer.adapt_depth_fn(10, 20), 10, "Adapt depth function defaults to select 0.")

    def test_saves_expression(self):
        expression = "a + 3 * b"
        layer = self._create_layer(expression)
        self.assertEqual(layer.expression, expression)

    def test_config_contains_expression(self):
        expression = "a3 + b4 * c"
        layer = self._create_layer(expression)
        config = layer.get_config()
        self.assertEqual(config["expression"], expression)

        self.assertTrue(
            {"name", "trainable", "dtype"} <= config.keys(), "Base configuration contained in configuration."
        )

    def test_build_faulty_expressions(self):
        def assert_raises(expression, msg):
            self.assertRaises(Exception, lambda: self._create_layer(expression).build, msg)

        for expression in ["", " ", "\t"]:
            assert_raises(expression, "Raise when expression is emtpy.")

        assert_raises("a = b", "Exception, when assignment is input.")

        assert_raises("[1,2]", "Raise when expression contains no variables.")

    def test_expression_depth(self):
        expressions = [("a", 0), ("[a,b]", 1), ("[[a,b],[c,d]]", 2)]
        for expression, depth in expressions:
            layer = self._create_layer(expression)
            layer.build(None)
            self.assertEqual(layer.depth, depth, "The depth of the expression should be calculated correctly.")

        expression = "[[a,b], c]"
        layer = self._create_layer(expression)
        self.assertRaises(Exception, lambda: layer.build(None), "Raise: No clear depth of the expression.")

    def test_set_variables(self):
        expressions = [("x1", ["x1"]), ("x1+x2", ["x1", "x2"]), ("[a,b]*c", ["a", "b", "c"])]
        for expression, expected in expressions:
            layer = self._create_layer(expression)
            layer.build(None)
            self.assertEqual(expected, layer.input_var_names)

    def test_adapt_depth_fun(self):
        test_items = [
            TestItem(expression="a", input=1, target=np_arr(1, 1)),
            TestItem(expression="[a,b]", input=[1, 2], target=[np_arr(1, 1), np_arr(2, 2)]),
            TestItem(
                expression="[[a,b],[c,d]]",
                input=[[3, 4], [0, 1]],
                target=[[np_arr(3, 3), np_arr(4, 4)], [np_arr(0, 0), np_arr(1, 1)]],
            ),
            TestItem(expression="[a,0]", input=[np_arr(1, 1), 0], target=[np_arr(1, 1), np_arr(0, 0)]),
            TestItem(
                expression="[[a,0],[1,b]]",
                input=[[np_arr(2, 3), 0], [1, np_arr(3, 4)]],
                target=[[np_arr(2, 3), np_arr(0, 0)], [np_arr(1, 1), np_arr(3, 4)]],
            ),
        ]
        zero_tensor = np.array([0, 0])
        for expression, tensor_to_adapt, target in test_items:
            layer = self._create_layer(expression)
            layer.build(None)
            adapted_tensor = layer.adapt_depth_fn(tensor_to_adapt, zero_tensor)

            if isinstance(target, list):
                self.assertIsInstance(adapted_tensor, list)
                if isinstance(target[0], list):
                    self.assertIsInstance(adapted_tensor[0], list)
            np_target = np.array(target)
            np_adapted = np.array(adapted_tensor)
            msg = "Adapting the depth of expressions should work with up to 2d expressions."
            self.assertNpEqual(np_target, np_adapted, msg)

    def test_stack_function(self):
        test_items = [
            TestItem(
                expression="[a,b]",
                input=[np_arr([1], [2], [3]), np_arr([3], [4], [5])],
                target=np_arr([[1, 3], [2, 4], [3, 5]]),
            ),
            TestItem(expression="a", input=np_arr([1]), target=np_arr([1])),
            TestItem(expression="[a,0]", input=[np_arr([1]), np_arr([0])], target=np_arr([[1, 0]])),
            TestItem(
                expression="[[-ka,0],[ka,-ke]]",
                input=[[np_arr([1]), np_arr([0])], [np_arr([-1]), np_arr([3])]],
                target=np_arr([[[1, 0], [-1, 3]]]),
            ),
        ]
        for expression, tensor_to_adapt, target in test_items:
            layer = self._create_layer(expression)
            layer.build(None)
            result = layer.stack_fn(tensor_to_adapt)

            target = tf.stack(target)

            self.assertNpEqual(result.numpy(), target.numpy())

    test_items_for_calling = [
        TestItem(expression="a", input=[tf_arr([1])], target=tf_arr([1])),
        TestItem(expression="a+1", input=[tf_arr([1, 2])], target=tf_arr([2, 3])),
        TestItem(expression="a * 1.5e2", input=[tf_arr([1], [2], [3])], target=tf_arr([1], [2], [3]) * 150),
        TestItem(expression="[a, b+1]", input=[tf_arr([3]), tf_arr([12])], target=tf_arr([[3, 13]])),
        TestItem(expression="[dose, 0, 0]", input=[tf_arr([24])], target=tf_arr([[24, 0, 0]])),
        TestItem(
            expression="[[-ka, 0,], [ka, ke]]", input=[tf_arr([12]), tf_arr([-3])], target=tf_arr([[-12, 0], [12, -3]])
        ),
        TestItem(
            expression="[a, b - 1]",
            input=[tf_arr([1], [2], [3]), tf_arr([4], [6], [8])],
            target=tf_arr([[1, 3], [2, 5], [3, 7]]),
        ),
    ]

    def test_call(self):
        for expression, tensors_to_evaluate, target in test_arithmetic_expression_layer.test_items_for_calling:
            layer = self._create_layer(expression)
            layer.build(None)
            result = layer(tensors_to_evaluate)
            self.assertNpEqual(result.numpy(), target.numpy())

    def test_matlab_style_input(self):
        expressions = [
            TestItem(expression="dose,0,0", input=[tf_arr([12])], target=tf_arr([[12, 0, 0]])),
            TestItem(expression="-ka,0;ka,ke", input=[tf_arr([12]), tf_arr([-3])], target=tf_arr([[-12, 0], [12, -3]])),
        ]
        for expression, tensors, target in expressions:
            layer = self._create_layer(expression)
            layer.build(None)
            result = layer(tensors)
            self.assertNpEqual(result.numpy(), target.numpy())

    def test_gradient(self):
        test_items = [
            GradTestItem(expression="a", input=[tf_arr([1])], target=tf_arr([1]), gradient=[tf_arr([1])]),
            GradTestItem(expression="a+1", input=[tf_arr([1, 2])], target=tf_arr([2, 3]), gradient=[tf_arr([1, 1])]),
            GradTestItem(
                expression="[a, b+1]",
                input=[tf_arr([3]), tf_arr([12])],
                target=tf_arr([[3, 13]]),
                gradient=[tf_arr([[1, 1], [1, 1]])],
            ),
            GradTestItem(
                expression="[dose, 0, 0]", input=[tf_arr([24])], target=tf_arr([[24, 0, 0]]), gradient=[tf_arr([[1]])]
            ),
            GradTestItem(expression="2*a+3", input=[tf_arr([3])], target=tf_arr([[9]]), gradient=[tf_arr([[2]])]),
        ]

        for expression, tensors, target, target_grad in test_items:
            layer = self._create_layer(expression)
            layer.build(None)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(tensors)
                res = layer(tensors)
                result = tf.reshape(res, [-1])
            pred_grad = tape.gradient(result, tensors)
            self.assertNpEqual(result.numpy(), target.numpy())
            for pred, target in zip(pred_grad, target_grad):
                self.assertNpEqual(pred.numpy(), target.numpy())


class test_build_arithmetic_expression_layer(test_arithmetic_expression_layer):
    def _create_layer_without_prefix(self, expression, *args, **kwargs):
        layer = ArithmeticExpressionLayer(expression, *args, **kwargs)
        return layer

    def _create_layer(self, expression, *args, **kwargs):
        expression = "output=" + expression
        layer = self._create_layer_without_prefix(expression, *args, **kwargs)
        return layer

    def test_vars_disjunct(self):
        expressions = ["a=a", "a=a*b", "a=[c,d,a]"]
        for expression in expressions:
            self.assertRaises(
                Exception,
                lambda: self._create_layer_without_prefix(expression),
                "Error, when trying to assign a variable to an expression containing itself.",
            )

    def test_assign_only_one_variable(self):
        expression = "a,b = 1,2"
        self.assertRaises(
            Exception, lambda: self._create_layer_without_prefix(expression), "Raise if trying to assign two variables."
        )


if __name__ == "__main__":
    unittest.main()
