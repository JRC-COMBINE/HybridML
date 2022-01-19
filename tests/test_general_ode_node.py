import os
import sys
import unittest

import numpy as np
import tensorflow as tf

import test_utility
from test_general_ode_layer import calc_const_ode

sys.path.append(os.path.split(os.path.dirname(__file__))[0])


def to_tf(var):
    return tf.constant(var, dtype=tf.float64)


class test_general_ode_node(test_utility.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.creator = test_utility.ModelFromTestJsonCreator(__file__)

    def test_basic_creation(self):
        self.creator.load_model_by_id("simple_node")

    def test_constant_rhs(self):
        x0 = np.array([[0]])
        t = np.array([[1, 2, 3]])
        expected = calc_const_ode(x0, t)
        expected_shape = expected.shape
        model = self.creator.load_model_by_id("const_ode")

        result = model.predict([x0, t])
        self.assertEqual(result.shape, expected_shape)
        self.assertClose(expected, result)

    def test_multiple_samples(self):
        x0 = np.array([[0], [0]])
        t = np.array([[1, 2, 3], [2, 3, 4]])
        expected = calc_const_ode(x0, t)
        expected_shape = expected.shape

        model = self.creator.load_model_by_id("const_ode")

        result = model.predict([x0, t])
        self.assertEqual(result.shape, expected_shape)
        self.assertClose(expected, result)

    def test_single_parameter(self):
        x0 = np.array([[0], [0], [0]])
        t = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        parameter = np.array([[1], [2], [3]])
        expected = calc_const_ode(x0, t, parameter[:, 0])
        expected_shape = expected.shape

        model = self.creator.load_model_by_id("single_parameter")

        result = model.predict([parameter, x0, t])
        self.assertEqual(result.shape, expected_shape)
        self.assertClose(expected, result)

    def test_multiple_parameters(self):
        x0 = np.array([[0], [0], [0]])
        t = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        a = np.array([[1], [2], [4]])
        b = np.array([[8], [16], [32]])
        expected = calc_const_ode(x0, t, 3 * a[:, 0] + b[:, 0])
        expected_shape = expected.shape

        model = self.creator.load_model_by_id("multiple_parameters")
        result = model.predict([a, b, x0, t])
        self.assertEqual(result.shape, expected_shape)
        self.assertClose(expected, result)


    def test_multiple_dimensions(self):
        x0 = np.array([[0,0,0]])
        t = np.array([[1, 2, 3]])
        a = np.array([[1]])
        b = np.array([[2]])
        c = np.array([[3]])
        expected = np.array([[[1, 2, 3], [2, 4, 6], [3, 6, 9]]])
        expected_shape = expected.shape

        model = self.creator.load_model_by_id("multiple_dimensions")
        result = model.predict([a, b, c, x0, t])
        self.assertEqual(result.shape, expected_shape)
        self.assertClose(expected, result)

    def test_training(self):
        random = np.random.RandomState(seed=42)
        n_samples = 100
        n_t = 20
        x0 = random.rand(n_samples, 1) * 100
        t = np.cumsum(random.rand(n_samples, n_t), axis=1)
        parameter = random.rand(n_samples) * 10
        x = [parameter, x0, t]
        y = calc_const_ode(x0, t, parameter)

        model = self.creator.load_model_by_id("test_training")
        loss_before = model.evaluate(x=x, y=y)
        model.fit(x, y, epochs=10)
        loss_after = model.evaluate(x=x, y=y)
        self.assertTrue(loss_after < loss_before)


if __name__ == "__main__":
    t = test_general_ode_node()
    t.test_multiple_dimensions()
    t.test_multiple_parameters()
    t.test_multiple_samples()
    t.test_constant_rhs()
    t.test_basic_creation()
    t.test_single_parameter()
    t.test_training()
    unittest.main()
