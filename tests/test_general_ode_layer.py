import os
import sys
import unittest

import numpy as np
import tensorflow as tf

import test_utility

sys.path.append(os.path.split(os.path.dirname(__file__))[0])

from HybridML.keras.layers.GeneralOde import GeneralOdeLayer  # noqa: E402
from HybridML.keras.layers.LinearOde import LinearOdeLayer  # noqa: E402


def calc_const_ode(x0, t, const_value=None):
    """
    x0: samples * dims
    t: samples *  n_t
    const_value = samples
    """
    # Default a = 1
    if const_value is None:
        const_value = np.ones([x0.shape[0]])

    x0_tiled = np.tile(x0[:, :, None], (1, 1, t.shape[-1]))
    x0_swapped = np.swapaxes(x0_tiled, 1, 2)
    t_expanded = t[:, :, np.newaxis]
    const_val_expanded = const_value[:, None, None]
    result = x0_swapped + t_expanded * const_val_expanded
    return result


class test_general_ode_layer(test_utility.TestCase):
    def test_creation(self):
        layer = GeneralOdeLayer("")
        self.assertIsNotNone(layer)

    def build_and_test(self, d):
        """Build a general ode layer, based on the dict d and check if it has the expected output."""

        layer = GeneralOdeLayer(d["expression"])
        if isinstance(d["ps"], list):
            result = layer.call([*d["ps"], d["x0"], d["t"]])
        else:
            result = layer.call([d["ps"], d["x0"], d["t"]])
        self.assertEqual(result.shape, d["expected_shape"])

        expected = d["expected"]
        actual = result.numpy()
        # plot(expected, actual)
        self.assertClose(expected, actual)

    def test_constant_rhs(self):
        d = {
            "expression": "1",
            "x0": np.array([[0]]),
            "t": np.array([[1, 2, 3]]),
            "ps": np.array([]),
        }
        d["expected"] = calc_const_ode(d["x0"], d["t"])
        d["expected_shape"] = d["expected"].shape
        self.build_and_test(d)

    def test_complex_ode(self):
        n_t = 50
        a = 0.2
        b = 0.3
        expression = "a + (x-(a*t +b))**5"
        expression = expression.replace("a", str(a))
        expression = expression.replace("b", str(b))

        t_points = np.linspace(0.0, 8.0, n_t)
        sol = np.array(a * t_points + b)
        x0 = sol[0]

        n_dim = 1
        n_samples = 1
        expected_shape = [n_samples, n_t, n_dim]
        d = {
            "expression": expression,
            "x0": np.array([x0]),
            "t": np.array([t_points]),
            "expected": np.array(sol).reshape(expected_shape),
            "expected_shape": expected_shape,
            "ps": np.array([]),
        }
        self.build_and_test(d)

    def test_compare_to_lin_ode_layer(self):
        expression = "[-ka * x[0], ka * x[0] - (ke + k12) * x[1]+ k21*x[2], k12 * x[1] -k21*x[2]]"
        ka = 6.171449e-04
        ke = 3.534819e-01
        k12 = 5.702387e04
        k21 = 1.573615e03
        expression = expression.replace("ka", str(ka))
        expression = expression.replace("ke", str(ke))
        expression = expression.replace("k12", str(k12))
        expression = expression.replace("k21", str(k21))

        linode = LinearOdeLayer()

        sys_mat = [[[-ka, 0, 0], [ka, -ke - k12, k21], [0, k12, -k21]]]
        x0 = np.array([[1, 1, 2]])
        t = np.array([[1, 2, 3]])
        expected = linode.call([to_tf(sys_mat), to_tf(x0), to_tf(t)])

        d = {
            "expression": expression,
            "x0": x0,
            "t": t,
            "expected": expected,
            "expected_shape": [1, 3, 3],
            "ps": np.array([]),
        }
        self.build_and_test(d)

    def test_parameters(self):
        d = {
            "expression": "a",
            "x0": np.array([[0], [0], [0]]),
            "t": np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
            "ps": np.array([1, 2, 3]),
        }
        d["expected"] = calc_const_ode(d["x0"], d["t"], d["ps"])
        d["expected_shape"] = d["expected"].shape
        self.build_and_test(d)

    def test_parameter_gradients(self):
        d = {
            "expression": "a",
            "x0": np.array([[0], [0], [0]]),
            "t": np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
            "ps": np.array([1, 2, 3]),
            "grad": np.array([[6], [6], [6]]),
        }
        d["expected"] = calc_const_ode(d["x0"], d["t"], d["ps"])
        d["expected_shape"] = d["expected"].shape

        layer = GeneralOdeLayer(d["expression"])

        # Obtain Gradients
        ps = tf.constant(d["ps"], tf.float64)
        x0 = tf.constant(d["x0"], tf.float64)
        t = tf.constant(d["t"], tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ps)
            result = layer.call([ps, x0, t])

        gradient = tape.gradient(result, ps)
        self.assertClose(d["grad"], gradient)

        self.assertEqual(result.shape, d["expected_shape"])
        expected = d["expected"]
        actual = result.numpy()
        self.assertClose(expected, actual)

    def test_multiple_samples(self):
        d = {
            "expression": "1",
            "x0": np.array([[0], [0], [0]]),
            "t": np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]]),
            "ps": np.array([]),
        }
        d["expected"] = calc_const_ode(d["x0"], d["t"])
        d["expected_shape"] = d["expected"].shape
        self.build_and_test(d)

    def test_multiple_parameters(self):
        d = {
            "expression": "a+b",
            "x0": np.array([[0], [0], [0]]),
            "t": np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
            "ps": [np.array([1, 3, 5]), np.array([2, 4, 6])],
        }
        d["expected"] = calc_const_ode(d["x0"], d["t"], d["ps"][0] + d["ps"][1])
        d["expected_shape"] = d["expected"].shape
        self.build_and_test(d)


def to_tf(var):
    return tf.constant(var, dtype=tf.float64)


if __name__ == "__main__":
    t = test_general_ode_layer()
    t.test_multiple_parameters()
    t.test_multiple_samples()
    t.test_parameter_gradients()
    t.test_parameters()
    t.test_complex_ode()
    t.test_creation()
    t.test_compare_to_lin_ode_layer()
    t.test_constant_rhs()
    unittest.main()
