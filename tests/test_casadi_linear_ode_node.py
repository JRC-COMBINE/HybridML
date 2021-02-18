import unittest
import os
import tensorflow as tf
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.split(os.path.dirname(__file__))[0])
from test_linear_ode_node import Wrapper, linear_node, node_type, sys_mat  # noqa: E402
import test_utility  # noqa: E402

# tf.keras.backend.set_floatx("float64")
casadi_node = "casadi_linear_ode"


class test_casadi_linear_ode_node(Wrapper.LinearOdeNodeTesterBase):
    def __init__(self, methodName="runTest", iterations_per_problem=5):
        super().__init__(
            current_file=__file__,
            methodName=methodName,
            iterations_per_problem=iterations_per_problem,
            close_threshold=5e-2,
        )
        self.ode_node_type = casadi_node

    def test_compare_to_closed_form(self):
        for _ in self.repeat("test_compare_to_closed_form"):
            # set up models
            prob = self.generator.linear_problem_with_parameters()
            model1 = self.load_model_replace_dict("with_parameters", {sys_mat: prob.A_str, node_type: linear_node})
            model2 = self.load_model_replace_dict("with_parameters", {sys_mat: prob.A_str, node_type: casadi_node})

            param_values = [[[val]] for val in prob.param_values[0]]

            # predict results
            inputs = [
                tf.constant(val, dtype=tf.keras.backend.floatx()) for val in (param_values + [prob.x_init, prob.t])
            ]
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(inputs)
                res1 = model1.model(inputs)
                res2 = model2.model(inputs)

            # get gradients
            grads1 = tape.gradient(res1, inputs[:4])
            grads2 = tape.gradient(res2, inputs[:4])

            # compare results
            test_utility.assertClose(self, res1, res2, self.close_threshold)

            # compare gradients
            for grad1, grad2 in zip(grads1, grads2):
                test_utility.assertClose(self, grad1, grad2, self.close_threshold)


if __name__ == "__main__":
    t = test_casadi_linear_ode_node()
    t.test_compare_to_closed_form()
    unittest.main()
