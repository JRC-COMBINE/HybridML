import unittest
import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.layers import Input
import tqdm
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.split(os.path.dirname(__file__))[0])
import test_utility  # noqa: E402


from HybridML.ModelCreator import KerasModelCreator  # noqa: E402
from HybridML.NodeRegistry import DefaultNodeRegistry  # noqa: E402

from HybridML.parsing.nodes.LinearOde import LinearOdeNodeParser  # noqa: E402
from HybridML.building.nodes.LinearOde import LinearOdeNodeBuilder  # noqa: E402


# tf.keras.backend.set_floatx("float64")
node_type = "#node_type#"
linear_node = "linear_ode"
sys_mat = "#system_matrix#"


class LinProblem:
    """ Dummy class for generated linear problems. Just contains variables."""

    def __init__(self, **kwargs):
        self.set(**kwargs)

    def set(self, **kwargs):
        """Set keywordarguments as instance variables and return itself, in order to have an elegant return statement"""
        for name, value in kwargs.items():
            setattr(self, name, value)
        return self


class LinearODEProblemGeneratorMaths:
    """Contains mathematical part of problem generation"""

    def sol_fn(self, A, t, x_init):
        op1 = self.grad_fn(A, t)
        return np.array([mat @ vec for mat, vec in zip(op1, x_init)])

    def grad_fn(self, A, t):
        # In case of constant A repeat A n times
        if len(A.shape) < 3:
            n_samples = t.shape[0]
            A = np.tile(A[np.newaxis, :, :], (n_samples, 1, 1))

        # Bring A and t in the right shape for broadcasting
        A_ = A[:, None, :, :]
        t_ = t[:, :, None, None]
        mult = t_ * A_
        return tf.linalg.expm(mult).numpy()


class LinearODEProblemGenerator:
    """Generates linear ODE problems to test ode solvers"""

    def __init__(self, seed=None, random_state=None):
        self.random = random_state if random_state is not None else np.random.RandomState(seed)
        self.maths = LinearODEProblemGeneratorMaths()

    def generate_basic_parameters(self, dim=2, n=1):
        """Generates parameters, needed for all supported linear problems"""
        A = self.random.rand(dim, dim) * 2 - 1
        x_init = self.random.rand(n, dim) * 2 - 1
        t = np.sort(self.random.rand(n, 5) * 5, axis=1)
        A_str = self.mat_to_str(A)
        return LinProblem(A=A, A_str=A_str, x_init=x_init, t=t)

    def linear_problem(self, dim=2, n=11):
        """ Generates a linear problem with a constant system matrix"""
        prob = self.generate_basic_parameters(dim=dim, n=n)
        A, t, x_init = prob.A, prob.t, prob.x_init

        x_sol = self.maths.sol_fn(A, t, x_init)
        grad_sol = self.maths.grad_fn(A, t).sum(axis=(1, 2))

        return prob.set(x_sol=x_sol, grad_sol=grad_sol)

    def linear_problem_with_parameters(self, param_names=None, dim=2, n=1):

        prob = self.generate_basic_parameters(dim=dim, n=n)
        A, x_init, t = prob.A, prob.x_init, prob.t

        # Generate parameters for System Matrix A
        param_names = ["p" + str(i) for i in range(dim * dim)] if param_names is None else param_names
        param_values = np.random.rand(n, dim * dim) * 2 - 1

        # Multiply A with the parameters
        A_w_params = param_values.reshape([n, dim, dim]) * A

        # Calculate solutions
        x_sol = self.maths.sol_fn(A_w_params, t, x_init)
        grad_sol = self.maths.grad_fn(A_w_params, t).sum(axis=(1, 2))

        A_str = self.mat_w_params_to_str(A, param_names)
        return prob.set(A_str=A_str, x_sol=x_sol, grad_sol=grad_sol, param_names=param_names, param_values=param_values)

    def linear_samples_for_training(self, dim=2, n=11):
        """Generates a simple linear problem for training.
        A is used to calculate a system matrix A(p) for each sample, wrt. a generated parameter p.

        Returns:
            prob (LinProblem): prob.x_init, prob.p, prob.t, prob.x_sol"""

        prob = self.generate_basic_parameters(dim=dim, n=n)
        base_A, x_init, t = prob.A, prob.x_init, prob.t

        # Generate input parameter (covariate) for each sample and calculate a system matrix based on the parameter
        parameter = self.random.rand(n, 1) * 2 - 1
        A = parameter[:, np.newaxis] * base_A[np.newaxis]  # should be shape (n, dim, dim)

        x_sol = self.maths.sol_fn(A, t, x_init)

        prob.A = None  # Initial value of A does not matter, could be interpreted wrongly
        return prob.set(p=parameter, x_sol=x_sol)

    def mat_w_params_to_str(self, A, param_names):
        """Gets a matrix in matlab-style and a parameter for each entry.
        Returns matrix string with elementwise multiplication of parameters and A"""
        A_ = ""
        i = 0
        for line in A:
            for elm in line:
                A_ += param_names[i] + "*" + str(elm)
                i += 1
                A_ += ","
            A_ = A_[:-1]
            A_ += ";"
        A_ = A_[:-1]
        return A_

    def mat_to_str(self, A):
        """Takes a matrix of numbers and combines them into a matlab style matrix sstring"""
        return ";".join(",".join(str(elm) for elm in line) for line in A)


# Wrap TesterBase into an empty class, s.t. it is not seen as a TestCase itself
class Wrapper:
    class LinearOdeNodeTesterBase(test_utility.TestCaseTimer):
        def __init__(self, current_file, methodName="runTest", iterations_per_problem=2, close_threshold=1e-4):
            super().__init__(methodName)
            self.close_threshold = close_threshold
            self.generator = LinearODEProblemGenerator(seed=42)
            self.iterations_per_problem = iterations_per_problem
            self.data = test_utility.load_relative_json(current_file)
            self.creator = KerasModelCreator(DefaultNodeRegistry())
            self.node_parser = LinearOdeNodeParser()
            self.node_builder = LinearOdeNodeBuilder()

        def repeat(self, desc=None, n=None):
            n = self.iterations_per_problem if n is None else n
            return tqdm.tqdm(range(n), desc)

        def load_model_by_id(self, json_id):
            data = self.data[json_id]
            model = self.creator.generate_models([data])[0]
            return model

        def load_model_replace_dict(self, model_id, replace_dict):
            s = json.dumps(self.data[model_id])
            for to_replace, replace_with in replace_dict.items():
                s = s.replace(to_replace, replace_with)
            data = json.loads(s)

            result = self.creator.generate_models([data])[0]
            return result

        def load_model_replace(self, model_id, to_replace, replace_with):
            return self.load_model_replace_dict(model_id, {to_replace: replace_with})

        def build_layer(self, A, dim, n_t, p_num=0):
            data = self.data["node_only"].copy()
            data["system_matrix"] = A

            p_input = Input(shape=(p_num,))
            t_input = Input(shape=(n_t,))
            x0_input = Input(shape=(dim,))
            inputs = [p_input] if p_num > 0 else []
            inputs += [x0_input, t_input]

            parsed_node = self.node_parser.parse(data)
            built_node = self.node_builder.build(parsed_node, inputs)
            return built_node

        def test_basic_creation(self):
            self.load_model_replace_dict("simple_node", {node_type: self.ode_node_type})

        def test_const_A_prediction(self):
            for i in self.repeat("test_const_A_prediction"):
                prob = self.generator.linear_problem()
                model = self.load_model_replace_dict(
                    "replace_matrix", {sys_mat: prob.A_str, node_type: self.ode_node_type}
                )
                prediction = model.predict([prob.x_init, prob.t])
                test_utility.assertClose(self, prob.x_sol, prediction, self.close_threshold)

        def test_batches(self):
            for i in self.repeat("test_batches"):
                prob = self.generator.linear_problem()
                model = self.load_model_replace_dict(
                    "replace_matrix", {sys_mat: prob.A_str, node_type: self.ode_node_type}
                )
                prediction = model.predict([prob.x_init, prob.t])
                test_utility.assertClose(self, prob.x_sol, prediction, self.close_threshold)

        def test_with_parameters(self):
            for _ in self.repeat("test_with_parameters"):
                prob = self.generator.linear_problem_with_parameters()
                model = self.load_model_replace_dict(
                    "with_parameters", {sys_mat: prob.A_str, node_type: self.ode_node_type}
                )
                prediction = model.predict([*prob.param_values.T, prob.x_init, prob.t])
                test_utility.assertClose(self, prob.x_sol, prediction, self.close_threshold)

        def test_training(self, epochs=2):
            for _ in self.repeat("test_training"):
                # Generate Model and Problem
                model = self.load_model_replace_dict("with_trainable_weights", {node_type: self.ode_node_type})
                prob = self.generator.linear_samples_for_training(dim=2, n=11)

                y_sol = prob.x_sol
                # 1. evaluate without training, should be bad
                base_quality = model.evaluate(x=[prob.p, prob.x_init, prob.t], y=y_sol)

                # 2. Train on data
                model.fit(x=[prob.p, prob.x_init, prob.t], y=y_sol, validation_split=0.2, epochs=epochs)

                # 3. Evaluate Results, should be better then at the start
                train_quality = model.evaluate(x=[prob.p, prob.x_init, prob.t], y=y_sol)

                self.assertLess(train_quality, base_quality)


class test_linear_ode_node(Wrapper.LinearOdeNodeTesterBase):
    def __init__(self, methodName="runTest", iterations_per_problem=5):
        super().__init__(current_file=__file__, methodName=methodName, iterations_per_problem=iterations_per_problem)
        self.ode_node_type = "linear_ode"


if __name__ == "__main__":
    t = test_linear_ode_node()
    t.test_training()
    unittest.main()
