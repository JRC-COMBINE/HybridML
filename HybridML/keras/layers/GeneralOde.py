import math

import casadi
import numpy as np
import tensorflow as tf
from casadi import MX, Function, integrator, vertcat
from HybridML.keras.layers.ArithmeticExpression import calculate_shape_of_expression
from HybridML.keras.layers.ArithmeticExpression import evaluation_environment as ext_evaluation_environment
from HybridML.keras.layers.ArithmeticExpression import find_vars_in_expression
from HybridML.keras.layers.CasadiLinearOde import to_numpy

evaluation_environment = {"exp": casadi.exp, "pi": math.pi, "sin": casadi.sin}
# Make sure that all functions from the arithmetic expression layer are also present in the general ode layer
for key in ext_evaluation_environment:
    assert key in evaluation_environment


def np_shape(tensor):
    return tf.shape(tensor).numpy()


class GeneralOdeLayer(tf.keras.layers.Layer):
    def __init__(self, rhs_expression, *args, **kwargs):
        kwargs["dynamic"] = True
        super().__init__(self, *args, **kwargs)
        self.rhs_expression = rhs_expression
        self.layer_built = False

    def compute_output_shape(self, input_shapes):
        # The 1. input are the x_inits. The 1. dimension of it is the outputshape
        return input_shapes[-1] + input_shapes[-2][-1]

    def _get_number_of_states(self):
        rhs_shape = calculate_shape_of_expression(self.rhs_expression)
        if len(rhs_shape) > 1:
            raise Exception("The rhs has a rank, higher than one. Only rhs with a rank of 1 are allowed.")

        self.state_order = len(rhs_shape)

        if self.state_order == 0:
            self.n_states = 1
        elif self.state_order == 1:
            self.n_states = rhs_shape[0]
        else:
            raise Exception("Odes with an order higher than 1 are not supported.")

    def _determine_input_vars(self):
        self.var_names = find_vars_in_expression(self.rhs_expression)

    def _build_layer(self, param_shapes, x_shape, t_shape):
        """Prebuilds the ode layer.

        Args:
            input_shapes ([Dimension]): [parameters: (samples * n_parameters), x0s: (samples * ode_dims), t_vals: (samples * time_points)]

        Raises:
            Exception: [description]
        """
        # Check dimensionality of the input
        for param_shape in param_shapes:
            if len(param_shape) > 1 and param_shape[1] > 1:
                raise Exception("Only one dimensional inputs with rank one are allowed.")

        p_number = len(param_shapes)
        x_size = x_shape[-1]
        t_size = t_shape[-1]

        self._determine_input_vars()
        self._get_number_of_states()

        assert x_size == self.n_states

        self._build_casadi_fn(p_number, x_size, t_size)
        self.layer_built = True

    def _eval_expression(self, x, t, ps):
        var_dict = dict(zip(self.var_names, ps))
        var_dict["x"] = x
        var_dict["t"] = t
        rhs = eval(self.rhs_expression, evaluation_environment, var_dict)
        if self.state_order == 1:
            rhs = vertcat(*rhs)
        return rhs

    def _build_integrator(self, parameters, x_size):

        # ode inputs
        x = MX.sym("x", x_size)
        t = MX.sym("t", 1)
        dt = MX.sym("dt", 1)

        rhs = self._eval_expression(x, t, parameters)
        transformed_rhs = rhs * dt
        ode = {"x": x, "p": vertcat(*parameters, t, dt), "ode": transformed_rhs}
        F = integrator("F", "cvodes", ode, {})
        return F

    def _build_casadi_fn(self, p_number, x_size, t_size):
        # state_fn inputs
        parameters = [MX.sym(f"p_{i}", 1) for i in range(p_number)]
        x0_sym = MX.sym("x0", x_size)

        # a zero time point is added at the beginning
        # thus the real time series is one longer
        expanded_t_size = t_size + 1

        t_val = MX.sym("t_val", expanded_t_size)

        F = self._build_integrator(parameters, x_size)

        states = list()
        Xk = x0_sym
        for k in range(1, expanded_t_size):  # skip initial value
            # Integrate till end of the interval
            t_diff = t_val[k] - t_val[k - 1]
            t_curr = t_val[k - 1]
            Fk = F(x0=Xk, p=vertcat(*parameters, t_curr, t_diff))
            Xk = Fk["xf"]  # "rxf"
            states.append(Xk)

        self.stateFunction = Function("states", [x0_sym, t_val, *parameters], [vertcat(*states)])
        n_adiont_derivates = 1
        self.gradientFunction = self.stateFunction.reverse(n_adiont_derivates)

    def call(self, inputs):
        """Calculates a solution for the specified ode problem.

        Args:
            inputs ([Tensor]): [parameters: (samples * n_parameters), x0s: (samples * ode_dims), t_vals: (samples * time_points)]

        Returns:
            Tensor: The solution to the specified ode problem. Dimensions: (samples * time_points * ode_dims).
        """

        # Convert inputs to numbers
        x0s, t_vals = inputs[-2:]
        if len(inputs) > 2 and len(inputs[0]) == len(inputs[-1]):
            parameter_list = inputs[:-2]
        else:
            parameter_list = []

        if not self.layer_built:
            param_shapes = [np_shape(param) for param in parameter_list]
            self._build_layer(param_shapes, np_shape(x0s), np_shape(t_vals))

        # Iteratively calculate solution for each sample
        results = []
        for *parameters, x0, t_val in zip(*parameter_list, x0s, t_vals):
            res = self.solve_single_sample(parameters, x0, t_val)
            results.append(res)

        stacked = tf.stack(results)
        target_shape = (*tf.shape(t_vals)[:2], tf.shape(x0s)[-1])
        reshaped = tf.reshape(stacked, target_shape)

        # Casadi returns float64. Cast, if target precision is float32
        casted = tf.cast(reshaped, dtype=tf.keras.backend.floatx())

        return casted

    @tf.custom_gradient
    def solve_single_sample(self, tf_ps, tf_x0, tf_t_val):
        *ps, x0, t_val = to_numpy(*tf_ps, tf_x0, tf_t_val)
        t_expanded = np.insert(t_val, 0, 0)

        out_states = self.stateFunction(x0, t_expanded, *ps).full()
        shaped_out_states = out_states

        def grad_fn(*tf_grad_ys):
            grad_ys = to_numpy(tf_grad_ys[0])
            # at x0 for free x_init, returns list then
            t_grads, x_grads, p_grads = self.gradientFunction(x0, t_expanded, ps, out_states, grad_ys.flatten())
            p_grads = p_grads.full()

            # Casadi returns float64. Cast, if target precision is float32
            p_grads = tf.cast(p_grads, dtype=tf.keras.backend.floatx())
            p_grads = [tf.reshape(p_grad, tf_p.shape) for p_grad, tf_p in zip(p_grads, tf_ps)]

            t_grad, x_init_grad = None, None

            return (*p_grads, t_grad, x_init_grad)

        return shaped_out_states, grad_fn
