import tensorflow as tf
import tqdm


class ConstSysMatTransformer(tf.keras.layers.Layer):
    """Transforms a constant System Matrix to a list of System matrices. Used before Linear Ode Layers."""

    def __init__(self, system_matrix, **kwargs):
        super().__init__(self, **kwargs)
        self.system_matrix = system_matrix

    def call(self, inputs):
        x_init, t = inputs
        n_samples = tf.shape(x_init)[0]
        A = tf.tile(self.system_matrix[tf.newaxis], (n_samples, 1, 1))
        return (A, x_init, t)


class BaseLinearOdeLayer(tf.keras.layers.Layer):
    def __init__(self, constant_system_matrix=None, **kwargs):
        if "dynamic" not in kwargs:
            kwargs["dynamic"] = True
        super(BaseLinearOdeLayer, self).__init__(**kwargs)
        self.system_matrix_is_constant = constant_system_matrix is not None
        self.system_matrix_tensor = constant_system_matrix

    def closed_form_ode_solve_single(self, A, t, x_init):
        """Returns the analytical solution to the linear ode.

        Parameters:
            A (n x dim x dim): system matrix tensor
            t (n x None): time series
            x_init (n x dim): initial values of ode)

        Returns:
            res (n x dim):  the analytical solution to the linear ode
        """
        # https://www.qc.uni-freiburg.de/files/kueng_differentialgleichungen_2013.pdf, Matrixexponential:
        exp = tf.linalg.expm(t[:, tf.newaxis, tf.newaxis] * A[tf.newaxis])
        res = tf.linalg.matvec(exp, x_init)

        return res

    def solve_ode(self, A, t, x_init):
        return self.closed_form_ode_solve(A, t, x_init)

    def call(self, inputs):
        """Calculates the results to the linear ode.
        n: number of samples
        dim: dimension of samples
        n_t: number of points in time
        Parameters:
            inputs[-2]: x_init (n x dim): Initial value for ode at t=0
            inputs[-1]: time_series (n x n_t): Time series to solve the ode for
            inputs[0]: system_matrix (n x dim x dim): System matrix only if not constant.
        Returns:
            result (n x n_t x dim)
        """
        # divide inputs into single input variables
        As, x_inits, time_series_list = inputs

        results = []
        # iterate over samples and calculate result

        elements = zip(As, x_inits, time_series_list)

        for (A, x_init, time_series) in tqdm.tqdm(elements, "Iterating over samples", total=As.shape[0]):
            res = self.solve_ode(A, time_series, x_init)
            if hasattr(res, "states"):
                res = res.states
            results.append(res)

        result = tf.stack(results)

        # Casadi returns float64. Cast, if target precision is float32
        result = tf.cast(result, dtype=tf.keras.backend.floatx())

        return result

    def get_config(self):
        return {"constant_system_matrix": self.system_matrix_tensor}

    def compute_output_shape(self, input_shapes):
        # The 1. input are the x_inits. The 1. dimension of it is the outputshape
        return input_shapes[-1] + input_shapes[-2][-1]


class LinearOdeLayer(BaseLinearOdeLayer):
    def __init__(self, *args, **kwargs):
        # Layer is static and thus can be converted into computational graph
        kwargs["dynamic"] = False
        super(LinearOdeLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        """Returns the analytical solution to the linear ode.

        Parameters:
            A (n x dim x dim): system matrix tensor
            t (n x None): time series
            x_init (n x dim): initial values of ode)

        Returns:
            res (n x dim):  the analytical solution to the linear ode
        """

        # seperate inputs
        A, x_init, t = inputs

        # get dimensions
        n_samples = tf.shape(x_init)[0]
        n_states = x_init.shape[-1]
        t_dim = tf.shape(t)[-1]

        # make A and t broadcastable by
        # 1. Expanding the dimensions of t and A to 4
        t_ = t[:, :, None, None]
        A_ = A[:, None, :, :]
        # 2. Repeating t and A in some dimensions to match the dimensionality of the respective other
        t__ = tf.tile(t_, (1, 1, n_states, n_states))
        A__ = tf.tile(A_, (1, t_dim, 1, 1))

        # adapt shape of x_init for matvec calculation
        x_init_ = tf.reshape(x_init, (n_samples, 1, n_states))

        # perform actual calculation res = exp(t * A) @ x
        exp = tf.linalg.expm(t__ * A__)
        res = tf.linalg.matvec(exp, x_init_)

        return res
