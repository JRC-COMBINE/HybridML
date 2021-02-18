import tensorflow as tf
from tensorflow_probability.python.math.ode import BDF  # , DormandPrince

from .LinearOde import BaseLinearOdeLayer


class TensorflowLinearOdeLayer(BaseLinearOdeLayer):
    def __init__(self, *args, **kwargs):
        kwargs["dynamic"] = True
        super().__init__(*args, **kwargs)
        self.solver = BDF()  # DormandPrince()

    def solve_ode(self, tf_p, tf_t, tf_x_init):
        """Uses the casadi ode-solver to solve the linear ode"""
        A = tf_p
        t_max = tf.reduce_max(tf_t)
        scaled_t = tf_t / t_max

        def ode_fn(t, x):
            return tf.linalg.matvec(A, x) * t_max

        t_init = 0
        states = self.solver.solve(ode_fn, t_init, tf_x_init, solution_times=scaled_t)

        return states
