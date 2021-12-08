import numpy as np
import tensorflow as tf
from casadi import MX, Function, integrator, vertcat

from .LinearOde import BaseLinearOdeLayer

algorithm = "cvodes"


def to_numpy(*tensors):
    result = [t.numpy() if hasattr(t, "numpy") and callable(getattr(t, "numpy")) else t for t in tensors]
    if len(tensors) == 1:
        return result[0]
    else:
        return result


class CasadiLinearOdeLayer(BaseLinearOdeLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialized = False
        self.stateFunction = None
        self.gradientFunction = None
        # self.jacFunction = None

    def build(self, input_shapes):
        t_shape = input_shapes[-1]
        time_dimension_is_set = t_shape[-1] is not None
        if time_dimension_is_set:
            p_shape, x_shape, t_shape = input_shapes
            p_size = p_shape[-2] * p_shape[-1]
            t_size = t_shape[-1] + 1
            x_size = x_shape[-1]
            self.build_state_fn(p_size, t_size, x_size)

    def lazy_build_state_fn(self, *input_shapes):
        if not self.initialized:
            self.build_state_fn(*input_shapes)

    def build_state_fn(self, p_size, t_size, x_size):
        # state_fn inputs
        x0_sym = MX.sym("x0", x_size)
        t_val = MX.sym("t_val", t_size)

        # ode inputs
        A = MX.sym("A", (x_size, x_size))
        x = MX.sym("x", x_size)
        t0 = MX.sym("t0", 1)
        dt = MX.sym("dt", 1)
        rhs = dt * A @ x  # dt * rhs, for time dependent odes; t becomes t0+t*dt after transformation
        # full example call for model rhs function for reference:
        # ode = {'x': model.x, 't': t, 'p': cs.vertcat(model.p, u, t0, dt),
        #  'ode': dt * model.ode(t0 + t * dt, model.x, u_release, model.p), 'quad': cs.vertcat(model.x, u_release)}
        ode = {"x": x, "p": vertcat(A.reshape((p_size, 1)), t0, dt), "ode": rhs}
        F = integrator("F", "cvodes", ode, {})

        states = list()
        Xk = x0_sym
        for k in range(1, t_size):  # skip initial value
            # Integrate till the end of the interval
            t_diff = t_val[k] - t_val[k - 1]
            t_curr = t_val[k - 1]
            Fk = F(x0=Xk, p=vertcat(A.reshape((p_size, 1)), t_curr, t_diff))
            Xk = Fk["xf"]  # 'rxf'
            states.append(Xk)

            # jacFk = jacF(x0=Xk, p=vertcat(A.reshape((dim_p,1)), t_val[k-1], dtgrid[k-1])) # full jacobian
            # dXk = jacFk["jac_xf_p"][:,:dim_p]
            # jacs.append(dXk)

        self.stateFunction = Function("states", [t_val, x0_sym, A], [vertcat(*states)])  # [x0_sym, A] for free x_init
        # self.jacFunction = Function('gradients',[x0_sym,A],jacs) # full jacobian -> not necessary
        n_adjoint_derivates = 1
        self.gradientFunction = self.stateFunction.reverse(n_adjoint_derivates)
        self.initialized = True

    @tf.custom_gradient
    def solve_ode(self, tf_p, tf_t, tf_x_init):
        """Uses the casadi ode-solver to solve the linear ode"""

        p, t_val, x0 = to_numpy(tf_p, tf_t, tf_x_init)
        t_expanded = np.insert(t_val, 0, 0)

        self.lazy_build_state_fn(p.size, t_expanded.size, x0.size)
        out_states = self.stateFunction(t_expanded, x0, p).full()  # (x0,p) for free x0
        shaped_out_states = out_states.reshape((-1, x0.size))

        def grad_fn(*grad_ys):
            # at x0 for free x_init, returns list then
            t_grads, x_grads, p_grads = self.gradientFunction(
                t_expanded, x0, p, out_states, grad_ys[0].numpy().flatten()
            )
            p_grads = p_grads.full()

            # Casadi returns float64. Cast, if target precision is float32
            p_grads = tf.cast(p_grads, dtype=tf.keras.backend.floatx())

            t_grad, x_init_grad = None, None

            return p_grads, t_grad, x_init_grad

        return shaped_out_states, grad_fn
