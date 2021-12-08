import math
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Regex Patterns
variable_pattern_string = r"([a-zA-Z]+\w*)"
variable_is_no_function_appendix = r"[ ]*($|[^ \w\(])"
variable_pattern = re.compile(variable_pattern_string + variable_is_no_function_appendix)
variable_or_function_pattern = re.compile(variable_pattern_string)
slice_pattern = re.compile(r"\[[\w]?\d+[\w]?\]")
scientific_number_escape_pattern = re.compile(r"\d+\.?\d*e[-+]?\d+")
number_pattern = re.compile(r"-?[\d.]+(?:e-?\d+)?")

# Variables and functions to be used in the arithmetic expression
evaluation_environment = {"exp": tf.math.exp, "pi": math.pi, "sin": tf.sin}


class DummyTensor:
    """Dummy object. It returns itself for all mathematical operations.
    This is used to evaluate python expressions without calculating anything."""

    def combine_two(self, other):
        return self

    def combine_one(self):
        return self

    def combine_n(self, *args, **kwargs):
        return self

    __add__ = combine_two
    __sub__ = combine_two
    __mul__ = combine_two
    __matmul__ = combine_two
    __truediv__ = combine_two
    __floordif__ = combine_two
    __mod__ = combine_two
    __divmod__ = combine_two
    __pow__ = combine_two
    __lshift__ = combine_two
    __rshift__ = combine_two
    __and__ = combine_two

    __xor__ = combine_two
    __or__ = combine_two
    __radd__ = combine_two
    __rsub__ = combine_two
    __rmul__ = combine_two
    __rmatmul__ = combine_two
    __rtruediv__ = combine_two
    __rfloordiv__ = combine_two
    __rmod__ = combine_two
    __rdivmod__ = combine_two
    __rpow__ = combine_two
    __rlshift__ = combine_two
    __rrshift__ = combine_two

    __neg__ = combine_one
    __pos__ = combine_one
    __abs__ = combine_one
    __invert__ = combine_one

    __getitem__ = combine_two

    __call__ = combine_n


def find_vars_in_expression(expression, sort=True):
    """Finds all python variables, used in the expression and returns them while keeping the original order."""

    # Escape numbers in scientific notation to simplify the variable finding.
    escaped_expression = re.sub(scientific_number_escape_pattern, "0", expression)

    variables = variable_pattern.findall(escaped_expression)

    vars2 = [var[0] for var in variables]
    # Magically removes duplicates while preserving the order. Its Stack Overflow Magic from Raymond Hettinger
    result = list(dict.fromkeys(vars2))
    if sort:
        result.sort()
    return result


def find_variables_in_assignment(expression):
    """Finds variables in expression.
    Returns:
        1. variables that are assigned
        2. variables that are used
    """
    if expression.count("=") != 1:
        raise Exception("We need exactly one equals sign for an assignment")

    left_str, right_str = expression.split("=")
    output_vars = find_vars_in_expression(left_str, sort=False)
    input_vars = find_vars_in_expression(right_str)
    return input_vars, output_vars


def build_system_matrix(expression, parameters, name=None):
    """Uses python code evaluation to build the system matrix"""
    input_vars = find_vars_in_expression(expression)

    matrix_is_constant = len(input_vars) == 0
    if matrix_is_constant:
        expression = ArithmeticExpressionLayer._reform_matlab_style_matrices(None, expression)
        result = eval(expression, {})
        return tf.constant(result, tf.keras.backend.floatx())
    else:
        layer = ArithmeticExpressionLayer(expression)
        return layer(parameters)


def calculate_shape_of_expression(expression):
    """Calculates the expression shape by replacing variables and values
    with a dummy object, calculating the expression using eval and taking the shape of it."""
    expression2 = re.sub(number_pattern, "-dummy", expression)
    expression3 = re.sub(slice_pattern, "", expression2)
    expression4 = re.sub(variable_or_function_pattern, "dummy", expression3)
    # Use -dummy to counteract the '-' operator before a number being parsed as a part of the number
    # (x - 4 --> x dummy) --> (x-4 --> x-dummy)

    shadow = eval(expression4, {"dummy": DummyTensor()})
    shadow = np.array(shadow)
    return shadow.shape


class ArithmeticExpressionLayer(Layer):
    """Gets an arethmetic expression string.
    When called it returns the value of the arithmetic expression.
    """

    def __init__(self, expression, *args, **kwargs):
        super(ArithmeticExpressionLayer, self).__init__(*args, **kwargs)
        self.input_var_names = []

        self.adapt_depth_fn = lambda x, y: x
        self.stack_fn = lambda x: x
        self.eval_fn = lambda inputs: inputs

        self._set_variable_names(expression)
        self._set_expression(expression)

    def build(self, input_shape):

        if not any(self.input_var_names):
            raise Exception("Constant expressions are currently not supported")

        self._calculate_depth()
        self._set_adapt_depth_fn()
        self._set_stack_fn()
        self._set_eval_fn()

    def call(self, inputs):
        result = self.eval_fn(inputs)
        return result

    def get_config(self):
        config = super(ArithmeticExpressionLayer, self).get_config()
        config["expression"] = self.expression
        return config

    def _set_eval_fn(self):
        def eval_fn(inputs):
            single_tensor_input = len(self.input_var_names) == 1 and not isinstance(inputs, (list, tuple))
            if single_tensor_input:
                inputs = [inputs]

            # Determine variables in expression and combine with inputs
            variable_dict = dict(zip(self.input_var_names, inputs))

            # Execute python code from expression
            output = eval(self.expression, variable_dict, evaluation_environment)

            # Shape all constants to the shape of the input
            zero_tensor = inputs[0] * 0
            adapted = self.adapt_depth_fn(output, zero_tensor)

            # Stack and shape matrix outputs
            stacked = self.stack_fn(adapted)

            return stacked

        self.eval_fn = eval_fn

    def _set_variable_names(self, expression):
        """Determine the names of the used variables and saves them."""
        if "=" in expression:
            self.input_var_names = self._get_variable_names_for_assignment(expression)
        else:
            self.input_var_names = find_vars_in_expression(expression)

    def _set_expression(self, expression):
        expression = self._reform_assignment_to_expression(expression)
        expression = self._reform_matlab_style_matrices(expression)
        if not expression.strip():
            raise Exception("Expression cannot be empty.")

        self.expression = expression

    def _calculate_depth(self):
        """Calculates the expression depth."""
        shape = calculate_shape_of_expression(self.expression)
        self.depth = len(shape)

    def _set_adapt_depth_fn(self):
        """Depending, on the depth of the expression, precalculates a function that adapts the dimensionality of the arithmetic expression."""

        def rec_calc_adapt_depth_fn(depth):
            """Gets the depth of the resulting array and returns a function that replaces all the constants in the result
            array with tensors of the right shape."""

            if depth == 0:  # Base: Do not reshape
                return lambda inputs, zero: inputs
            if depth == 1:
                return lambda inputs, zero: [inp + zero for inp in inputs]
            else:  # Higher depth: Use recursion to get lower depth result and wrap into list.
                return lambda inputs, zero: [rec_calc_adapt_depth_fn(depth - 1)(inp, zero) for inp in inputs]

        self.adapt_depth_fn = rec_calc_adapt_depth_fn(self.depth)

    def _move_sample_dimension_to_front(self, tensor):
        """Swaps the first and second last dimension of the tensor, so that the sample dimension is at the front"""
        permutation = list(range(len(tensor.shape)))
        permutation.insert(0, permutation.pop(len(permutation) - 2))
        return tf.transpose(tensor, permutation)

    def _set_stack_fn(self):
        """Creates a function that concatenates the lowest level
        of the nestet input arrays of depth 'depth' and stacks the rest."""

        def recursively_calc_stack_function(depth):
            if depth == 0:
                return lambda inputs: inputs
            if depth == 1:
                return lambda inputs: tf.concat(inputs, axis=1)
            else:
                return lambda inputs: [recursively_calc_stack_function(depth - 1)(inp) for inp in inputs]

        concat_and_stack_tensors = recursively_calc_stack_function(self.depth)

        def stack_fn(inputs):
            stacked = tf.stack(concat_and_stack_tensors(inputs))

            transposed = self._move_sample_dimension_to_front(stacked)

            return transposed

        self.stack_fn = stack_fn

    def _reform_assignment_to_expression(self, expression):
        """Removes the left hand side and the '=' from the expression."""
        if "=" not in expression:
            return expression
        else:
            if expression.count("=") != 1:
                raise Exception("We need exactly one equals sign for an assignment")
            _, post = expression.split("=")
            return post

    def _reform_matlab_style_matrices(self, expression):
        """Replaces: 'a,b;c,d' with '[[a,b],[c,d]]'."""
        if ";" in expression:
            expression = expression.replace(";", "],[")
            expression = f"[[{expression}]]"
        elif "," in expression and "[" not in expression:
            expression = f"[{expression}]"
        return expression

    def _get_variable_names_for_assignment(self, expression):
        """When an assignment 'a = b+c' is entered, returns the right hand variables (b,c)"""
        input_vars, output_vars = find_variables_in_assignment(expression)
        if any(var in input_vars for var in output_vars):
            raise Exception(
                "Assignment and assigned side are sharing variables. This is not supported. Expression: ", expression
            )
        if len(output_vars) > 1:
            raise Exception("Assigning multiple variables at once is not supported.")
        return input_vars
