from .parsing.nodes.NeuralNetwork import NNNodeParser
from .parsing.nodes.BasicFunction import BasicFunctionNodeParser
from .parsing.nodes.ArithmeticExpression import ArithmeticExpressionNodeParser
from .parsing.nodes.LinearOde import LinearOdeNodeParser

from .building.nodes.NeuralNetwork import NNNodeBuilder
from .building.nodes.BasicFunction import BasicFunctionNodeBuilder
from .building.nodes.ArithmeticExpression import ArithmeticExpressionNodeBuilder
from .building.nodes.LinearOde import LinearOdeNodeBuilder

from .keras.layers.ArithmeticExpression import ArithmeticExpressionLayer
from .keras.layers.LinearOde import LinearOdeLayer
from .keras.layers.CasadiLinearOde import CasadiLinearOdeLayer


class DefaultNodeRegistry:
    node_parsers = [NNNodeParser(), BasicFunctionNodeParser(), ArithmeticExpressionNodeParser(), LinearOdeNodeParser()]

    node_builders = [
        NNNodeBuilder(),
        BasicFunctionNodeBuilder(),
        ArithmeticExpressionNodeBuilder(),
        LinearOdeNodeBuilder(),
    ]

    custom_losses = {}
    """Needed to load the models.
    If custom Layers, Losses, Metrics or Optimizers are used, Keras needs a way to identify them.
    This is later inserted into tf.keras.load_model(path, custom_objects)"""
    custom_objects = {
        "LinearOdeLayer": LinearOdeLayer,
        "ArithmeticExpressionLayer": ArithmeticExpressionLayer,
        "CasadiLinearOdeLayer": CasadiLinearOdeLayer,
    }
