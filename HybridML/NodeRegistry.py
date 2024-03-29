from .building.nodes.ArithmeticExpression import ArithmeticExpressionNodeBuilder
from .building.nodes.BasicFunction import BasicFunctionNodeBuilder
from .building.nodes.ExistingModel import ExistingModelNodeBuilder
from .building.nodes.GeneralOde import GeneralOdeNodeBuilder
from .building.nodes.LinearOde import LinearOdeNodeBuilder
from .building.nodes.NeuralNetwork import NNNodeBuilder
from .keras.layers.ArithmeticExpression import ArithmeticExpressionLayer
from .keras.layers.CasadiLinearOde import CasadiLinearOdeLayer
from .keras.layers.GeneralOde import GeneralOdeLayer
from .keras.layers.LinearOde import LinearOdeLayer
from .parsing.nodes.ArithmeticExpression import ArithmeticExpressionNodeParser
from .parsing.nodes.BasicFunction import BasicFunctionNodeParser
from .parsing.nodes.ExistingModel import ExistingModelNodeParser
from .parsing.nodes.GeneralOde import GeneralOdeNodeParser
from .parsing.nodes.LinearOde import LinearOdeNodeParser
from .parsing.nodes.NeuralNetwork import NNNodeParser


class DefaultNodeRegistry:
    node_parsers = [
        NNNodeParser(),
        BasicFunctionNodeParser(),
        ArithmeticExpressionNodeParser(),
        LinearOdeNodeParser(),
        GeneralOdeNodeParser(),
        ExistingModelNodeParser(),
    ]

    node_builders = [
        NNNodeBuilder(),
        BasicFunctionNodeBuilder(),
        ArithmeticExpressionNodeBuilder(),
        LinearOdeNodeBuilder(),
        GeneralOdeNodeBuilder(),
        ExistingModelNodeBuilder(),
    ]

    custom_losses = {}
    """Needed to load the models.
    If custom Layers, Losses, Metrics or Optimizers are used, Keras needs a way to identify them.
    This is later inserted into tf.keras.load_model(path, custom_objects)"""
    custom_objects = {
        "LinearOdeLayer": LinearOdeLayer,
        "GeneralOdeLayer": GeneralOdeLayer,
        "ArithmeticExpressionLayer": ArithmeticExpressionLayer,
        "CasadiLinearOdeLayer": CasadiLinearOdeLayer,
    }
