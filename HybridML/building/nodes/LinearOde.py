from HybridML.building.BaseBuilders import NodeBuilder
from HybridML.building.DataModel import BuiltNodeContainer
from HybridML.keras.layers.ArithmeticExpression import build_system_matrix
from HybridML.keras.layers.CasadiLinearOde import CasadiLinearOdeLayer
from HybridML.keras.layers.LinearOde import ConstSysMatTransformer, LinearOdeLayer
from HybridML.keras.layers.TensorflowLinearOde import TensorflowLinearOdeLayer
from HybridML.parsing.nodes.LinearOde import LinearOdeNode, json_dict


class LinearOdeNodeBuilder(NodeBuilder):
    def __init__(self):
        super().__init__(json_dict.ids)

    def build(self, node: LinearOdeNode, inputs) -> BuiltNodeContainer:
        # Determine the type of LinearOde
        if node.type == json_dict.ids[0]:
            LayerClass = LinearOdeLayer
        elif node.type == json_dict.ids[1]:
            LayerClass = CasadiLinearOdeLayer
        elif node.type == json_dict.ids[2]:
            LayerClass = TensorflowLinearOdeLayer
        else:
            raise Exception("Unkown node type: ", node.type)

        # divide inputs into single input vars
        p_input = inputs[:-2] if len(inputs) > 2 else []

        x_init_input, times_input = inputs[-2], inputs[-1]
        layer_inputs = [x_init_input, times_input]

        sys_mat = build_system_matrix(node.system_matrix_str, p_input, name=node.id + "_sys_mat")
        sys_mat_is_constant = len(inputs) < 3

        if sys_mat_is_constant:
            layer_inputs = ConstSysMatTransformer(sys_mat)(layer_inputs)
        else:
            layer_inputs = [sys_mat] + layer_inputs

        # wire inputs to linear ode layer
        layer = LayerClass()
        outputs = layer(layer_inputs)

        return BuiltNodeContainer(node, outputs, layer)
