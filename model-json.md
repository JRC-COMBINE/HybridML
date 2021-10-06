# Anatomy of a Model Json File
A model json file contains the description of a model.
This description is read by HybridML to generate models and to load previously saved models.

## Meta Parameters
* `name` can be choosen freely
* `loss, metrics, optimizer` can be chosen from the corresponding tensorflow/keras objects.
* Furthermore comments can be added at will.
```json
{
    "name": "my model",
    "loss": "mse",
    "metrics": ["accuracy"],
    "optimizer": "adam",
    "comment": "A comment",
    ...
}
```

## Custom Losses
All loss functions defined in keras (https://keras.io/api/losses/) are supported. In addition, it is possible to use a completely custom loss function, which has to be a callable taking two arguments: `y_true` for the ground truth and `y_pred` for the prediction.

### Imports for Custom Loss Functions

Inside a custom loss function, fast tensor operations supported by tensorflow or keras should be used.
```python
import tensorflow as tf
from keras import backend as K
```

### Example Loss Function
(from https://keras.io/api/losses/)
```python
def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`
```

Following this example, the custom loss function can be used with the HybridML model by supplying a reference to the function in `model.compile` after loading the model from the JSON description:

```python
model.compile(optimizer="adam",loss=my_loss_fn)
```

The loss function originally specified in the model's JSON description is ignored in this case.

## Inputs
* `id` assigns the input a unique id. The id is used to reference the input in the model.
* `size` determines how many parameters go into the input. Can be `null`, if the length is variable.

```json
    "inputs": [
        {
            "comment": "Covariates loaded using flux_datasource.py",
            "id": "cov",
            "size": 8
        },
        {
            "id": "dose",
            "size": 1
        },
        {
            "id": "t",
            "size": null
        }
    ]
```
## Outputs
* `outputs` name the node connectors from the model that are outputs. These will be used as output for training the model.
* `additional_outputs` name the node connectors that can be predicted, but are not used for training. These can be used to get the inner state of the model.

```json
    "outputs": [
        "ode_output"
    ],
    "additional_outputs":[
        "k1","k2"
    ]
```    
## Nodes
The model is defined as feed forward network.
It consists of nodes, which themselves can be models.
The nodes are connected with node connectors.
* `type` determines which type a node has.
* Each type has a unique `id` which should not contain spaces.
  * The id is used to identify the node when debugging or when showing the model.
### 1. Blackbox/NeuralNetwork
* `"type":"nn"`
* `inputs` is a list of node connectors which are then stacked to form the input for the neural network.
* `outputs` defines the node connectors that contain the output of the neural net. There are two cases
    1. There is one output: The output has the size of the last layer of the neural network.
    2. There are multiple outputs: The number of the outputs has to mach the size of the last layer of the neural network. The output of the neural network is split to match the node connectors.
* `layers` is a list of layer objects that form the neural network.
  * `size` is the size of the layer.
  * `activation` is the activation function of the layer. All activation functions from tensorflow/keras are possible, such as [`tanh,relu,sigmoid`]. Setting it to `None` results in a linear activation
  * `kernel_regularizer` applies a kernel regularizer to the layer. Possible are `L1` and `L2`. The regularization factor is appended in parenthesis.
  * `activity_regularizer` is similar to `kernel_regularizer


```json
{
        "id": "BlackBox",
        "type": "nn",
        "inputs": [
            "inp"
        ],
        "layers": [
            {
                "activation": "tanh",
                "size": 10
            },
            {
                "activation": "None",
                "size": 2,
                "kernel_regularizer" : "L1(0.1)",
                "activity_regularizer" : "L2(0.1)"
            }
        ],
        "outputs": [
            "out1",
            "out2",
        ]
    }
```

### 2. Arithmetic Expression
* `"type" : "arithmetic"`
* `expression` defines an expression to be evaluated.
  * It has the form `output = calculation(input)` where output is a single new node connnector, that carries the output of the calculation.
  * Possible calculations:
    * `+-*/` basic calculations
    * `**` powers
    * `(a+b)*c` parenthesis
    * `input[0]`, `all_compartments[:,:,1]` slicing. Here the `:` means keeping the whole dimension and slice only the ones with an actual index.
    * **experimental:** `[[a,b],[c,d]]` building multidimensional outputs
    * Generally most things possible in python. **This gives great opportunity to break the model.**
```json
{
    "id": "expression",
    "type": "arithmetic",
    "expression": "k_out = k_in*1000 + 1.573615e+03"
}
```

### 3. Basic Function
*Will possibly be deprecated, due doubling functionality with arithmetic expressions*
* `inputs` is a list of node connector inputs.
* `outputs` is a single new node connector, carrying the result of the calculation
* `type` Possible types:
  * [`addition, multiplication, substraction, average, maximum, minimum, concatenate`]

```json
{
    "type" : "addition",
    "inputs" : ["in1", "in2"],
    "output" : "out"
}
```

### 4. Linear Ode
This node solves a special case of a linear ode, that is characterized by a system matrix.
* `"type" : "linear_ode"` implementation of the closed form solution of the ode special case.
* `"type" : "casadi_linear_ode"` implementation using a general ode solver over casadi/sundials.
* `time_series_input` the node connector containing the time series for the sample.
* `initial_value_input` the node connector containing the initial value of the ode for the sample.
* `system_matrix` an arithmetic expression containing the system matrix.
    * Can be constant or dynamic.
    * Matrix can be written in python style `[[a,b],[c,d]]` or in matlab style `a, b; c, d`.
* `output` single node connector containing the result of the linear ode.
  * *Attention* The output has one more dimension as other node connectors (samples x time x ode_dimension). When further processed it might be necessary to slice it to get one of the states. This can be archieved through an arithmetic expression layer with the expression: `out = ode_output[:,:,2]`, replacing `2` with the desired state number.


### Defining custom nodes
More nodes can be defined.
Read (create-json.md)[create-json.md] for reference.
