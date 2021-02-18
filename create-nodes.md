# On creating new node types


Creating a model consists of to phases:
1. Parsing the model-json and creating an abstract representation.
2. Using the abstract representation to actually build the model.
This bears the advantage, that the math-library used in the model can be changed, by only chaning the building process.

## First Phase
### ParsedNode
The base class `ParsedNode` can be found in HybridML.parsing.DataModel.
Implement your `CustomParsedNode` in `parsing/nodes/Custom.py`, where `Custom` is the name of your new node.
Abstract representation of the parsed node.
Contains information that is filled in by the parser.
#### Mandatory implementation:
* `determine_output_sizes` is needed to propagate the size of the in and outputs. It returns a list of the sizes of the outputs
#### Optional overwrites:
* `get_input_ids` in the base class the value of the `inputs` attribute from the model-json is used. If another way of determining the ids of the inputs is wished this can be overwritten.
* `get_output_ids`: similar to `get_input_ids`
* `__init__`: in the base class only saves the json-data.
### NodeParser
Can be found in HybridML.parsing.BaseParsers.
Implement your `CustomNodePaser` in `parsing/nodes/Custom.py`.
Class used to parse a model-json dict and create a `ParsedNode`.
The `parses_types` attribute saves a list of strings, saying which types of nodes it can parse. The types are thoses, that are used in the model-json to type a node, eg. "nn", "ArithmeticExpression","linear_ode".
Usually the parser does not do much apart from handing the information to the ParsedNode.


## Second Phase
### NodeBuilder
The base class `NodeBuilder` can be found in HybridML.building.BaseBuilders.
Implement your `CustomNodeBuilder` in `building/nodes/Custom.py`
This class creates a Node and puts it into a BuiltNodeContainer.
If it is necessary to create a new custom keras layer, it can be saved in `keras/layers/Custom.py`


## Add it to the Tool
In the file `NodeRegistry.py`, all parsers and builders are registered.
In case a new Tensorflow Model or Layer has been implemented, it has to be added to the `custom_objects` dictionary.


## Example
A simple implementation of a new node can be found in the BasicFunctionLayer