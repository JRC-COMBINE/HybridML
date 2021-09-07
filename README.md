<meta name="google-site-verification" content="64MKUXUdvgq1SuzTc1-tdbVsPbpC6my_xjSXSSq1Bvg" />

# HybridML - Open Source Platform for Hybrid Modeling

- [HybridML - Open Source Platform for Hybrid Modeling](#HybridML---Open-Source-Platform-for-Hybrid-Modeling)
  - [Introduction](#introduction)
  - [Technologies](#technologies)
  - [Demos](#demos)
    - [Simple Demo](#simple-demo)
    - [Extrapolation Demo](#extrapolation-demo)
    - [Theophylline Demo](#theophylline-demo)
    - [Fluvoxamin Demo](#fluvoxamin-demo)
  - [Poject Structure](#poject-structure)
  - [Software Architecture](#software-architecture)
    - [Creation](#creation)
      - [1. Parsing](#1-parsing)
      - [2. Building](#2-building)
    - [NodeBuilder, NodeParser and Node](#nodebuilder-nodeparser-and-node)
  - [Fluvoxamine](#fluvoxamine)
  - [How to define a model-json](#how-to-define-a-model-json)
  - [How to implement new node types](#how-to-implement-new-node-types)
## Introduction
This tool aims to create hybrid machine learning models using tensorflow and keras.
In this context hybrid means, that the models consist of submodels that are connected in a feed forward graph.
These submodels, called nodes are machine learning models themselves.
We currently support neural networks, mechanistic models (formulas) and linear odes as nodes.
It is possible to add more types of models.
More can be found under "How to to implement new node types".


## Technologies
* tensorflow
* casadi
* numpy
* matlotlib
* tqdm

Install the requirements with:
```pip install -r ./requirements.txt```

## Demos
We include several demos of varying complexity into the project.


### Simple Demo
Basic functionality of the Tool:


Models can be loaded by using the Project Module.
The behaviour of the models mimics the behaviour of tensorflow/keras models.
A simple demo can be found in [simple_demo.py](demo/simple_demo.py).
Here is an excerpt of `simple_demo.py`.
```python
from HybridML import Project

# Create model from description
model_description_path =  "simple_model.json"
model = Project.create_model(model_description_path)

# Train model with generated data
model.fit(X_train, y_train, validation_split=0.8, epochs=1)

# Evaluate model on test data
model.evaluate(X_test, y_test)

# Save model to file
model_path = model_description_path + ".h5"
model.save_to_file(model_path)

# Load model from file. The model description is needed to load the model.
loaded_model = Project.load_model(model_description_path, model_path)

# Predict both models and compare output
prediction = model.predict(X_test)
loaded_model_prediction = loaded_model.predict(X_test)
assert np.all(prediction - loaded_model_prediction < 1e-5)
```

### Extrapolation Demo
[extrapolation_demo.py](demo/extrapolation_demo/extrapolation_demo.py)
Contains a project with at hybrid model and a black box model.
It demonstrates the project api and anecdotally shows the extrapolation capability of hybrid models.

### Theophylline Demo
[theoph_demo.py](demo/theoph/theoph_demo.py)
This demo contains a complex model, that uses neural nets and mechanistic models to estimate parameters for an ode system.
The model resembles a two compartiment model for drug concentration in the blood system.
It makes use of the Theophylline dataset, that comes with the R language.

### Fluvoxamin Demo
[flux_demo.py](demo/flux/flux_demo.py), 
[structured_flux_demo.py](demo/flux/structured_flux_demo.py)
Similar type of model to the theophylline demo.
Contains a more refined model for a use case that is closer to the industry.


## Poject Structure
HybridML can also be used with the project architecture.
A project contains a name, one or multiple models and one or multiple datasources.
A new model can be created by using:
```python
import Project
project = Project.create(containing_dir, "project_name")
```
Then a file of the name `project_name.json` is created in the containing directory.
There, new models and datasources can be added.
To open an already existing model the method  `Project.open(project_file_path)` can be used, where `project_file_path` is the path to the formerly created `project_name.json`.
The project class has the ability to create and load all of its models at once, using the methods `generate_models()` and `load_models()`.
Note, that it is not necessary to state the location of the model description or the saved model file, as both are handled by the project instance.
All models can be saved, using the `save()` method.
The data sources can be loaded, using `load_data_sources()`, where we again don't have to state the location of the data sources, because they are handled by the project instance.
The data sources can then be accessed via the `data_sources` propoerty, which is a list of data_sources.


## Software Architecture

The model creation process consists of the two phases parsing and building.
In the parsing phase, the model description file is read and loaded into memory.
In the building phase, the parsed model is then built.
The two phases are independant from each other.

### Creation
The model creation process is coordinated by the `ModelCreator`.
It has a `NodeRegistry`, which contains parsers and builders for all possible types of nodes, as well as all custom Tensorflow objects.
Furthermore, it has a `ModelParser`, which recieves the model-json.
The parsed output is then given to its `ModelBuilder`, which performs the final step of building the tensorflow model.

#### 1. Parsing
At first the simple parameters of the model, such as `name`, `mse`, `loss`, etc are parsed.
Then the network structure of the model is parsed by an instance of `NetParser`.
The parsing itself consists of two passes.
In the first pass, all nodes of the network are parsed by `NodeParsers`.
The resulting `ParsedNode`s are connected with `NodeConnors`, which are represented by variable names in the model-json.s
In the second pass the parser walks through the network and propagates the input and output sizes of each node thorugh the `ParsedNodes` and `NodeConnectors`.

#### 2. Building
In the first step, a `NetworkBuilder` is used to build the network.
The tensorflow/keras functional api is used to construct the network.
At first, `Input`s are built, which represent the `NodeConnectors`, labelled as inputs in the model description.
Then the builder walks throught the network, building the nodes using `NodeBuilders`, and connecting them to their input tensors, and collecting all input and output tensors in a dictionary.

This parsed network is then used to build a `model` and an `additional_outputs_model`.


### NodeBuilder, NodeParser and Node
For each type of node there is an implementation of each of the three classes `NodeBuilder`, `NodeParser`, `Node`.
The parsers and builders select the correct implementation during the process.
More on that in [How to implement new node types](create-nodes.md).


## Fluvoxamine
An example of the usage is the fluvoxamine use case.
The goal is to fit a model that predicts the concentration of the drug fluvoxamine in the blood of a patient.
The inputs of the model are a set of covariates of the patient (eg. weight, gender, ...), the initial dose and a time series.
The output is a concentration for each time point.
Input sizes: [covariates: (n_covariates), init_dose:(1), time_series:(n_time_steps)]
Output Sizes: [(n_time_steps)]

The chosen model is a [pharmacokinetic three-compartement model](http://www.turkupetcentre.net/petanalysis/pk_3cm.html).
The model estimates the k parameters and solves the ode-system.
The k parameters are estimated, using a black box.
The outputs of the blackbox are transposed and scaled to initial guesses.

The fluvoxamine use case can be found in [flux_demo.py](demo/flux/flux_demo.py).


## How to define a model-json
See [model-json.md](model-json.md).

## How to implement new node types
See [create-nodes.md](create-nodes.md).
