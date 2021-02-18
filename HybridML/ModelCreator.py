import os
from typing import List
from HybridML.building.DataModel import ModelContainer
from HybridML.building.BaseBuilders import NodeBuilderSelector
from HybridML.building import ModelBuilder
from HybridML.keras.Builders import KerasModelBuilder, KerasModelLoader
from HybridML.parsing.ParserImplementation import NodeParserSelector, NetParser, ModelParser
from HybridML.keras.Builders import KerasNetBuilder


class ModelCreator:
    """Abstract base class that allows a model to be created or loaded from file."""

    def generate_model(self, path) -> List[ModelContainer]:
        raise NotImplementedError()

    def load_model(self, path) -> List[ModelContainer]:
        raise NotImplementedError()


class ModelParserFactory:
    """Factory for model parsers. Can create model parsers used for loading a model from file or for generating a model.
    Parsers parse a model-json file and create an abstract representation
    that can be put into a builder to actually build a model."""

    def create_for_generation(node_parsers) -> ModelParser:
        """Create a parser, that can be used to generate a new model."""
        selector = ModelParserFactory._create_selector(node_parsers)
        return ModelParserFactory._create_model_parser(selector)

    def create_for_loading() -> ModelParser:
        """Create a parser, that can be used to load a model from file."""
        return ModelParser()

    def _create_selector(node_parsers):
        selector = NodeParserSelector()
        for parser in node_parsers:
            selector.append_parser(parser)
        return selector

    def _create_model_parser(node_parser):
        net_parser = NetParser(node_parser)
        return ModelParser(net_parser)


class ModelBuilderFactory:
    """Factory for model builders. Can create model builders used for loadgin a model from file or for generating a model.
    Builders get a ParsedModel, an abstract representation of a model, which is then used to build an actual model."""

    def create_for_generation(node_builders, custom_losses) -> ModelBuilder:
        """Create a builder, that can be used to generate a new model."""
        selector = NodeBuilderSelector()
        for builder in node_builders:
            selector.append_builder(builder)
        return ModelBuilderFactory._create_model_builder(selector, custom_losses)

    def create_for_loading(custom_objects, custom_losses) -> ModelBuilder:
        """Create a builder, that can be used to load a model from file."""
        return KerasModelLoader(custom_objects=custom_objects, custom_losses=custom_losses)

    def _create_model_builder(node_builder, custom_losses) -> KerasModelBuilder:
        net_builder = KerasNetBuilder(node_builder)
        return KerasModelBuilder(net_builder, custom_losses)


class KerasModelCreator(ModelCreator):
    """Can be used to read a model-json and generate a tensorflow model by it. Or to load an existing model from file.
    It ombines ModelParser and ModelBuilder to archieve this task."""

    def __init__(self, node_registry):
        self.node_registry = node_registry

    def generate_models(self, data_items) -> [ModelContainer]:
        """Generate multiple new models.
        :param data_items: Contains a list of json-dicts, with the data from the model-jsons.
        """

        if not isinstance(data_items, List):
            raise Exception("Argument error: Expected data_items to be List.")
        node_parsers = self.node_registry.node_parsers
        node_builders = self.node_registry.node_builders
        custom_losses = self.node_registry.custom_losses

        model_parser = ModelParserFactory.create_for_generation(node_parsers)
        model_builder = ModelBuilderFactory.create_for_generation(node_builders, custom_losses)

        models = self._do_creation(model_parser, model_builder, data_items)

        return models

    def generate_model(self, data_item) -> ModelContainer:
        """Generate a new model."""
        return self.generate_models([data_item])[0]

    def load_models(self, data_items, model_dir) -> [ModelContainer]:
        """Load multiple models from a directory.
        :param data_items: Contains a list of json-dicts, with the data from the model-jsons.
        :param model_dir: The directory containing the saved models.
        """
        models = []
        for data in data_items:
            name = data["name"]
            model_path = os.path.join(model_dir, name + ".h5")
            model = self.load_model_from_file(data, model_path)
            models.append(model)
        return models

    def load_model_from_file(self, model_description, model_path):
        """Load a single model from file.
        :param data: Contains the data from the model-json.
        :param model_dir: The path to the saved model.
        """
        if not os.path.exists(model_path):
            raise Exception(f"The Model does not exist: {model_path}")

        model_parser = ModelParserFactory.create_for_loading()
        model_builder = ModelBuilderFactory.create_for_loading(
            self.node_registry.custom_objects, self.node_registry.custom_losses
        )

        parsed_model = model_parser.parse(model_description)
        container = model_builder.build(parsed_model, model_path)
        return container

    def _unpack_name_data(self, item):
        if isinstance(item, tuple):
            name, data = item
        else:
            name = None
            data = item
        return name, data

    def _do_creation(self, parser, builder, data_items):
        models = []
        for item in data_items:
            name, data = self._unpack_name_data(item)
            parsed_model = parser.parse(data)
            model = builder.build(parsed_model)
            models.append(model)
        return models
