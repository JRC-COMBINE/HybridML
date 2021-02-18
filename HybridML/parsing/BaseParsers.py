import collections
import numpy as np


class Parser:
    """Base implementation of a parser.
    Parsers get a model-json file and create an abstract representation that can be put into a builder to actually."""

    def parse(self, data):
        raise Exception("Not Implemented")


class NodeParser(Parser):
    """Base implementation of a node parser.
    Saves, which types of nodes it is able to parse."""

    def __init__(self, parses_types):
        super().__init__()
        self.net = None
        if isinstance(parses_types, (collections.Sequence, np.ndarray)) and not isinstance(parses_types, str):
            self.parses_types = parses_types
        else:
            self.parses_types = [parses_types]

    def parse(self, data):
        raise NotImplementedError()
