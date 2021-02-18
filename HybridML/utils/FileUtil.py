import json


def load_json(path):
    """Loads a json file into a python dictionary."""
    with open(path, "r") as f:
        datastore = json.load(f)
    return datastore
