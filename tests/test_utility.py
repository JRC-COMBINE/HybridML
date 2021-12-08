import json
import os
import shutil
import sys
import unittest

import numpy as np

sys.path.append(os.path.split(os.path.dirname(__file__))[0])

from HybridML.ModelCreator import KerasModelCreator  # noqa: E402
from HybridML.NodeRegistry import DefaultNodeRegistry  # noqa: E402


class ModelFromTestJsonCreator:
    def __init__(self, file):
        self.data = load_relative_json(file)
        self.creator = KerasModelCreator(DefaultNodeRegistry())

    def load_model_by_id(self, json_id):
        data = self.data[json_id]
        model = self.creator.generate_models([data])[0]
        return model

    def load_model_replace_dict(self, model_id, replace_dict):
        s = json.dumps(self.data[model_id])
        for to_replace, replace_with in replace_dict.items():
            s = s.replace(to_replace, replace_with)
        data = json.loads(s)

        result = self.creator.generate_models([data])[0]
        return result

    def load_model_replace(self, model_id, to_replace, replace_with):
        return self.load_model_replace_dict(model_id, {to_replace: replace_with})


def load_json(path):
    with open(path, "r") as f:
        datastore = json.load(f)
    return datastore


def load_relative_json(file_path):
    path = file_path + ".json"
    data = load_json(path)
    return data


def delete_folder(d):
    if os.path.exists(d):
        shutil.rmtree(d)


def read_json_replace_dict(path, to_replace):
    with open(path, "r") as file:
        data = file.read()
    for key, value in to_replace.items():
        data = data.replace(key, str(value))
    return json.loads(data)


def read_json_and_replace(path, expression, replace):
    return read_json_replace_dict(path, {expression: replace})


def contains_all_words(text, words):
    return all(word in text for word in words)


def contains_any_word(text, words):
    return any(word in text for word in words)


class TestCase(unittest.TestCase):
    def assertClose(self, expected, actual, threshold=1e-4):
        abs_err = np.abs(expected - actual)
        rel_err = np.abs(np.nan_to_num(abs_err / expected))
        max_err = np.max(rel_err)
        self.assertTrue(
            max_err < threshold, f"The maximum difference of {max_err} exceeded the threshold of {threshold}."
        )
