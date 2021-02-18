import os
import shutil
import json
import numpy as np
import unittest
import time

profiling_list = []


class TestCaseTimer(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        profiling_list.append((self.id(), t))


def assertClose(testcase, expected, actual, threshold=1e-4):
    abs_err = expected - actual
    rel_err = np.nan_to_num(abs_err / expected)
    max_err = np.max(rel_err)
    testcase.assertTrue(
        max_err < threshold, f"The maximum difference of {max_err} exceeded the threshold of {threshold}."
    )


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
