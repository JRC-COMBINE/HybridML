import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.split(os.path.dirname(__file__))[0])

from HybridML.ModelCreator import KerasModelCreator  # noqa: E402
from HybridML.NodeRegistry import DefaultNodeRegistry  # noqa: E402

from test_utility import TestCaseTimer, delete_folder, load_relative_json  # noqa: E402


class test_containers(TestCaseTimer):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.data = load_relative_json(__file__)
        self.creator = KerasModelCreator(DefaultNodeRegistry())

    def check_outputs_of_simple_model(self, model):
        data = [np.array([1]), np.array([2]), np.array([3]), np.array([4])]
        prediction = model.predict(data)
        self.assertEqual(prediction, [np.array([[1.0]]), np.array([[4.0]])])

        prediction2 = model.predict(data, consider_additional_outputs=True)
        self.assertEqual(prediction2, [np.array([[1.0]]), np.array([[4.0]]), np.array([[2.0]]), np.array([[3.0]])])

    def test_additional_outputs_generation(self):
        model = self.creator.generate_models([self.data["test_additional_outputs_generation"]])[0]
        self.check_outputs_of_simple_model(model)

    def test_additional_outputs_simple_loading(self):
        model = self.creator.generate_models([self.data["test_additional_outputs_generation"]])[0]
        # Creat Dir to save models
        model_dir = os.path.join(os.path.dirname(__file__), "temp", "models")
        if os.path.exists(model_dir):
            delete_folder(model_dir)
        os.makedirs(model_dir)
        model.save(model_dir)

        model = self.creator.load_models([self.data["test_additional_outputs_generation"]], model_dir)[0]
        self.check_outputs_of_simple_model(model)
        delete_folder(model_dir)


if __name__ == "__main__":
    unittest.main()
