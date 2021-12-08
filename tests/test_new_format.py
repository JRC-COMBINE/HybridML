import os
import sys
import unittest

import numpy as np

import test_utility

sys.path.append(os.path.split(os.path.dirname(__file__))[0])

import HybridML.Project as Project  # noqa: E402
from HybridML.ModelCreator import KerasModelCreator  # noqa: E402
from HybridML.NodeRegistry import DefaultNodeRegistry  # noqa: E402


class test_new_format(test_utility.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.data = test_utility.load_relative_json(__file__)
        self.creator = KerasModelCreator(DefaultNodeRegistry())

    def test_model_creation(self):
        result = self.creator.generate_models([self.data["general"]])[0]
        self.assertEquals(result.model.name, "Peter")

    def test_project_creation(self):
        project_path = os.path.dirname(__file__)
        project_name = "test_new_format_project"
        project_folder = os.path.join(project_path, project_name)

        # project_path = os.path.join(project_folder, project_name + ".json")
        models_dir = os.path.join(project_folder, "models")
        test_utility.delete_folder(models_dir)

        project = Project.open_create(project_path, project_name)

        project.generate_models()
        project.save_models()

        models = project.load_models()

        project.save_models()

        data = [np.array([[1, 2]]), np.array([[3, 4]])]

        for model in models:
            model.predict(data)

        test_utility.delete_folder(models_dir)

    def test_keras_model_structure(self):
        pass

    def test_error_on_wrong_input(self):
        self.assertRaises(Exception, lambda: self.creator.generate_models(self.data))

    def test_error_on_short_format(self):
        with self.assertRaises(Exception) as cm:
            self.creator.generate_models([self.data["single_number_layer"]])
        self.assertTrue(
            test_utility.contains_any_word(cm.exception.args[0], ["Single", "number", "definition", "deactivated"])
        )


if __name__ == "__main__":
    t = test_new_format()
    t.test_project_creation()
    unittest.main()
