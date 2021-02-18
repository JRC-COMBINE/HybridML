import os
import shutil
from typing import Dict, List

import HybridML.utils.FileUtil as FileUtil
from HybridML.building.DataModel import ModelContainer
from HybridML.ModelCreator import KerasModelCreator
from HybridML.NodeRegistry import DefaultNodeRegistry
from HybridML.utils.DataSource import DataSource, FileDataSourceLoader, TimeExpandedDataSourceLoader

template_dir = os.path.join(os.path.dirname(__file__), "templates")

model_template = os.path.join(template_dir, "model_template.json")
project_template = os.path.join(template_dir, "project_template.json")


def create_model(model_description_path):
    """Create a model.
    model_description_path: Path to the model description json file."""

    model_description = FileUtil.load_json(model_description_path)
    return create_model_from_description(model_description)


def create_model_from_description(model_description):
    """Create a model.
    model_description: Model description in form of python dictionaries that result from loading the json description."""
    model_creator = KerasModelCreator(DefaultNodeRegistry())
    model = model_creator.generate_model(model_description)
    return model


def load_model(model_description_path, model_path):
    """Load a saved model from file.
    model_description_path: Path to the model description json file.
    model_path: Path to the saved model."""

    model_creator = KerasModelCreator(DefaultNodeRegistry())
    model_description = FileUtil.load_json(model_description_path)
    model = model_creator.load_model_from_file(model_description, model_path)
    return model


def project_exists(project_dir, name):
    """
    Checks if a project exists
    :param project_dir: parent directory
    :param name: name of the project
    :return: True if project exists, False if it does not
    """
    containing_dir = os.path.join(project_dir, name)
    proj_file_path = os.path.join(containing_dir, name + ".json")
    result = os.path.exists(containing_dir)
    result = result and os.path.isdir(containing_dir)
    result = result and os.path.exists(proj_file_path)
    result = result and os.path.isfile(proj_file_path)
    return result


def path_from_name(name):
    return os.path.join("./projects", name)


def project_file_path_from_name(name):
    return os.path.join("./projects", name, name + ".json")


def open(project_file_path):
    return Project(project_file_path)


def open_project(containing_dir, name):
    """
    Opens an existing project
    :param containing_dir: parent directory of project
    :param name: name of the project
    :return: opened Project
    """
    return Project(os.path.join(containing_dir, name, name + ".json"))


def _prepare_project_files(project_dir, project_name):
    """
    Creates a new project by copying template files for the project and the model configuration.
    :param project_dir: project parent directory
    :param project_name: name of the project
    :return:
    """
    os.makedirs(project_dir, exist_ok=True)
    shutil.copyfile(model_template, os.path.join(project_dir, "model.json"))
    shutil.copyfile(project_template, os.path.join(project_dir, project_name + ".json"))


def create(containing_dir, name):
    """
    Creates a new project
    :param containing_dir: parent directory
    :param name: name of the project
    :return: opened Project
    """
    project_dir = os.path.join(containing_dir, name)
    if project_exists(containing_dir, name):
        print(f"Path {project_dir} already exists. No Project created.")
        return
    _prepare_project_files(project_dir, name)
    print(f"Created new project {name}")
    project_file_path = os.path.join(project_dir, name + ".json")
    return Project(project_file_path)


def open_create(containing_dir, name):
    """
    Opens a project. If the project does not exist, creates it.
    :param containing_dir: project parent directory
    :param name: name of the project (use filename-safe names)
    :return: opened Project
    """
    if project_exists(containing_dir, name):
        return open_project(containing_dir, name)
    else:
        return create(containing_dir, name)


class ProjectFile_JSON_DICT:
    name = "name"
    models = "models"
    model_save_dir = "model_save_dir"
    data_files = "data_files"

    class DataFile:
        name = "name"
        path = "path"
        x = "x"
        y = "y"


class ProjectFile:
    def __init__(self, project_file_path):
        self.project_file_path = project_file_path
        self.root_dir = self._extract_root_path(project_file_path)

        d = FileUtil.load_json(project_file_path)
        self._project_data = d
        self.id = d[ProjectFile_JSON_DICT.name]

        self.model_descriptions = [os.path.join(self.root_dir, model) for model in d[ProjectFile_JSON_DICT.models]]
        self.model_dir = os.path.join(self.root_dir, d[ProjectFile_JSON_DICT.model_save_dir])
        self.data_files = {df["name"]: df for df in d[ProjectFile_JSON_DICT.data_files]}

    def get_model_path(self, model_name):
        return os.path.join(self.model_dir, model_name)

    def get_data_path(self, data_name):
        return os.path.join(self.root_dir, self.data_files[data_name])

    def _extract_root_path(self, file):
        return os.path.dirname(file)


class Project:
    def __init__(self, project_file_path: str):
        self.project_file = ProjectFile(project_file_path)
        self.models: List[ModelContainer] = []
        self.data_sources: Dict[str:DataSource] = {}
        self.model_creator = KerasModelCreator(DefaultNodeRegistry())
        self.root_dir = "."
        self.ds_loader_dict = {None: FileDataSourceLoader(), "time_expanded": TimeExpandedDataSourceLoader()}

    def get_model_files(self) -> List[str]:
        return self.project_file.model_descriptions

    def generate_models(self, files=None) -> List[ModelContainer]:
        """
        Loads model definitions from file and instantiates the models of this Project
        :param files: If given, allows loading model definitions from external files as opposed to the
        Project's model definitions
        :return: List of loaded models
        """
        if files is None:
            model_descriptions = self.project_file.model_descriptions
        else:
            model_descriptions = files
        data_items = []
        for model_desc in model_descriptions:
            if not os.path.isfile(model_desc):
                raise Exception(f"Failed loading model: {model_desc} file does not exist")
            data = FileUtil.load_json(model_desc)
            data_items.append((model_desc, data))

        self.models = self.model_creator.generate_models(data_items)

        print("Models have been generated")
        return self.models

    def load_models(self, files=None) -> List[ModelContainer]:
        if files is None:
            model_descriptions = self.project_file.model_descriptions
        else:
            model_descriptions = files

        data_items = []
        for model_desc in model_descriptions:
            if not os.path.isfile(model_desc):
                raise Exception(f"Failed loading model: {model_desc} file does not exist")
            data = FileUtil.load_json(model_desc)
            data_items.append(data)

        self.models = self.model_creator.load_models(data_items, self.project_file.model_dir)

        print("Models have been loaded")
        return self.models

    def load_data_sources(self) -> List[DataSource]:
        """
        Loads Project's data sources from file into Project's data_sources attribute
        :param split: If True, splits the loaded data into different sets.
        :return: data_sources attribute of Project
        """
        for data in self.project_file.data_files.values():
            path = os.path.join(self.project_file.root_dir, data["path"])
            id = data["name"]
            type = data.get("type")
            self.data_sources[id] = self.ds_loader_dict[type].load(data, path)
        return self.data_sources

    def save_models(self):
        model_dir = self.project_file.model_dir
        for model in self.models:
            model.save(model_dir)
        print(f"Models have been saved in directory {model_dir}")
