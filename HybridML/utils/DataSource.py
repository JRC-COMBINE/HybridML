import numpy as np
import csv
import os
from collections import Iterable
import pandas as pd


class DataSource:
    def load(self, shuffle=True) -> None:
        raise NotImplementedError()

    def get_train_data(self) -> None:
        raise NotImplementedError()

    def get_val_data(self) -> None:
        raise NotImplementedError()

    def get_test_data(self) -> None:
        raise NotImplementedError()


class FileDataSourceLoader:
    def load(self, data, path):
        x = data["x"]
        y = data["y"]
        return FileDataSource(x, y, path).load(split=False)


class teds_json_const:
    """TimeExpandedDataSourceLoader json constants, contains json identifiers"""

    identifier = "identifier"
    properties = "properties"
    time = "time"
    target = "target"


class TimeExpandedDataSourceLoader:
    def load(self, data, path):
        # get information on role the columns of the input file take
        identifier_id = data[teds_json_const.identifier]
        property_ids = data[teds_json_const.properties]
        time_id = data[teds_json_const.time]
        target_id = data[teds_json_const.target]
        ids = [identifier_id, *property_ids, time_id, target_id]
        # read input file
        df = pd.read_csv(path)
        # check that all chosen columns exist
        for id in ids:
            assert id in df.columns
        # group by subject
        groups = df.groupby(df[identifier_id])
        keys = groups.keys.unique()

        X = []
        for _ in range(len(property_ids) + 1):
            X.append([])
        y = []
        for key in keys:
            group = groups.get_group(key)
            for i, id in enumerate(property_ids):
                X[i].append([group[id].iloc[0]])
            X[-1].append(group[time_id].values)
            y.append(group[target_id].values)
        X = [np.array(samples) for samples in X]
        y = np.array(y)
        return TimeExpandedDataSource(X, y)


class TimeExpandedDataSource(DataSource):
    def __init__(self, X, y):
        super(TimeExpandedDataSource, self).__init__()
        self.X = X
        self.y = y

    def get_train_data(self):
        return self.X, self.y


class FileDataSource(DataSource):
    val_split_percentage = 0.7
    test_split_percantage = 0.9

    def __init__(self, x_sizes, y_sizes, path) -> None:
        self.path = path
        if not isinstance(x_sizes, Iterable):
            x_sizes = [x_sizes]
        if not isinstance(y_sizes, Iterable):
            y_sizes = [y_sizes]
        self.x_sizes = x_sizes
        self.y_sizes = y_sizes
        self.split1 = 0
        self.split2 = 0
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []

    def load(self, shuffle=True, split=False):
        data, length = self.__get_csv_data(self.path)
        x, y = self.__shape_data(data, shuffle)
        if split:
            split1 = int(length * self.val_split_percentage)
            split2 = int(length * self.test_split_percantage)
            self.__split_data(x, y, split1, split2)
        else:
            self.xs = x
            self.ys = y
        return self

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_val_data(self):
        return self.x_val, self.y_val

    def get_test_data(self):
        return self.x_test, self.y_test

    def __get_csv_data(self, path):
        if not os.path.isfile(path):
            raise Exception(f"The specified data file does not exist: {path}")
        data = []
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            first_line = True
            length = 0
            for row in csv_reader:
                if first_line:
                    first_line = False
                else:
                    length += 1
                    data.append(row)
        data = np.array(data).astype(float)
        return data, data.shape[0]

    def __shape_data(self, data, shuffle):
        if shuffle:
            np.random.shuffle(data)
        xs = []
        pos = 0
        for x_size in self.x_sizes:
            xs.append(data[:, pos : (pos + x_size)])
            pos += x_size
        ys = []
        for y_size in self.y_sizes:
            ys.append(data[:, pos : (pos + y_size)])
            pos += y_size
        return xs, ys

    def __split_data(self, xs, ys, split1, split2):
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []

        for i, x in enumerate(xs):
            self.x_train.append(xs[i][0:split1])
            self.x_val.append(xs[i][split1:split2])
            self.x_test.append(xs[i][split2:-1])
        for i, y in enumerate(ys):
            self.y_train.append(ys[i][0:split1])
            self.y_val.append(ys[i][split1:split2])
            self.y_test.append(ys[i][split2:-1])
