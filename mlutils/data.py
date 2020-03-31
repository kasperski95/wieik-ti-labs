import numpy as np
import pandas as pd
import sklearn.datasets as datasets
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate


class Data:
    @staticmethod
    def from_keras(dataset):
        return Data(dataset[0][0], dataset[0][1], dataset[1][0], dataset[1][1])

    @staticmethod
    def from_xy(X, Y, split_options={"test_size": 0.25}):
        return Data(*train_test_split(X, Y, **split_options))

    def __init__(self, X_train, y_train, X_test, y_test):
        self._X_train = np.asarray(X_train)
        self._y_train = np.asarray(y_train)
        self._X_test = np.asarray(X_test)
        self._y_test = np.asarray(y_test)

    def print_info(self, title="DATA INFO"):
        print(title)
        df = pd.DataFrame(
            {
                "No. of Features": [self.get_features_count()],
                "No. of Training Samples": [self._X_train.shape[0]],
                "No. of Test Samples": [self._X_test.shape[0]],
                "Avg. SD of Training Set": [
                    self._calculate_avg_std(self._calculate_avg_std(self._X_train))
                ],
                "Avg. Mean of Training Set": [
                    self._calculate_avg_mean(self._calculate_avg_mean(self._X_train))
                ],
            }
        )
        df = df.transpose()
        df.to_string(header=False)
        print(tabulate(df, tablefmt="grid"))
        print()

    def apply_standard_scaler(self):
        scaler = StandardScaler()
        self._X_train = scaler.fit_transform(self._X_train)
        self._X_test = scaler.transform(self._X_test)
        return self

    def get_features_count(self):
        return self._X_train[0].shape[0]

    def get_X_train(self):
        return self._X_train

    def get_y_train(self):
        return self._y_train

    def get_X_test(self):
        return self._X_test

    def get_y_test(self):
        return self._y_test

    def _calculate_avg_std(self, V):
        return reduce(lambda el, acc: acc + np.std(el), V, 0) / V.size

    def _calculate_avg_mean(self, V):
        return reduce(lambda el, acc: acc + np.average(el), V, 0) / V.size
