import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate


def r_square(y_true, y_pred):
    from keras import backend as K

    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


class NeuralNetwork:
    @staticmethod
    def create_model(layers, dropout=0.0):
        model = Sequential()

        for i in range(len(layers)):
            if i == 0:
                model.add(Dense(layers[i], input_shape=(13,)))
            else:
                model.add(Dense(layers[i]))

            model.add(Dropout(dropout))

        model.add(Dense(1))
        model.compile(loss="mse", metrics=[r_square], optimizer="Adam")

        return model

    def __init__(self, data):
        self._data = data
        self._model = None
        self._recent_grid_search = None
        self._training_info = None

    def grid_search(self, param_grid):
        model = KerasRegressor(
            build_fn=NeuralNetwork.create_model, epochs=10, verbose=False
        )
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, refit=False
        )
        grid_search.fit(self._data.get_X_train(), self._data.get_y_train())
        self._recent_grid_search = grid_search

    def train(self, n_epochs):
        self._model = NeuralNetwork.create_model(
            **self._recent_grid_search.best_params_
        )
        self._training_info = self._model.fit(
            self._data.get_X_train(),
            self._data.get_y_train(),
            epochs=n_epochs,
            verbose=False,
            validation_data=(self._data.get_X_test(), self._data.get_y_test()),
        )

    def print_info(self, title="NN INFO"):
        print(title)
        print(
            "R2 score",
            r2_score(
                self._data.get_y_test(), self._model.predict(self._data.get_X_test())
            ),
        )
        df = pd.DataFrame(self._recent_grid_search.best_params_)
        df = df.transpose()
        df.to_string(header=False)
        print("hyperparameters\n" + tabulate(df, tablefmt="grid"))

        plt.plot(
            self._training_info.history["loss"], "r", marker=".", label="Train Loss"
        )
        plt.plot(
            self._training_info.history["val_loss"],
            "b",
            marker=".",
            label="Validation Loss",
        )

        plt.title("Train loss")
        plt.legend()
        plt.xlabel("Epoch"), plt.ylabel("Error")
        plt.grid()
        plt.show()
        print()
