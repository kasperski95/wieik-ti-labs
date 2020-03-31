from keras.datasets import boston_housing
from .mlutils import *


def task3():
    data = Data.from_keras(boston_housing.load_data())
    data.apply_standard_scaler()
    data.print_info()

    nn = NeuralNetwork(data)
    nn.grid_search({"layers": [(80, 80), (40, 40, 40, 40)]})
    nn.train(50)
    nn.print_info()

    nn.grid_search(
        {"layers": [(80, 80), (0, 0, 0, 0)], "dropout": [0.5],}
    )
    nn.train(50)
    nn.print_info()
