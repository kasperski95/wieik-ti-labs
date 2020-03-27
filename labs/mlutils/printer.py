import pandas as pd
from tabulate import tabulate
from textwrap import indent


class Printer:
    def __init__(self, name):
        self._name = name

    def print_table(self, data, transpose=False, title=None):
        tablefmt = "github"
        df = pd.DataFrame(data)
        if transpose:
            df = df.transpose()
        if title:
            print(f"{self._name}.{title}:\n")
        print(indent(tabulate(df, tablefmt=tablefmt, headers=data.keys()), "  "))
        print()
