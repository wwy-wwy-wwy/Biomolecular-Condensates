import pandas as pd
from pathlib import Path
import numpy as np


def get_example_data_file_path(filename, data_dir='example_data'):
    # Path.cwd() returns the location of the source file currently in use (so
    # in this case io.py). We can use it as base path to construct
    # other paths from that should end up correct on other machines or
    # when the package is installed
    current_directory = Path.cwd()
    data_path = Path(current_directory, data_dir, filename)
    print(data_path)
    return data_path


def load_data(filename,data_dir):
    data_file=get_example_data_file_path(filename,data_dir)
    return pd.read_csv(data_file, header=None)
