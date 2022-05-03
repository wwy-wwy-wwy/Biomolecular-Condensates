import pandas as pd
from pathlib import Path
import numpy as np
import math
from matplotlib import pyplot as plt


def get_example_data_file_path(filename, data_dir='example_data'):
    """
       input:
       filename: string
       data_dir: string
       return:
       data_path: string
    """
    # Path.cwd() returns the location of the source file currently in use (so
    # in this case io.py). We can use it as base path to construct
    # other paths from that should end up correct on other machines or
    # when the package is installed
    current_directory = Path.cwd()
    data_path = Path(current_directory, data_dir, filename)
    print(data_path)
    return data_path


def load_data(filename, data_dir):
    """
    input:
    ----------
    filename: string
    data_dir: string
    ----------
    return:
    pandas data frame, time points, intensity data
    """
    data_file=get_example_data_file_path(filename,data_dir)
    X = pd.read_csv(data_file, header=None).to_numpy()
    data_pds={'t':X[0],'Intensity':X[1]}
    dataframe=pd.DataFrame(data_pds)
    plt.plot(dataframe['t'],dataframe['Intensity'])
    
    return dataframe

def analyze_data(filename, data_dir):
    """
       input:
       filename: string
       data_dir: string
       return:
       mean and standard deviation of data
    """
    data_file=get_example_data_file_path(filename,data_dir)
    data = pd.read_csv(data_file, header=None, nrows=2).to_numpy()
    return np.mean(data[1]), np.std(data[1])