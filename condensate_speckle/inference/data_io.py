import pandas as pd
import numpy as np

def load_data(data_file):
    return pd.read_csv(data_file, sep=' ', header=None)