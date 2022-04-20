from unittest import TestCase
from condensate_speckle.data_io import get_example_data_file_path, load_data
import pandas as  pd
import numpy as np

class TestIo(TestCase):
    def test_data_io(self): 
        data = load_data('simulated_data.csv',data_dir='../example_data').to_numpy()
        
        assert data[0,0]==0
        
    def test_data_mean(self):
        mean, variance = analyze_data('simulated_data.csv',data_dir='../example_data')
        assert math.isclose(mean, 85.386)
        
        
