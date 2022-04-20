from unittest import TestCase
from condensate_speckle.data_io import get_example_data_file_path, load_data, analyze_data
import pandas as  pd
import numpy as np
import math

class TestIo(TestCase):
    def test_data_io(self): 
        data = load_data('simulated_data.csv',data_dir='../example_data').to_numpy()
        assert data[0,0]==0
        
    def test_data_mean(self):
        mean, variance = analyze_data('simulated_data.csv',data_dir='../example_data')
        assert np.allclose(mean, 83.3, rtol=0, atol=1)
        
    def test_data_mean2(self):
        mean, variance = analyze_data('test.csv',data_dir='../example_data')
        data = load_data('test.csv',data_dir='../example_data').to_numpy()
        meta_mean = data[3,1]
        assert np.allclose(mean, 85, rtol=0, atol=1)
