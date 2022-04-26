from unittest import TestCase
from condensate_speckle.data_io import get_example_data_file_path, load_data, analyze_data
from condensate_speckle.inference.prior import UniformPrior
import pandas as  pd
import numpy as np
import math

class TestIo(TestCase):
    def test_data_io(self): 
        """
        This function tests data read in.
        """
        data = load_data('simulated_data.csv',data_dir='condensate_speckle/example_data').to_numpy()
        assert data[0,0]==0
        
    def test_data_mean(self):
        """
        This function tests mean of the data.
        """
        mean, variance = analyze_data('simulated_data.csv',data_dir='condensate_speckle/example_data')
        assert np.allclose(mean, 83.3, rtol=0, atol=1)
        
    def test_data_mean2(self):
        """
        This function tests mean and metadata (mean information encoded).
        """
        mean, variance = analyze_data('simulated_data_w_meta.csv',data_dir='condensate_speckle/example_data')
        data = load_data('simulated_data_w_meta.csv',data_dir='condensate_speckle/example_data').to_numpy()
        meta_mean = data[3,1]
        assert np.allclose(mean, meta_mean, rtol=0, atol=1)
        
    def test_uniform_prior(self):
        """
        This function test the priors
        """
        p = np.exp(UniformPrior(0, 1).logp(0.5))
        assert np.allclose(p, 1)
