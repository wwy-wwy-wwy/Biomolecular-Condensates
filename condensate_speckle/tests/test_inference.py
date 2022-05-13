from condensate_speckle.data_io import get_example_data_file_path,load_data
from condensate_speckle.inference.model import set_model, set_double_precision_model
from condensate_speckle.inference.simulation import simulate_single_decay_data, simulate_double_decay_data
import numpy as np

from matplotlib import pyplot as plt
import pymc3 as pm
import arviz as az
import statsmodels.api as sm
import csv

import unittest
from unittest import TestCase


class TestModel(TestCase):
    
    def test_model(self):
        """
        reads in data and run in the set up model, checks if the function returns a pymc3 model.
        """
        data = load_data('[110, 145]_intensity.csv',data_dir='condensate_speckle/example_data').to_numpy()
        quantization=255
        ar1_model=set_model(data[1,:],quantization, '2h')
        estimate = pm.find_MAP(model = ar1_model)
        #assert np.allclose(estimate['decay_time'], 8, rtol=0, atol=1.1)
        self.assertTrue(isinstance(ar1_model, pm.Model))

    def test_double_precision_model(self):
        """
        reads in data and run in the set up model, checks if the function returns a pymc3 model.
        """
        data = load_data('[110, 145]_intensity.csv',data_dir='condensate_speckle/example_data').to_numpy()
        quantization=255
        ar1_model=set_double_precision_model(data[1,:],quantization)
        self.assertTrue(isinstance(ar1_model, pm.Model))
        
class TestSimulation(TestCase):
    
    def test_single_decay(self):
        """
        This function tests whether data simulation function works for a singel decay time model.
        """
        simulated_data_single, simulated_t_single=simulate_single_decay_data(5000,10,255,80,20,5,1)
        real_mean = np.average(simulated_data_single)
        assert (real_mean>80 and real_mean<90)
    
    def test_double_decay(self):
        """
        This function tests whether data simulation function works for a double decay time model.
        """
        simulated_data_double, simulated_t_double=simulate_double_decay_data(5000,3,300,0.5,255,80,40,8,3)
        real_mean = np.average(simulated_data_double)
        assert (real_mean>80)

if __name__ == '__main__':
    unittest.main()
