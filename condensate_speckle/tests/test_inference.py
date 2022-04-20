from condensate_speckle.data_io import get_example_data_file_path,load_data
from condensate_speckle.inference.model import set_model
import numpy as np

import unittest
from unittest import TestCase


class TestModel(TestCase):
    def test_model(self):
        data = load_data('simulated_data.csv',data_dir='../example_data').to_numpy()
        ar1_model=set_model(data)
        estimate = pm.find_MAP(model = ar1_model)
        assert np.allclose(estimate['decay_time'], 10.24767332, rtol=0, atol=1.1)

if __name__ == '__main__':
    unittest.main()
