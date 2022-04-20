from condensate_speckle.data_io import get_example_data_file_path,load_data
from condensate_speckle.inference.model import set_model
import numpy as np

import unittest
from unittest import TestCase


class TestModel(TestCase):
    def test_model(self):
        data = load_data(get_example_data_file_path('example_data.txt'))
        set_model(data)
        assert np.allclose(np.exp(UniformPrior(3, 5).logp(4)), .5)

if __name__ == '__main__':
    unittest.main()
