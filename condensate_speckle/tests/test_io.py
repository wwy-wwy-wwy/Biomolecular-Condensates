from unittest import TestCase
from condensate_speckle.data_io import get_example_data_file_path, load_data
import pandas as  pd

class TestIo(TestCase):
    def test_data_io(self):
        data = load_data(get_example_data_file_path('simulated_data.txt'))
        assert data[1,0]>73
