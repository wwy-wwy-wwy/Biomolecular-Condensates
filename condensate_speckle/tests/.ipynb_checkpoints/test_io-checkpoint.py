from unittest import TestCase
from condensate_speckle.data_io import get_example_data_file_path, load_data
import pandas as  pd

class TestIo(TestCase):
    def test_data_io(self):
        
        data = load_data('simulated_data.csv',data_dir='../example_data')
        assert data[0,0]==0
