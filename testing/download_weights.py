import unittest
import tensorflow as tf
from core.astromer import ASTROMER

class TestStringMethods(unittest.TestCase):
    def test_download(self):
        '''
        Sample random window
        '''

        model = ASTROMER()
        model.from_pretrained('macho')
        
if __name__ == '__main__':
    unittest.main()
