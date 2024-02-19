import unittest
import warnings

warnings.simplefilter('error')

from data_tests import *
from dataset_tests import *
from ltsf_tests import *
from nn_tests import *
from utils_tests import *


if __name__ == '__main__':
    unittest.main()
