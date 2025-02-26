import unittest
import warnings

from data_tests import *
from dataset_tests import *
from ltsf_tests import *
from nn_tests import *
from utils_tests import *

# This needs to go *after* the imports, because certain dependencies can emit
# warnings on import under some circumstances.
warnings.simplefilter('error')


if __name__ == '__main__':
    unittest.main()
