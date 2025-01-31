from .wrapper import *
from .test_functions import *
from .test import *

try : from .cec_2011 import CEC2011
except : pass

# Define the package's version
__version__ = "0.1.3"

# Optionally, define the package name
__name__ = "minionpy"
