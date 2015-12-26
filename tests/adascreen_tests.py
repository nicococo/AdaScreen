import numpy as np
from nose.tools import *


def setup(exms=10, train=5, deps=1, add_intercept=True):
    pass

@with_setup(setup(exms=10, train=0, deps=1, add_intercept=True))
def test_toy_data_setting():
    print "I RAN!"
