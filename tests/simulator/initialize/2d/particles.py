

import unittest
from ddt import ddt, data, unpack
from pyphare.core.box import Box, Box1D, nDBox
from tests.simulator.test_initialization import InitializationTest

import matplotlib
matplotlib.use("Agg")  # for systems without GUI

ndim = 1
interp_orders = [1, 2, 3]

@ddt
class Initialization1dTest(InitializationTest):

    def test_nbr_particles_per_cell_is_as_provided(self):
        for interp_order in interp_orders:
            self._test_nbr_particles_per_cell_is_as_provided(ndim, interp_order)



if __name__ == "__main__":
    unittest.main()
