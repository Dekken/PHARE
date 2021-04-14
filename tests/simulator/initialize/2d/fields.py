

import unittest
from ddt import ddt, data, unpack
from pyphare.core.box import Box, Box2D, nDBox
from tests.simulator.test_initialization import InitializationTest

import matplotlib
matplotlib.use("Agg")  # for systems without GUI

ndim = 2
interp_orders = [1, 2, 3]

@ddt
class Initialization2dTest(InitializationTest):

    def test_B_is_as_provided_by_user(self):
        for interp_order in interp_orders:
            self._test_B_is_as_provided_by_user(ndim, interp_order)


    @data({"L0": {"B0": Box2D(10, 20)}},
          {"L0": {"B0": Box2D( 2, 12), "B1": Box2D(13, 25)}})
    def test_overlaped_fields_are_equal(self, interp_order, refinement_boxes):
        print(f"test_overlaped_fields_are_equal_{ndim}d")
        hier = super().getHierarchy(interp_order, refinement_boxes, "b")
        for interp_order in interp_orders:
            super()._test_overlaped_fields_are_equal(interp_order, refinement_boxes)

    def test_bulkvel_is_as_provided_by_user(self):
        for interp_order in interp_orders:
            self._test_bulkvel_is_as_provided_by_user(ndim, interp_order)

    def test_density_is_as_provided_by_user(self):
        for interp_order in interp_orders:
            self._test_density_is_as_provided_by_user(ndim, interp_order)

    def test_density_decreases_as_1overSqrtN(self):
        for interp_order in interp_orders:
          self._test_density_decreases_as_1overSqrtN(ndim, interp_order)


if __name__ == "__main__":
    unittest.main()
