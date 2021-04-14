

from tests.simulator.test_initilization import InitializationTest

ndim = 2

@ddt
class Initialization2dTest(InitializationTest):

    def test_B_is_as_provided_by_user(self):
        for interp_order in [1, 2, 3]:
            self._test_B_is_as_provided_by_user(ndim, interp_order)


