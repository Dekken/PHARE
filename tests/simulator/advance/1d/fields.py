

from ..test_initilization import InitializationTest

ndim = 1

@ddt
class Initialization1dTest(InitializationTest):

    def test_B_is_as_provided_by_user(self):
        for interp_order in [1, 2, 3]:
            self._test_B_is_as_provided_by_user(ndim, interp_order)


