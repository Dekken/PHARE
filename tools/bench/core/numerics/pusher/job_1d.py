#!/usr/bin/env python3

import pyphare.pharein as ph
from pyphare.pharein import ElectronModel
from tests.simulator import basicSimulatorArgs, makeBasicModel
from tests.diagnostic import dump_all_diags

out = "phare_outputs/diags_1d/"
simInput = {
    "diag_options": {"format": "phareh5", "options": {"dir": out, "mode": "overwrite"}},
    "cells" : 50,
    "refinement_boxes": {},
    "largest_patch_size": [10],
}

ph.Simulation(**basicSimulatorArgs(dim=1, interp=1, **simInput))
model = makeBasicModel(ppc=4e5)
ElectronModel(closure="isothermal", Te=0.12)
dump_all_diags(model.populations)
