from tests.simulator import test_simulator as tst

import unittest
import phare.pharein as ph, numpy as np, math

from phare.pp.diagnostics import Diagnostics


class InitValueValidation(unittest.TestCase):
    diag_options = lambda diag_out_dir: {
        "diag_options": {"format": "phareh5", "options": {"dir": diag_out_dir},}
    }

    def _simulate_diagnostics(self, dim, interp, input):
        self.dman, self.sim, self.hier = create_simulator(dim, interp, **input)
        self.dman.dump(0, 1)
        del self.dman, self.sim, self.hier  # force hdf5 flush
        return Diagnostics.extract(ph.globals.sim)

    def tearDown(self):
        for k in ["dman", "sim", "hier"]:
            if hasattr(self, k):
                delattr(self, k)
        tst.reset()

    def truncate(self, number, digits) -> float:
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper


def basicSimulatorArgs(dim: int, interp: int, **kwargs):
    cells = [65 for i in range(dim)]
    if "cells" in kwargs:
        cells = kwargs["cells"]
    if not isinstance(cells, list):
        cells = [cells]
    dl = [1.0 / v for v in cells]
    b0 = [[10 for i in range(dim)], [50 for i in range(dim)]]
    boundary = ["periodic" for i in range(dim)]
    args = {
        "interp_order": interp,
        "smallest_patch_size": 5,
        "largest_patch_size": 64,
        "time_step_nbr": 1000,
        "final_time": 1.0,
        "boundary_types": boundary,
        "cells": cells,
        "dl": dl,
        "max_nbr_levels": 2,
        "refinement_boxes": {"L0": {"B0": b0}},
        "diag_options": {},
    }
    for k, v in kwargs.items():
        if k in args:
            args[k] = v
    return args


def defaultPopulationSettings():
    background_particles = 0.1  # avoids 0 density
    xmax = ph.globals.sim.simulation_domain()[0]
    pi_over_xmax = np.pi / xmax
    pop_fn = {
        1:{
        "density": lambda x: 1.0 / np.cosh((x - xmax * 0.5)) ** 2 + background_particles,
        "vbulkx": lambda x: np.sin(1 * pi_over_xmax * x),
        "vbulky": lambda x: np.sin(1 * pi_over_xmax * x),
        "vbulkz": lambda x: np.sin(1 * pi_over_xmax * x),
        "vthx": lambda x: 1,
        "vthy": lambda x: 1,
        "vthz": lambda x: 1,
      },2 :{
        "density": lambda x, y: 1.0 / np.cosh((x - xmax * 0.5)) ** 2 + background_particles,
        "vbulkx": lambda x, y: np.sin(1 * pi_over_xmax * x),
        "vbulky": lambda x, y: np.sin(1 * pi_over_xmax * x),
        "vbulkz": lambda x, y: np.sin(1 * pi_over_xmax * x),
        "vthx": lambda x, y: 1,
        "vthy": lambda x, y: 1,
        "vthz": lambda x, y: 1,
      }
    }
    dim = len(ph.globals.sim.cells)
    return {
        "charge": 1,
        "density": pop_fn[dim]["density"],
        "vbulkx": pop_fn[dim]["vbulkx"],
        "vbulky": pop_fn[dim]["vbulky"],
        "vbulkz": pop_fn[dim]["vbulkz"],
        "vthx": pop_fn[dim]["vthx"],
        "vthy": pop_fn[dim]["vthy"],
        "vthz": pop_fn[dim]["vthz"],
    }


def makeBasicModel(extra_pops={}):
    xmax = ph.globals.sim.simulation_domain()[0]
    pi_over_xmax = np.pi / xmax
    EM_fn = {
        1:{
        "bx":lambda x: np.cos(2 * pi_over_xmax * x),
        "by":lambda x: np.sin(1 * pi_over_xmax * x),
        "bz":lambda x: np.cos(2 * pi_over_xmax * x),
        "ex":lambda x: np.sin(1 * pi_over_xmax * x),
        "ey":lambda x: np.cos(2 * pi_over_xmax * x),
        "ez":lambda x: np.sin(1 * pi_over_xmax * x)
      },2 :{
        "bx":lambda x, y: np.cos(2 * pi_over_xmax * x),
        "by":lambda x, y: np.sin(1 * pi_over_xmax * x),
        "bz":lambda x, y: np.cos(2 * pi_over_xmax * x),
        "ex":lambda x, y: np.sin(1 * pi_over_xmax * x),
        "ey":lambda x, y: np.cos(2 * pi_over_xmax * x),
        "ez":lambda x, y: np.sin(1 * pi_over_xmax * x)
      }
    }
    pops = {
        "protons": {
            **defaultPopulationSettings(),
            "nbr_part_per_cell": 100,
            "init": {"seed": 1337},
        },
        "alpha": {
            **defaultPopulationSettings(),
            "nbr_part_per_cell": 1000,
            "init": {"seed": 13337},
        },
    }
    pops.update(extra_pops)
    dim = len(ph.globals.sim.cells)
    return ph.MaxwellianFluidModel(
        bx=EM_fn[dim]["bx"],
        by=EM_fn[dim]["by"],
        bz=EM_fn[dim]["bz"],
        ex=EM_fn[dim]["ex"],
        ey=EM_fn[dim]["ey"],
        ez=EM_fn[dim]["ez"],
        **pops
    )


def create_simulator(dim, interp, **input):

    tst.reset()
    ph.globals.sim = None
    ph.Simulation(**basicSimulatorArgs(dim, interp, **input))
    extra_pops = {}
    if "populations" in input:
        for pop, vals in input["populations"].items():
            extra_pops[pop] = defaultPopulationSettings()
            extra_pops[pop].update(vals)

    model = makeBasicModel(extra_pops)
    if "diags_fn" in input:
        input["diags_fn"](model)
    ph.populateDict()
    hier = tst.make_hierarchy()
    sim = tst.make_simulator(hier)
    sim.initialize()
    return [tst.make_diagnostic_manager(sim, hier), sim, hier]
