#!/usr/bin/env python3

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import pyphare.pharein as ph #lgtm [py/import-and-import-from]
from pyphare.pharein import Simulation
from pyphare.pharein import MaxwellianFluidModel
from pyphare.pharein import ElectromagDiagnostics,FluidDiagnostics, ParticleDiagnostics
from pyphare.pharein import ElectronModel
from pyphare.simulator.simulator import Simulator, startMPI
from pyphare.pharein import global_vars as gv

import numpy as np

diag_outputs="."
time_step_nbr=10
final_time=30.
time_step=.001

def config():

    Simulation(
        smallest_patch_size=20,
        largest_patch_size=20,
        time_step_nbr=time_step_nbr,
        time_step=time_step,
        #boundary_types="periodic",
        cells=(120, 60),
        dl=(0.2, 0.2),
        #refinement="tagging",
        #max_nbr_levels = 3,
        #refinement_boxes={"L0": {"B0": [(50, ), (150, )]},
        #                  "L1":{"B0":[(125,),(175,)]}},
        hyper_resistivity=0.01,
        resistivity=0.000,
        diag_options={"format": "phareh5",
                      "options": {"dir": diag_outputs,
                                  "mode":"overwrite"}}
    )

    def density(x, y):
        from pyphare.pharein.global_vars import sim
        L = sim.simulation_domain()[1]
        return 1.#0.5 + 1./np.cosh((y-L*0.25)/0.5)**2 + 1./np.cosh((y-L*0.75)/0.5)**2


    def S(y, y0, l):
        return 0.5*(1. + np.tanh((y-y0)/l))


    def by(x, y):
        from pyphare.pharein.global_vars import sim
        Lx = sim.simulation_domain()[0]
        Ly = sim.simulation_domain()[1]
        w1 = 0.2
        w2 = 1.
        x0 = (x - 0.5 * Lx);
        y1 = (y - 0.25 * Ly);
        y2 = (y - 0.75 * Ly);
        w3 = np.exp(-(x0*x0 + y1*y1) / (w2*w2));
        w4 = np.exp(-(x0*x0 + y2*y2) / (w2*w2));
        w5 = 2.0*w1/w2;
        return 0# (w5 * x0 * w3) + ( -w5 * x0 * w4);


    def bx(x, y):
        from pyphare.pharein.global_vars import sim
        Lx = sim.simulation_domain()[0]
        Ly = sim.simulation_domain()[1]
        w1 = 0.2
        w2 = 1.
        x0 = (x - 0.5 * Lx);
        y1 = (y - 0.25 * Ly);
        y2 = (y - 0.75 * Ly);
        w3 = np.exp(-(x0*x0 + y1*y1) / (w2*w2));
        w4 = np.exp(-(x0*x0 + y2*y2) / (w2*w2));
        w5 = 2.0*w1/w2;
        v1=-1
        v2=1.
        return 1.#v1 + (v2-v1)*(S(y,Ly*0.25,1) -S(y, Ly*0.75, 1))# + (-w5*y1*w3) + (+w5*y2*w4)

    def bz(x, y):
        return 0.

    def b2(x, y):
        return bx(x,y)**2 + by(x, y)**2 + bz(x, y)**2

    def T(x, y):
        K = 1
        return 1.#/density(x, y)*(K - b2(x, y)*0.5)

    def vx(x, y):
        return 1.

    def vy(x, y):
        return 1.

    def vz(x, y):
        return 0.

    def vthx(x, y):
        return T(x, y)

    def vthy(x, y):
        return T(x, y)

    def vthz(x, y):
        return T(x, y)

    vvv = {
        "vbulkx": vx, "vbulky": vy, "vbulkz": vz,
        "vthx": vthx, "vthy": vthy, "vthz": vthz,
        "nbr_part_per_cell":100, "init":{"seed":1234}
    }

    MaxwellianFluidModel(
        bx=bx, by=by, bz=bz,
        protons={"charge": 1, "density": density,  **vvv}
    )

    ElectronModel(closure="isothermal", Te=0.0)

    sim = ph.global_vars.sim
    dt = 1*sim.time_step
    nt = sim.final_time/dt+1
    timestamps = dt * np.arange(nt)
    print(timestamps)

    for quantity in ["E", "B"]:
        ElectromagDiagnostics(
            quantity=quantity,
            write_timestamps=timestamps,
            compute_timestamps=timestamps,
        )

    for quantity in ["density", "bulkVelocity"]:
        FluidDiagnostics(
            quantity=quantity,
            write_timestamps=timestamps,
            compute_timestamps=timestamps,
        )

   for popname in ("protons",):
      for name in ["domain", "levelGhost", "patchGhost"]:
          ParticleDiagnostics(quantity=name,
                              compute_timestamps=timestamps,
                              write_timestamps=timestamps,
                              population_name=popname)


def main():

    config()
    startMPI()
    Simulator(gv.sim).run()

    from pyphare.cpp import cpp_lib
    cpp = cpp_lib()
    if cpp.mpi_rank() == 0:
        from datetime import datetime
        from pyphare.pharesee.hierarchy import hierarchy_from
        from tests.simulator.test_advance import AdvanceTestBase

        test = AdvanceTestBase()
        for time_step_idx in range(time_step_nbr + 1):
            time     = "{:.10f}".format(time_step_idx * ph.global_vars.sim.time_step)
            datahier = None
            datahier = hierarchy_from(h5_filename=diag_outputs+"/EM_E.h5", time=time, hier=datahier)
            datahier = hierarchy_from(h5_filename=diag_outputs+"/EM_B.h5", time=time, hier=datahier)
            test.base_test_overlaped_fields_are_equal(datahier, time)
            test.base_test_overlaped_fields_are_equal(datahier, time)


if __name__=="__main__":
    main()

