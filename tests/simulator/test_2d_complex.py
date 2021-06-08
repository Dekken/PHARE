#!/usr/bin/env python3
import pyphare.pharein as ph #lgtm [py/import-and-import-from]
from pyphare.pharein import Simulation
from pyphare.pharein import MaxwellianFluidModel
from pyphare.pharein import ElectromagDiagnostics,FluidDiagnostics, ParticleDiagnostics
from pyphare.pharein import ElectronModel
from pyphare.simulator.simulator import Simulator
from pyphare.pharein import global_vars as gv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

diag_outputs = "."
time_step_nbr=50,
time_step=.001,

def config():
    Simulation(
        smallest_patch_size=20,
        largest_patch_size=20,
        time_step_nbr=30,
        time_step=.001,
        cells=(120, 60),
        dl=(0.2, 0.2),
        hyper_resistivity=0.01,
        resistivity=0.001,
        diag_options={"format": "phareh5",
                      "options": {"dir": diag_outputs,
                                  "mode":"overwrite"}}
    )
    def density(x, y):
        from pyphare.pharein.global_vars import sim
        L = sim.simulation_domain()[1]
        return 0.5 + 1./np.cosh((y-L*0.25)/0.5)**2 + 1./np.cosh((y-L*0.75)/0.5)**2
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
        return v1 + (v2-v1)*(S(y,Ly*0.25,1) -S(y, Ly*0.75, 1))# + (-w5*y1*w3) + (+w5*y2*w4)
    def bz(x, y):
        return 0.
    def b2(x, y):
        return bx(x,y)**2 + by(x, y)**2 + bz(x, y)**2
    def T(x, y):
        K = 1
        return 1./density(x, y)*(K - b2(x, y)*0.5)
    def vx(x, y):
        return 0.
    def vy(x, y):
        return 0.
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
        "vthx": vthx, "vthy": vthy, "vthz": vthz
    }
    MaxwellianFluidModel(
        bx=bx, by=by, bz=bz,
        protons={"charge": 1, "density": density, **vvv}
    )
    ElectronModel(closure="isothermal", Te=0.0)
    sim = ph.global_vars.sim
    dt = 10*sim.time_step
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

def main():
    config()
    Simulator(gv.sim).run()

if __name__=="__main__":
    main()

    from pyphare.pharesee.hierarchy import hierarchy_from


    eb_hier = None
    eb_hier = hierarchy_from(h5_filename=diag_outputs+"/EM_E.h5", hier=eb_hier)
    eb_hier = hierarchy_from(h5_filename=diag_outputs+"/EM_B.h5", hier=eb_hier)
    from tests.simulator.test_advance import _test_overlaped_fields_are_equal

    test = AdvanceTestBase()
    test._test_overlaped_fields_are_equal(eb_hier, time_step_nbr, time_step)

