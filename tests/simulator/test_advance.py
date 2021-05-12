

from pyphare.cpp import cpp_lib
cpp = cpp_lib()

from pyphare.simulator.simulator import Simulator, startMPI
from pyphare.pharesee.hierarchy import hierarchy_from, merge_particles
from pyphare.pharein import MaxwellianFluidModel
from pyphare.pharein.diagnostics import ParticleDiagnostics, FluidDiagnostics, ElectromagDiagnostics
from pyphare.pharein import ElectronModel
from pyphare.pharein.simulation import Simulation, supported_dimensions
from pyphare.pharesee.geometry import level_ghost_boxes, hierarchy_overlaps
from pyphare.core.gridlayout import yee_element_is_primal
from pyphare.pharesee.particles import aggregate as aggregate_particles, any_assert as particles_any_assert
import pyphare.core.box as boxm
from pyphare.core.box import Box, Box1D
import numpy as np
import unittest
from ddt import ddt, data, unpack


from tests.simulator import spawn_tests_from

@ddt
class AdvanceTestBase(unittest.TestCase):

    def ddt_test_id(self):
        return self._testMethodName.split("_")[-1]


    def _density(*xyz):
        x = xyz[0]
        return 0.3 + 1./np.cosh((x-6)/4.)**2

    def getHierarchy(self, interp_order, refinement_boxes, qty, nbr_part_per_cell=100,
                     diag_outputs="phare_outputs", density = _density,
                     smallest_patch_size=5, largest_patch_size=20,
                     cells=120, time_step=0.001, model_init={},
                     dl=0.1, extra_diag_options={}, time_step_nbr=1, timestamps=None, ndim=1):

        from pyphare.pharein import global_vars
        global_vars.sim = None
        startMPI()
        extra_diag_options["mode"] = "overwrite"
        extra_diag_options["dir"] = diag_outputs
        Simulation(
            smallest_patch_size=smallest_patch_size,
            largest_patch_size=largest_patch_size,
            time_step_nbr=time_step_nbr,
            time_step=time_step,
            boundary_types=["periodic"] * ndim,
            cells=[cells] * ndim,
            dl=[dl] * ndim,
            interp_order=interp_order,
            refinement_boxes=refinement_boxes,
            diag_options={"format": "phareh5",
                          "options": extra_diag_options}
        )


        def S(x,x0,l):
            return 0.5*(1+np.tanh((x-x0)/l))

        def bx(*xyz):
            return 1.

        def by(*xyz):
            from pyphare.pharein.global_vars import sim
            L = sim.simulation_domain()
            _ = lambda i: 0.1*np.cos(2*np.pi*xyz[i]/L[i])
            return np.asarray([_(i) for i,v in enumerate(xyz)]).prod(axis=0)

        def bz(*xyz):
            from pyphare.pharein.global_vars import sim
            L = sim.simulation_domain()
            _ = lambda i: 0.1*np.sin(2*np.pi*xyz[i]/L[i])
            return np.asarray([_(i) for i,v in enumerate(xyz)]).prod(axis=0)

        def vx(*xyz):
            from pyphare.pharein.global_vars import sim
            L = sim.simulation_domain()
            _ = lambda i: 0.1*np.cos(2*np.pi*xyz[i]/L[i])
            return np.asarray([_(i) for i,v in enumerate(xyz)]).prod(axis=0)

        def vy(*xyz):
            from pyphare.pharein.global_vars import sim
            L = sim.simulation_domain()
            _ = lambda i: 0.1*np.cos(2*np.pi*xyz[i]/L[i])
            return np.asarray([_(i) for i,v in enumerate(xyz)]).prod(axis=0)

        def vz(*xyz):
            from pyphare.pharein.global_vars import sim
            L = sim.simulation_domain()
            _ = lambda i: 0.1*np.sin(2*np.pi*xyz[i]/L[i])
            return np.asarray([_(i) for i,v in enumerate(xyz)]).prod(axis=0)

        def vth(*xyz):
            return 0.01 + np.zeros_like(xyz[0])

        def vthx(*xyz):
            return vth(*xyz)

        def vthy(*xyz):
            return vth(*xyz)

        def vthz(*xyz):
            return vth(*xyz)


        MaxwellianFluidModel(bx=bx, by=by, bz=bz,
                             protons={"charge": 1,
                                      "density": density,
                                      "vbulkx": vx, "vbulky": vy, "vbulkz": vz,
                                      "vthx": vthx, "vthy": vthy, "vthz": vthz,
                                      "nbr_part_per_cell": nbr_part_per_cell,
                                      "init": model_init})

        ElectronModel(closure="isothermal", Te=0.12)

        if timestamps is None:
            timestamps = np.arange(0, global_vars.sim.final_time + global_vars.sim.time_step, global_vars.sim.time_step)

        for quantity in ["E", "B"]:
            ElectromagDiagnostics(
                quantity=quantity,
                write_timestamps=timestamps,
                compute_timestamps=timestamps
            )

        for quantity in ["density", "bulkVelocity"]:
            FluidDiagnostics(
                quantity=quantity,
                write_timestamps=timestamps,
                compute_timestamps=timestamps
            )

        poplist = ["protons"]
        for pop in poplist:
            for quantity in ["density", "flux"]:
                FluidDiagnostics(quantity=quantity,
                                 write_timestamps=timestamps,
                                 compute_timestamps=timestamps,
                                 population_name=pop)

            for quantity in ['domain', 'levelGhost', 'patchGhost']:
                ParticleDiagnostics(quantity=quantity,
                                    compute_timestamps=timestamps,
                                    write_timestamps=timestamps,
                                    population_name=pop)

        Simulator(global_vars.sim).run()

        eb_hier = None
        if qty in ["e", "eb"]:
            eb_hier = hierarchy_from(h5_filename=diag_outputs+"/EM_E.h5", hier=eb_hier)
        if qty in ["b", "eb"]:
            eb_hier = hierarchy_from(h5_filename=diag_outputs+"/EM_B.h5", hier=eb_hier)
        if qty in ["e", "b", "eb"]:
            return eb_hier

        is_particle_type = qty == "particles" or qty == "particles_patch_ghost"

        if is_particle_type:
            particle_hier = None

        if qty == "particles":
            particle_hier = hierarchy_from(h5_filename=diag_outputs+"/ions_pop_protons_domain.h5")
            particle_hier = hierarchy_from(h5_filename=diag_outputs+"/ions_pop_protons_levelGhost.h5", hier=particle_hier)

        if is_particle_type:
            particle_hier = hierarchy_from(h5_filename=diag_outputs+"/ions_pop_protons_patchGhost.h5", hier=particle_hier)

        if qty == "particles":
            merge_particles(particle_hier)

        if is_particle_type:
            return particle_hier

        if qty == "moments":
            mom_hier = hierarchy_from(h5_filename=diag_outputs+"/ions_density.h5")
            mom_hier = hierarchy_from(h5_filename=diag_outputs+"/ions_bulkVelocity.h5", hier=mom_hier)
            mom_hier = hierarchy_from(h5_filename=diag_outputs+"/ions_pop_protons_density.h5", hier=mom_hier)
            mom_hier = hierarchy_from(h5_filename=diag_outputs+"/ions_pop_protons_flux.h5", hier=mom_hier)
            return mom_hier



    def _test_overlaped_fields_are_equal(self, time_step, time_step_nbr, datahier):
        check=0
        for time_step_idx in range(time_step_nbr + 1):
            coarsest_time =  time_step_idx * time_step

            for ilvl, overlaps in hierarchy_overlaps(datahier, coarsest_time).items():

                for overlap in overlaps:

                    pd1, pd2 = overlap["pdatas"]
                    box      = overlap["box"]
                    offsets  = overlap["offset"]

                    self.assertEqual(pd1.quantity, pd2.quantity)

                    if pd1.quantity == 'field':
                        check+=1

                        # we need to transform the AMR overlap box, which is thus
                        # (because AMR) common to both pd1 and pd2 into local index
                        # boxes that will allow to slice the data

                        # the patchData ghost box that serves as a reference box
                        # to transfrom AMR to local indexes first needs to be
                        # shifted by the overlap offset associated to it
                        # this is because the overlap box has been calculated from
                        # the intersection of possibly shifted patch data ghost boxes

                        loc_b1 = boxm.amr_to_local(box, boxm.shift(pd1.ghost_box, offsets[0]))
                        loc_b2 = boxm.amr_to_local(box, boxm.shift(pd2.ghost_box, offsets[1]))

                        data1 = pd1.dataset[:].reshape(pd1.ghost_box.shape + pd1.primal_directions())
                        data2 = pd2.dataset[:].reshape(pd2.ghost_box.shape + pd2.primal_directions())

                        if box.ndim == 1:
                            slice1 = data1[loc_b1.lower[0]:loc_b1.upper[0] + 1]
                            slice2 = data2[loc_b2.lower[0]:loc_b2.upper[0] + 1]

                        if box.ndim == 2:
                            slice1 = data1[loc_b1.lower[0]:loc_b1.upper[0] + 1, loc_b1.lower[1]:loc_b1.upper[1] + 1]
                            slice2 = data2[loc_b2.lower[0]:loc_b2.upper[0] + 1, loc_b2.lower[1]:loc_b2.upper[1] + 1]

                        try:
                            np.testing.assert_allclose(slice1, slice2, atol=1e-6)
                        except AssertionError as e:
                            print("error", coarsest_time, overlap)
                            raise e

        self.assertGreater(check, time_step_nbr)
        self.assertEqual(check % time_step_nbr, 0)




    def _test_overlapped_particledatas_have_identical_particles(self, ndim, interp_order, refinement_boxes, ppc=100, **kwargs):
        print("test_overlapped_particledatas_have_identical_particles, interporder : {}".format(interp_order))
        from copy import copy

        time_step_nbr=3
        time_step=0.001
        diag_outputs=f"phare_overlapped_particledatas_have_identical_particles/{ndim}/{self.ddt_test_id()}"
        datahier = self.getHierarchy(interp_order, refinement_boxes, "particles", diag_outputs=diag_outputs, ndim=ndim,
                                      time_step=time_step, time_step_nbr=time_step_nbr, nbr_part_per_cell=ppc, **kwargs)

        for time_step_idx in range(time_step_nbr + 1):
            coarsest_time =  time_step_idx * time_step

            overlaps = hierarchy_overlaps(datahier, coarsest_time)

            for ilvl, lvl in datahier.patch_levels.items():

                print("testing level {}".format(ilvl))
                for overlap in overlaps[ilvl]:
                    pd1, pd2 = overlap["pdatas"]
                    box      = overlap["box"]
                    offsets  = overlap["offset"]

                    self.assertEqual(pd1.quantity, pd2.quantity)

                    if "particles" in pd1.quantity:

                        # the following uses 'offset', we need to remember that offset
                        # is the quantity by which a patch has been moved to detect
                        # overlap with the other one.
                        # so shift by +offset when evaluating patch data in overlap box
                        # index space, and by -offset when we want to shift box indexes
                        # to the associated patch index space.

                        # overlap box must be shifted by -offset to select data in the patches
                        part1 = copy(pd1.dataset.select(boxm.shift(box, -np.asarray(offsets[0]))))
                        part2 = copy(pd2.dataset.select(boxm.shift(box, -np.asarray(offsets[1]))))

                        # periodic icell overlaps need shifting to be the same
                        part1.iCells = part1.iCells + offsets[0]
                        part2.iCells = part2.iCells + offsets[1]
<<<<<<< HEAD

                        self.assertEqual(part1, part2)
=======
>>>>>>> 4e773ae (better, needs C++ diag shaping)

                        self.assertEqual(part1, part2)


    def _test_L0_particle_number_conservation(self, ndim, ppc=100):
        cells=120
        time_step_nbr=10
        time_step=0.001

        n_particles = ppc * (cells ** ndim)
        for interp_order in [1, 2, 3]:
            diag_outputs=f"phare_L0_particle_number_conservation_{ndim}_{interp_order}"
            datahier = self.getHierarchy(interp_order, None, "particles", diag_outputs=diag_outputs,
                                      time_step=time_step, time_step_nbr=time_step_nbr,
                                      nbr_part_per_cell=ppc, cells=cells, ndim=ndim)
            for time_step_idx in range(time_step_nbr + 1):
                coarsest_time =  time_step_idx * time_step
                n_particles_at_t = 0
                for patch in datahier.level(0, coarsest_time).patches:
                    n_particles_at_t += patch.patch_datas["protons_particles"].dataset[patch.box].size()
                self.assertEqual(n_particles, n_particles_at_t)




    def _test_field_coarsening_via_subcycles(self, dim, interp_order, refinement_boxes, **kwargs):
        print("test_field_coarsening_via_subcycles for dim/interp : {}/{}".format(dim, interp_order))

        from tests.amr.data.field.coarsening.test_coarsen_field import coarsen
        from pyphare.pharein import global_vars

        time_step_nbr=3

        diag_outputs=f"phare_outputs_subcycle_coarsening_{self.ddt_test_id()}"
        datahier = self.getHierarchy(interp_order, refinement_boxes, "eb", cells=30,
                                      diag_outputs=diag_outputs, time_step=0.001,
                                      extra_diag_options={"fine_dump_lvl_max": 10},
                                      time_step_nbr=time_step_nbr, smallest_patch_size=5,
                                      largest_patch_size=30, ndim=dim, **kwargs)

        lvl_steps = global_vars.sim.level_time_steps
        print("LEVELSTEPS === ", lvl_steps)
        assert len(lvl_steps) > 1, "this test makes no sense with only 1 level"

        finestTimeStep = lvl_steps[-1]
        secondFinestTimeStep = lvl_steps[-2]

        finest_level_step_nbr = global_vars.sim.level_step_nbr[-1]
        uniqTimes = set([0])

        for step in range(1, finest_level_step_nbr + 1):
            checkTime = datahier.format_timestamp(finestTimeStep * step)
            self.assertIn(checkTime, datahier.times())
            uniqTimes.add(checkTime)

        self.assertEqual(len(uniqTimes), len(datahier.time_hier.items()))

        syncSteps = global_vars.sim.level_step_nbr[-2] # ignore finest subcycles

        # FIX THIS AFTER NO MORE REGRIDS
        #  SEE: https://github.com/PHAREHUB/PHARE/issues/400
        assert syncSteps % time_step_nbr == 0 # perfect division
        startStep = int(syncSteps / time_step_nbr) + 1 # skip first coarsest step due to issue 400

        def reshape_if(data, field): # could be a function on field patch data
            real_shape = field.ghost_box.shape + field.primal_directions()
            if (data.shape != real_shape).all():
                return data.reshape(real_shape)
            return data


        for step in range(startStep, syncSteps + 1):
            checkTime = datahier.format_timestamp(secondFinestTimeStep * step)
            self.assertIn(checkTime, datahier.times())
            nLevels = datahier.levelNbr(checkTime)
            self.assertGreaterEqual(nLevels, 2)
            levelNbrs = datahier.levelNbrs(checkTime)
            finestLevelNbr = max(levelNbrs)
            coarsestLevelNbr = min(levelNbrs)

            for coarseLevelNbr in range(coarsestLevelNbr, finestLevelNbr):
                coarsePatches = datahier.level(coarseLevelNbr, checkTime).patches
                finePatches = datahier.level(coarseLevelNbr + 1, checkTime).patches

                for coarsePatch in coarsePatches:
                    for finePatch in finePatches:
                        lvlOverlap = boxm.refine(coarsePatch.box, 2) * finePatch.box
                        if lvlOverlap is not None:
                            for EM in ["E", "B"]:
                                for xyz in ["x", "y", "z"]:
                                    qty = f"{EM}{xyz}"
                                    coarse_pd = coarsePatch.patch_datas[qty]
                                    fine_pd  = finePatch.patch_datas[qty]
                                    coarseBox = boxm.coarsen(lvlOverlap, 2)

                                    nGhosts = coarse_pd.layout.nbrGhostFor(qty)

                                    coarse_pdDataset = coarse_pd.dataset[:]
                                    fine_pdDataset = fine_pd.dataset[:]

                                    coarseOffset = coarseBox.lower - coarse_pd.layout.box.lower
                                    dataBox_lower = coarseOffset + nGhosts
                                    dataBox = Box(dataBox_lower, dataBox_lower + coarseBox.shape - 1)

                                    coarse_pdDataset = reshape_if(np.copy(coarse_pdDataset), coarse_pd)
                                    fine_pdDataset = reshape_if(np.copy(fine_pdDataset), fine_pd)

                                    afterCoarse = np.copy(coarse_pdDataset)

                                    # change values that should be updated to make failure obvious
                                    if dim == 1:
                                        afterCoarse[dataBox.lower[0] : dataBox.upper[0] + 1] = -144123
                                    if dim == 2:
                                        afterCoarse[dataBox.lower[0] : dataBox.upper[0] + 1,
                                                    dataBox.lower[1] : dataBox.upper[1] + 1] = -144123

                                    coarsen(qty, coarse_pd, fine_pd, coarseBox, fine_pdDataset, afterCoarse)
                                    np.testing.assert_allclose(coarse_pdDataset, afterCoarse, atol=1e-6)





    def _test_field_level_ghosts_via_subcycles_and_coarser_interpolation(self, ndim, interp_order, refinement_boxes):
        """
          This test runs two virtually identical simulations for one step.
            L0_datahier has no refined levels
            L0L1_datahier has one refined level

          This is done to compare L0 values that haven't received the coarsened values of L1 because there is no L1,
            to the level field ghost of L1 of L0L1_datahier

          The simulations are no longer comparable after the first advance, so this test cannot work beyond that.
        """

        print("test_field_coarsening_via_subcycles for dim/interp : {}/{}".format(ndim, interp_order))

        from tests.amr.data.field.refine.test_refine_field import refine_time_interpolate
        from pyphare.pharein import global_vars

        import random
        rando = random.randint(0, 1e10)

        def _getHier(diag_dir, boxes=[]):
            return self.getHierarchy(interp_order, boxes, "eb", cells=30,
                time_step_nbr=1, smallest_patch_size=5, largest_patch_size=30,
                diag_outputs=diag_dir, extra_diag_options={"fine_dump_lvl_max": 10}, time_step=0.001,
                model_init={"seed": rando}, ndim=ndim
            )

        def assert_time_in_hier(*ts):
            for t in ts:
                self.assertIn(L0L1_datahier.format_timestamp(t), L0L1_datahier.times())

        L0_datahier = _getHier(f"phare_lvl_ghost_interpolation_L0_diags_{self.ddt_test_id()}")
        L0L1_datahier = _getHier(
          f"phare_lvl_ghost_interpolation_L0L1_diags_{self.ddt_test_id()}", refinement_boxes
        )

        lvl_steps = global_vars.sim.level_time_steps
        assert len(lvl_steps) == 2, "this test is only configured for L0 -> L1 refinement comparisons"

        coarse_ilvl = 0
        fine_ilvl   = 1
        coarsest_time_before = 0 # init
        coarsest_time_after = coarsest_time_before + lvl_steps[coarse_ilvl]
        assert_time_in_hier(coarsest_time_before, coarsest_time_after)

        fine_subcycle_times = []
        for fine_subcycle in range(global_vars.sim.level_step_nbr[fine_ilvl] + 1):
            fine_subcycle_time   = coarsest_time_before + (lvl_steps[fine_ilvl] * fine_subcycle)
            assert_time_in_hier(fine_subcycle_time)
            fine_subcycle_times += [fine_subcycle_time]

        quantities = [f"{EM}{xyz}" for EM in ["E", "B"] for xyz in ["x", "y", "z"]]
        interpolated_fields = refine_time_interpolate(
          L0_datahier, quantities, coarse_ilvl, coarsest_time_before, coarsest_time_after, fine_subcycle_times
        )

        checks = 0
        for fine_subcycle_time in fine_subcycle_times:
            fine_level_qty_ghost_boxes = level_ghost_boxes(L0L1_datahier, quantities, fine_ilvl, fine_subcycle_time)
            for qty in quantities:
                for fine_level_ghost_box_data in fine_level_qty_ghost_boxes[qty]:
                    fine_subcycle_pd = fine_level_ghost_box_data["pdata"]
                    for fine_level_ghost_box in fine_level_ghost_box_data["boxes"]:

                        # trim the border level ghost nodes from the primal fields to ignore them in comparison checks
                        fine_level_ghost_boxes = fine_level_ghost_box - boxm.grow(fine_subcycle_pd.box, fine_subcycle_pd.primal_directions())
                        self.assertEqual(len(fine_level_ghost_boxes), 1) # should not be possibly > 1
                        self.assertEqual(fine_level_ghost_boxes[0].shape, fine_level_ghost_box.shape - fine_subcycle_pd.primal_directions())
                        fine_level_ghost_box = fine_level_ghost_boxes[0]

                        upper_dims = fine_level_ghost_box.lower > fine_subcycle_pd.box.upper
                        for refinedInterpolatedField in interpolated_fields[qty][fine_subcycle_time]:
                            lvlOverlap = refinedInterpolatedField.box * fine_level_ghost_box
                            if lvlOverlap is not None:

                                fine_ghostbox_data = fine_subcycle_pd[fine_level_ghost_box]
                                refinedInterpGhostBox_data = refinedInterpolatedField[fine_level_ghost_box]

                                fine_ds = fine_subcycle_pd.dataset
                                if fine_level_ghost_box.ndim == 1: # verify selecting start/end of L1 dataset from ghost box
                                    if upper_dims[0]:
                                        assert all(fine_ghostbox_data == fine_ds[-fine_ghostbox_data.shape[0]:])
                                    else:
                                        assert all(fine_ghostbox_data == fine_ds[:fine_ghostbox_data.shape[0]])

                                assert refinedInterpGhostBox_data.shape == fine_subcycle_pd.ghosts_nbr
                                assert fine_ghostbox_data.shape == fine_subcycle_pd.ghosts_nbr
                                np.testing.assert_allclose(fine_ghostbox_data, refinedInterpGhostBox_data, atol=1e-7)
                                checks += 1

        self.assertGreater(checks, len(refinement_boxes["L0"]) * len(quantities))




if __name__ == "__main__":
    unittest.main()
