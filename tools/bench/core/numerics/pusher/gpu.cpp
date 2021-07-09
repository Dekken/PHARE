
#include <algorithm>

#include "kul/gpu.hpp"
#include "kul/gpu/asio.hpp"

#include "pusher_bench.h"
#include "benchmark/benchmark.h"

#include "core/numerics/pusher/boris.h"
#include "core/numerics/pusher/granov.h"
#include "tests/gpu/gpu.hpp"

using namespace PHARE::core::bench;

static constexpr std::uint32_t BATCHES  = 4;
static constexpr std::uint32_t NUM      = 1024 * 1024 * BATCHES;
static constexpr std::uint32_t TP_BLOCK = 256;
static constexpr double TIMESTEP        = .00001;


template<typename PHARE_TYPES>
class GPU_setup
{
    using Hierarchy_t   = typename PHARE_TYPES::hierarchy_t;
    using HybridModel_t = typename PHARE_TYPES::HybridModel_t;
    using HybridState_t = typename HybridModel_t::State_t;

    auto static fill_states(GPU_setup& self)
    {
        std::vector<PHARE::gpu::PatchState<PHARE_TYPES>> states;
        PHARE::amr::visitHierarchy<typename PHARE_TYPES::GridLayout_t>(
            self.hierarchy, *self.hybridModel.resourcesManager,
            [&](auto& gridLayout, std::string, size_t) {
                states.emplace_back(gridLayout, self.state);
            },
            self.topLvl, self.topLvl + 1, self.hybridModel);
        return states;
    }

public:
    static constexpr auto dim           = PHARE_TYPES::dimension;
    static constexpr auto interp        = PHARE_TYPES::interp_order;
    static auto constexpr nbRefineParts = PHARE_TYPES::nbRefinedPart;

    GPU_setup(std::string job_id)
        : sim{job_id}
    {
    }

    SimulatorTestParam<dim, interp, nbRefineParts> sim;
    Hierarchy_t& hierarchy{*sim.hierarchy};
    HybridModel_t& hybridModel{*sim.getHybridModel()};
    HybridState_t& state{hybridModel.state};
    int topLvl{hierarchy.getNumberOfLevels() - 1};

    std::vector<PHARE::gpu::PatchState<PHARE_TYPES>> states{fill_states(*this)};
    PHARE::gpu::ParticlePatchState<PHARE_TYPES> packer{states};
};


template<typename PHARE_TYPES>
__global__ void gpu_particles_in(PHARE::gpu::PatchStatePerParticle<PHARE_TYPES, true>* ppsp)
{
    auto i = kul::gpu::idx();
    if (i >= ppsp->n_particles())
        return;

    auto patchStateIDX = (*ppsp)[i];
    auto& layout       = ppsp->gridLayouts->layouts[patchStateIDX];
    auto electromag    = ppsp->electromags->electromag(patchStateIDX);
    auto& particle     = ppsp->particles->particles[i];

    static constexpr auto dim    = PHARE_TYPES::dimension;
    static constexpr auto interp = PHARE_TYPES::interp_order;

    using Interpolator      = PHARE::core::Interpolator<dim, interp>;
    using BoundaryCondition = PHARE::core::BoundaryCondition<dim, interp>;
    using Ions_t            = typename PHARE_TYPES::Ions_t;
    // using Electromag_t = typename PHARE_TYPES::Electromag_t;
    using Electromag_t = decltype(electromag);
    // using GridLayout_t = typename PHARE_TYPES::GridLayout_t;
    using GridLayout_t = std::decay_t<decltype(layout)>;

    using ParticleArray = typename Ions_t::particle_array_type;
    using PartIterator  = typename ParticleArray::iterator;

    using Pusher_t = PHARE::core::GranovPusher<dim, PartIterator, Electromag_t, Interpolator,
                                               BoundaryCondition, GridLayout_t>;

    Interpolator interpolator;
    Pusher_t pusher;
    pusher.setMeshAndTimeStep(layout.meshSize(), TIMESTEP);
    pusher.accelerate_setup(1);
    pusher.move_in_place(
        /*Particle_t&*/ particle,                                              //
        /*Electromag const&*/ electromag,                                      //
        /*Interpolator&*/ interpolator,                                        //
        /*ParticleSelector const&*/ [](auto const& /*part*/) { return true; }, //
        /*GridLayout const&*/ layout                                           //
    );
}




template<std::size_t dim = 1, std::size_t interp = 1, std::size_t nbRefineParts = 2>
void push_sync(benchmark::State& state /*std::string job_id*/)
{
    static constexpr size_t X = 1024, Y = 1024, Z = 40;        // 40;
    static constexpr size_t TPB_X = 16, TPB_Y = 16, TPB_Z = 4; // 4;
    static constexpr size_t MAX_PARTICLES = X * Y * Z;

    std::string job_id = "job_1d";

    using PHARE_TYPES       = PHARE::PHARE_Types<dim, interp, nbRefineParts /*, Float*/>;
    using Interpolator      = PHARE::core::Interpolator<dim, interp>;
    using BoundaryCondition = PHARE::core::BoundaryCondition<dim, interp>;
    using Ions_t            = typename PHARE_TYPES::Ions_t;
    using Electromag_t      = typename PHARE_TYPES::Electromag_t;
    using GridLayout_t      = typename PHARE_TYPES::GridLayout_t;
    using ParticleArray     = typename Ions_t::particle_array_type;
    using PartIterator      = typename ParticleArray::iterator;


    KLOG(INF) << "MAX_PARTICLES: " << MAX_PARTICLES << " "
              << (MAX_PARTICLES * sizeof(PHARE::gpu::PatchStatePerParticle<PHARE_TYPES, true>)
                  + MAX_PARTICLES * sizeof(PHARE::core::Particle<dim>));


    using BorisPusher_t = PHARE::core::BorisPusher<dim, PartIterator, Electromag_t, Interpolator,
                                                   BoundaryCondition, GridLayout_t>;

    for (auto _ : state)
    {
        GPU_setup<PHARE_TYPES> setup{job_id};
        KLOG(INF) << "GPU PARTICLES: " << setup.packer.n_particles;

        KLOG(INF) << " CPU particle 0 delta before " << (*setup.states[0].ions[0])[0].delta[0];

        kul::gpu::Launcher{X, Y, Z, TPB_X, TPB_Y, TPB_Z}(gpu_particles_in<PHARE_TYPES>,
                                                         setup.packer());

        auto gpu_particles = setup.packer.particles->particles();
        KLOG(INF) << " GPU particle 0 delta after  " << gpu_particles[0].delta[0];

        Interpolator interpolator;

        PHARE::amr::visitHierarchy<GridLayout_t>(
            setup.hierarchy, *setup.hybridModel.resourcesManager,
            [&](auto& layout, std::string patchID, size_t) {
                if (patchID == "0#0")
                    for (auto& pop : setup.state.ions)
                    {
                        BorisPusher_t pusher;
                        pusher.setMeshAndTimeStep(layout.meshSize(), TIMESTEP);

                        auto range = PHARE::core::makeRange(pop.domainParticles());
                        pusher.move(
                            /*ParticleRange const&*/ range,
                            /*ParticleRange&*/ range,
                            /*Electromag const&*/ setup.state.electromag,
                            /*double mass*/ 1,
                            /*Interpolator&*/ interpolator,
                            /*ParticleSelector const&*/ [](auto const& /*part*/) { return true; },
                            /*GridLayout const&*/ layout);

                        KLOG(INF) << "CPU particle 0 delta after  " << (*range.begin()).delta[0];
                        break;
                    }
            },
            setup.topLvl, setup.topLvl + 1, setup.hybridModel);

        if (PHARE::core::mpi::any(PHARE::core::Errors::instance().any()))
            throw std::runtime_error("errors");
    }
}


// BENCHMARK_TEMPLATE(push_async, /*dim=*/1, /*interp=*/3)->Unit(benchmark::kMicrosecond);

// BENCHMARK_TEMPLATE(push, /*dim=*/1, /*interp=*/1)->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE(push, /*dim=*/1, /*interp=*/2)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(push_sync, /*dim=*/1, /*interp=*/3)->Unit(benchmark::kMicrosecond);

// BENCHMARK_TEMPLATE(push, /*dim=*/2, /*interp=*/1)->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE(push, /*dim=*/2, /*interp=*/2)->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE(push_sync, /*dim=*/2, /*interp=*/3)->Unit(benchmark::kMicrosecond);

// BENCHMARK_TEMPLATE(push, /*dim=*/3, /*interp=*/1)->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE(push, /*dim=*/3, /*interp=*/2)->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE(push_sync, /*dim=*/3, /*interp=*/3)->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv, char** envp)
{
    KLOG(INF);
    PHARE::SamraiLifeCycle samsam(argc, argv);

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

    KLOG(INF);
}
