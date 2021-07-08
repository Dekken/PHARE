
#include <algorithm>

#include "kul/gpu.hpp"
#include "kul/gpu/asio.hpp"

#include "pusher_bench.h"
#include "benchmark/benchmark.h"

#include "core/numerics/pusher/granov.h"
#include "tests/gpu/gpu.hpp"


static constexpr std::uint32_t GIGS = 4;
using namespace PHARE::core::bench;

template<typename Type>
auto alloc_type_around_size_gb(std::size_t gigabytes)
{
}

static constexpr std::uint32_t BATCHES  = 4;
static constexpr std::uint32_t NUM      = 1024 * 1024 * BATCHES;
static constexpr std::uint32_t TP_BLOCK = 256;

/*
template<typename Particle>
__global__ void push_particle(Particle* particle, int offset)
{
    auto i = kul::gpu::asio::idx() + offset;
    // a[i].f0 = a[i].f0 + 1;
}*/



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
        , hierarchy{*sim.hierarchy}
        , hybridModel{*sim.getHybridModel()}
        , state{hybridModel.state}
        , topLvl{hierarchy.getNumberOfLevels() - 1}
        , states{fill_states(*this)}
        , packer{states}

    {
        KLOG(INF) << hierarchy.domainBox()[0];
    }

    SimulatorTestParam<dim, interp, nbRefineParts> sim;
    Hierarchy_t& hierarchy;
    HybridModel_t& hybridModel;
    HybridState_t& state;
    int topLvl;

    std::vector<PHARE::gpu::PatchState<PHARE_TYPES>> states;
    PHARE::gpu::ParticlePatchState<PHARE_TYPES> packer;
};


template<typename PHARE_TYPES>
__global__ void gpu_particles_in(PHARE::gpu::PatchStatePerParticle<PHARE_TYPES, true>* ppsp)
{
    auto i = kul::gpu::idx();
    if (i >= ppsp->n_particles())
        return;

    auto patchStateIDX = (*ppsp)[i];
    auto layout        = ppsp->gridLayouts->gridLayout(patchStateIDX);
    auto electromag    = ppsp->electromags->electromag(patchStateIDX);
    auto& particle     = ppsp->particles->particles[i];

    static constexpr auto dim    = PHARE_TYPES::dimension;
    static constexpr auto interp = PHARE_TYPES::interp_order;

    using Interpolator      = PHARE::core::Interpolator<dim, interp>;
    using BoundaryCondition = PHARE::core::BoundaryCondition<dim, interp>;
    using Ions_t            = typename PHARE_TYPES::Ions_t;
    // using Electromag_t = typename PHARE_TYPES::Electromag_t;
    using Electromag_t = decltype(electromag);
    // using GridLayout_t  = typename PHARE_TYPES::GridLayout_t;
    using GridLayout_t = decltype(layout);
    ;
    using ParticleArray = typename Ions_t::particle_array_type;
    using PartIterator  = typename ParticleArray::iterator;

    using Pusher_t = PHARE::core::GranovPusher<dim, PartIterator, Electromag_t, Interpolator,
                                               BoundaryCondition, GridLayout_t>;

    Interpolator interpolator;
    Pusher_t pusher;
    pusher.accelerate_setup(1);
    pusher.setMeshAndTimeStep(layout.meshSize(), .001);
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
    static constexpr size_t X = 1024, Y = 1024, Z = 1;         // 40;
    static constexpr size_t TPB_X = 16, TPB_Y = 16, TPB_Z = 1; // 4;
    static constexpr size_t MAX_PARTICLES = X * Y * Z;

    KLOG(INF) << "MAX_PARTICLES: " << MAX_PARTICLES;

    std::string job_id = "job_1d";

    using PHARE_TYPES = PHARE::PHARE_Types<dim, interp, nbRefineParts /*, Float*/>;

    for (auto _ : state)
    {
        GPU_setup<PHARE_TYPES> setup{job_id};
        KLOG(INF) << "GPU PARTICLES: " << setup.packer.n_particles;

        KLOG(INF) << "particle 0 delta before";
        KLOG(INF) << (*setup.states[0].ions[0])[0].delta[0];

        kul::gpu::Launcher{X, Y, Z, TPB_X, TPB_Y, TPB_Z}(gpu_particles_in<PHARE_TYPES>,
                                                         setup.packer());

        KLOG(INF) << "particle 0 delta after";
        KLOG(INF) << setup.packer.particles->particles()[0].delta[0];
    }
}


// template<std::size_t dim, std::size_t interp>
// void push(benchmark::State& state)
// {
//     while (state.KeepRunning())

// constexpr std::uint32_t cells = 65;
// constexpr std::uint32_t parts = 1e8;

// using PHARE_Types       = PHARE::core::PHARE_Types<dim, interp>;
// using Interpolator      = PHARE::core::Interpolator<dim, interp>;
// using BoundaryCondition = PHARE::core::BoundaryCondition<dim, interp>;
// using Ions_t            = typename PHARE_Types::Ions_t;
// using Electromag_t      = typename PHARE_Types::Electromag_t;
// using GridLayout_t      = typename PHARE_Types::GridLayout_t;
// using ParticleArray     = typename Ions_t::particle_array_type;
// using PartIterator      = typename ParticleArray::iterator;


// using Pusher_t = PHARE::core::PavlovPusher<dim, PartIterator, Electromag_t, Interpolator,
//                                            BoundaryCondition, GridLayout_t>;

// Interpolator interpolator;
// ParticleArray domainParticles{parts, particle<dim>(/*icell =*/34)};

// auto range    = PHARE::core::makeRange(domainParticles);
// auto meshSize = PHARE::core::ConstArray<double, dim>(1.0 / cells);
// auto nCells   = PHARE::core::ConstArray<std::uint32_t, dim>(cells);
// auto origin   = PHARE::core::Point<double, dim>{PHARE::core::ConstArray<double, dim>(0)};

// GridLayout_t layout{meshSize, nCells, origin};

// PHARE::core::bench::Electromag<GridLayout_t, VecField<dim>> electromag{layout};

// PetrovPusher_t pusher;
// pusher.setMeshAndTimeStep(layout.meshSize(), .001);

// while (state.KeepRunning())
// {
//     pusher.move(
//         /*ParticleRange const&*/ range,
//         /*ParticleRange&*/ range,
//         /*Electromag const&*/ electromag,
//         /*double mass*/ 1,
//         /*Interpolator&*/ interpolator,
//         /*ParticleSelector const&*/ [](auto const& /*part*/) { return true; },
//         /*GridLayout const&*/ layout);
// }
// }
/*
std::uint32_t test_single()
{
    kul::gpu::HostArray<A, NUM> a;
    for (std::uint32_t i = 0; i < NUM; ++i)
        a[i].f0 = i;
    kul::gpu::asio::Batch batch{BATCHES, a};
    kul::gpu::asio::Launcher{TP_BLOCK}(single, batch).async_back();
    for (std::size_t i = 0; i < batch.streams.size(); ++i)
    {
        auto offset    = i * batch.streamSize;
        auto copy_back = batch[i];
        for (std::uint32_t j = 0; j < batch.streamSize; ++j)
            if (copy_back[j].f0 != a[j + offset].f0 + 1)
                return 1;
    }
    return 0;
}*/

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
