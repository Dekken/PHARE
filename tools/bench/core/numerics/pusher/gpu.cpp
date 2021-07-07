
#include <algorithm>

#include "kul/gpu.hpp"
#include "kul/gpu/asio.hpp"


#include "pusher_bench.h"
#include "benchmark/benchmark.h"

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


template<typename Particle>
__global__ void push_particle(Particle* particle, int offset)
{
    auto i = kul::gpu::asio::idx() + offset;
    // a[i].f0 = a[i].f0 + 1;
}

template<std::size_t dim, std::size_t interp>
void push(benchmark::State& state)
{
    constexpr std::uint32_t cells = 65;
    constexpr std::uint32_t parts = 1e8;

    using PHARE_Types       = PHARE::core::PHARE_Types<dim, interp>;
    using Interpolator      = PHARE::core::Interpolator<dim, interp>;
    using BoundaryCondition = PHARE::core::BoundaryCondition<dim, interp>;
    using Ions_t            = typename PHARE_Types::Ions_t;
    using Electromag_t      = typename PHARE_Types::Electromag_t;
    using GridLayout_t      = typename PHARE_Types::GridLayout_t;
    using ParticleArray     = typename Ions_t::particle_array_type;
    using PartIterator      = typename ParticleArray::iterator;


    using PetrovPusher_t = PHARE::core::PetrovPusher<dim, PartIterator, Electromag_t, Interpolator,
                                                     BoundaryCondition, GridLayout_t>;

    Interpolator interpolator;
    ParticleArray domainParticles{parts, particle<dim>(/*icell =*/34)};

    auto range    = PHARE::core::makeRange(domainParticles);
    auto meshSize = PHARE::core::ConstArray<double, dim>(1.0 / cells);
    auto nCells   = PHARE::core::ConstArray<std::uint32_t, dim>(cells);
    auto origin   = PHARE::core::Point<double, dim>{PHARE::core::ConstArray<double, dim>(0)};

    GridLayout_t layout{meshSize, nCells, origin};

    PHARE::core::bench::Electromag<GridLayout_t, VecField<dim>> electromag{layout};

    PetrovPusher_t pusher;
    pusher.setMeshAndTimeStep(layout.meshSize(), .001);

    while (state.KeepRunning())
    {
        pusher.move(
            /*ParticleRange const&*/ range,
            /*ParticleRange&*/ range,
            /*Electromag const&*/ electromag,
            /*double mass*/ 1,
            /*Interpolator&*/ interpolator,
            /*ParticleSelector const&*/ [](auto const& /*part*/) { return true; },
            /*GridLayout const&*/ layout);
    }
}
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

// BENCHMARK_TEMPLATE(push, /*dim=*/1, /*interp=*/1)->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE(push, /*dim=*/1, /*interp=*/2)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(push, /*dim=*/1, /*interp=*/3)->Unit(benchmark::kMicrosecond);

// BENCHMARK_TEMPLATE(push, /*dim=*/2, /*interp=*/1)->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE(push, /*dim=*/2, /*interp=*/2)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(push, /*dim=*/2, /*interp=*/3)->Unit(benchmark::kMicrosecond);

// BENCHMARK_TEMPLATE(push, /*dim=*/3, /*interp=*/1)->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE(push, /*dim=*/3, /*interp=*/2)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(push, /*dim=*/3, /*interp=*/3)->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv, char** envp)
{
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
}
