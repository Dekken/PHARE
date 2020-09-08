#ifndef PHARE_PARTICLE_INITIALIZER_FACTORY_H
#define PHARE_PARTICLE_INITIALIZER_FACTORY_H


#include "core/utilities/types.h"
#include "initializer/data_provider.h"
#include "maxwellian_particle_initializer.h"
#include "particle_initializer.h"

#include <memory>

namespace PHARE
{
namespace core
{
    template<typename ParticleArray, typename GridLayout>
    class ParticleInitializerFactory
    {
        using ParticleInitializerT      = ParticleInitializer<ParticleArray, GridLayout>;
        static constexpr auto dimension = GridLayout::dimension;


    public:
        template<typename FunctionType>
        static std::unique_ptr<ParticleInitializerT> createType(initializer::PHAREDict& dict)
        {
            auto initializerName = dict["name"].template to<std::string>();

            if (initializerName == "maxwellian")
            {
                auto& density = dict["density"].template to<FunctionType>();

                auto& bulkVelx = dict["bulk_velocity_x"].template to<FunctionType>();

                auto& bulkVely = dict["bulk_velocity_y"].template to<FunctionType>();

                auto& bulkVelz = dict["bulk_velocity_z"].template to<FunctionType>();

                auto& vthx = dict["thermal_velocity_x"].template to<FunctionType>();

                auto& vthy = dict["thermal_velocity_y"].template to<FunctionType>();

                auto& vthz = dict["thermal_velocity_z"].template to<FunctionType>();

                auto charge = dict["charge"].template to<double>();

                auto nbrPartPerCell = dict["nbr_part_per_cell"].template to<int>();

                auto basisName = dict["basis"].template to<std::string>();

                std::array<FunctionType, 3> v = {bulkVelx, bulkVely, bulkVelz};

                std::array<FunctionType, 3> vth = {vthx, vthy, vthz};

                std::optional<std::size_t> seed;
                if (dict.contains("init") && dict["init"].contains("seed"))
                    seed = dict["init"]["seed"].template to<std::optional<std::size_t>>();

                if (basisName == "cartesian")
                {
                    return std::make_unique<
                        MaxwellianParticleInitializer<ParticleArray, GridLayout, FunctionType>>(
                        density, v, vth, charge, nbrPartPerCell, seed);
                }
                else if (basisName == "magnetic")
                {
                    [[maybe_unused]] Basis basis = Basis::Magnetic;
                    [[maybe_unused]] auto& bx
                        = dict["magnetic_x"].template to<initializer::ScalarFunction<dimension>>();
                    [[maybe_unused]] auto& by
                        = dict["magnetic_x"].template to<initializer::ScalarFunction<dimension>>();
                    [[maybe_unused]] auto& bz
                        = dict["magnetic_x"].template to<initializer::ScalarFunction<dimension>>();

                    return std::make_unique<
                        MaxwellianParticleInitializer<ParticleArray, GridLayout, FunctionType>>(
                        density, v, vth, charge, nbrPartPerCell, seed);
                }
            }
            // TODO throw?
            return nullptr;
        }

        static std::unique_ptr<ParticleInitializerT> create(initializer::PHAREDict& dict)
        {
            if (dict.contains("fn_type") and dict["fn_type"].template to<std::string>() == "vector")
                return createType<initializer::VectorFunction<dimension>>(dict);
            return createType<initializer::ScalarFunction<dimension>>(dict);
        }
    };

} // namespace core

} // namespace PHARE

#endif
