#ifndef PHARE_TEST_TAG_STRATEGY_H
#define PHARE_TEST_TAG_STRATEGY_H

#include <SAMRAI/geom/CartesianPatchGeometry.h>
#include <SAMRAI/hier/Patch.h>
#include <SAMRAI/hier/RefineOperator.h>
#include <SAMRAI/mesh/StandardTagAndInitStrategy.h>
#include <SAMRAI/xfer/PatchLevelBorderFillPattern.h>
#include <SAMRAI/xfer/PatchLevelEnhancedFillPattern.h>
#include <SAMRAI/xfer/PatchLevelInteriorFillPattern.h>
#include <SAMRAI/xfer/RefineAlgorithm.h>

#include "core/data/grid/gridlayout.h"
#include "core/data/grid/gridlayoutdefs.h"
#include "amr/data/particles/particles_data.h"
#include "amr/data/particles/refine/particles_data_split.h"
#include "core/utilities/constants.h"
#include "core/utilities/point/point.h"


#include <map>
#include <string>
#include <type_traits>

using namespace PHARE::core;
using namespace PHARE::amr;

template<std::size_t dimension>
class TagStrategy : public SAMRAI::mesh::StandardTagAndInitStrategy
{
public:
    TagStrategy(std::map<std::string, int> const& dataToAllocate)
        : dataToAllocate_{dataToAllocate}
    {
        for (auto const& nameToIds : dataToAllocate_)
        {
            algorithm_.registerRefine(nameToIds.second, nameToIds.second, nameToIds.second,
                                      nullptr);
        }
    }


    virtual ~TagStrategy() = default;

    void initializeLevelData(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                             int const levelNumber, double const, bool const, bool const,
                             std::shared_ptr<SAMRAI::hier::PatchLevel> const& = std::shared_ptr<
                                 SAMRAI::hier::PatchLevel>(),
                             bool const allocateData = true) override
    {
        if (allocateData)
        {
            auto level = hierarchy->getPatchLevel(levelNumber);
            for (auto& patch : *level)
            {
                for (auto const& dataPair : dataToAllocate_)
                {
                    auto const& dataId = dataPair.second;
                    patch->allocatePatchData(dataId);
                }
            }
        }

        if (levelNumber == 0)
        {
            auto level = hierarchy->getPatchLevel(levelNumber);
            for (auto& patch : *level)
            {
                for (auto const& variablesId : dataToAllocate_)
                {
                    auto const& dataId = variablesId.second;
                    auto particlesData = std::dynamic_pointer_cast<
                        ParticlesData<ParticleArray<double, dimension>>>(
                        patch->getPatchData(dataId));

                    auto& interior = particlesData->domainParticles;

                    auto particlesBox = particlesData->getBox();

                    auto lowerZ = dimension == 3 ? particlesBox.lower(dirZ) : 0;
                    auto upperZ = dimension == 3 ? particlesBox.upper(dirZ) : 0;

                    auto lowerY = dimension == 2 ? particlesBox.lower(dirY) : 0;
                    auto upperY = dimension == 2 ? particlesBox.upper(dirY) : 0;

                    auto lowerX = particlesBox.lower(dirX);
                    auto upperX = particlesBox.upper(dirX);

                    for (auto iCellZ = lowerZ; iCellZ <= upperZ; ++iCellZ)
                    {
                        for (auto iCellY = lowerY; iCellY <= upperY; ++iCellY)
                        {
                            for (auto iCellX = lowerX; iCellX <= upperX; ++iCellX)
                            {
                                float middle = 0.5;
                                float delta  = 0.30;

                                Particle<double, dimension> particle;

                                particle.iCell = as_sized_array<dimension>(iCellX, iCellY, iCellZ);
                                particle.delta.fill(middle);
                                particle.weight = 1.;
                                particle.charge = 1.;
                                particle.v      = {{1.0, 0.0, 0.0}};

                                interior.push_back(particle);

                                particle.delta[dirX] = middle - delta;
                                interior.push_back(particle);

                                particle.delta[dirX] = middle + delta;
                                interior.push_back(particle);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            // create schedule
            /*if (splitType_ == ParticlesDataSplitType::coarseBoundary)
            {
                auto refineScheduleBorder = algorithm_.createSchedule(
                    std::make_shared<SAMRAI::xfer::PatchLevelBorderFillPattern>(),
                    hierarchy->getPatchLevel(levelNumber), nullptr, levelNumber - 1, hierarchy);

                refineScheduleBorder->fillData(0.);
            }
            else if (splitType_ == ParticlesDataSplitType::interior)
            {
                auto refineScheduleInterior = algorithm_.createSchedule(
                    std::make_shared<SAMRAI::xfer::PatchLevelInteriorFillPattern>(),
                    hierarchy->getPatchLevel(levelNumber), nullptr, levelNumber - 1, hierarchy);

                refineScheduleInterior->fillData(0.);
            }
            */
        }
    }




    void resetHierarchyConfiguration(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const&,
                                     int const, int const) override
    {
        // do nothing
    }



private:
    std::map<std::string, int> dataToAllocate_;
    SAMRAI::xfer::RefineAlgorithm algorithm_;
};

#endif
