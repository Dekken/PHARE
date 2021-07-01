#ifndef PHARE_SRC_AMR_FIELD_FIELD_VARIABLE_FILL_PATTERN_H
#define PHARE_SRC_AMR_FIELD_FIELD_VARIABLE_FILL_PATTERN_H

#include <cassert>

#include "SAMRAI/xfer/VariableFillPattern.h"

#include "core/utilities/types.h"
#include "core/utilities/mpi_utils.h"
#include "amr/data/field/field_geometry.h"

namespace PHARE::amr
{
class FieldFillPattern : public SAMRAI::xfer::VariableFillPattern
{
public:
    FieldFillPattern() {}
    virtual ~FieldFillPattern() {}

    std::shared_ptr<SAMRAI::hier::BoxOverlap>
    calculateOverlap(SAMRAI::hier::BoxGeometry const& dst_geometry,
                     SAMRAI::hier::BoxGeometry const& src_geometry,
                     SAMRAI::hier::Box const& dst_patch_box, SAMRAI::hier::Box const& src_mask,
                     SAMRAI::hier::Box const& fill_box, bool const overwrite_interior_,
                     SAMRAI::hier::Transformation const& transformation) const
    {
#ifndef DEBUG_CHECK_DIM_ASSERTIONS
        NULL_USE(dst_patch_box);
#endif
        TBOX_ASSERT_OBJDIM_EQUALITY2(dst_patch_box, src_mask);

        bool overwrite_interior = true; // replace func param
        assert(overwrite_interior_ == overwrite_interior);

        if (auto env = core::get_env("NO_OVER_INTER"); env and *env == "1")
            overwrite_interior = false;

        auto& dst_cast = dynamic_cast<AFieldGeometry const&>(dst_geometry);
        auto& src_cast = dynamic_cast<AFieldGeometry const&>(src_geometry);

        // for shared border node value sync
        if (auto env = core::get_env("PRIORITY"); env and *env == "1")
        {
            if (src_cast.patchBox.getGlobalId().getOwnerRank()
                != dst_cast.patchBox.getGlobalId().getOwnerRank())
                overwrite_interior
                    = src_cast.patchBox.getGlobalId() > dst_cast.patchBox.getGlobalId();
            // overwrite_interior
            //     = src_cast.patchBox.getLocalId() > dst_cast.patchBox.getLocalId();
        }

        std::cout << __FILE__ << " " << __LINE__ << " " << src_cast.patchBox << std::endl;
        std::cout << __FILE__ << " " << __LINE__ << " " << dst_cast.patchBox << std::endl;
        std::cout << __FILE__ << " " << __LINE__ << " " << overwrite_interior << std::endl;

        return dst_geometry.calculateOverlap(src_geometry, src_mask, fill_box, overwrite_interior,
                                             transformation);
    }

    std::string const& getPatternName() const { return s_name_id; }

private:
    FieldFillPattern(FieldFillPattern const&) = delete;
    FieldFillPattern& operator=(FieldFillPattern const&) = delete;

    static const std::string s_name_id; // = "GHOST_ONLY_FILL_PATTERN";

    SAMRAI::hier::IntVector const& getStencilWidth()
    {
        TBOX_ERROR("getStencilWidth() should not be\n"
                   << "called.  This pattern creates overlaps based on\n"
                   << "the BoxGeometry objects and is not restricted to a\n"
                   << "specific stencil.\n");

        /*
         * Dummy return value that will never get reached.
         */
        return SAMRAI::hier::IntVector::getZero(SAMRAI::tbox::Dimension(1));
    }

    /*
     *************************************************************************
     *
     * Compute BoxOverlap that specifies data to be filled by refinement
     * operator.
     *
     *************************************************************************
     */
    std::shared_ptr<SAMRAI::hier::BoxOverlap>
    computeFillBoxesOverlap(SAMRAI::hier::BoxContainer const& fill_boxes,
                            SAMRAI::hier::BoxContainer const& node_fill_boxes,
                            SAMRAI::hier::Box const& patch_box, SAMRAI::hier::Box const& data_box,
                            SAMRAI::hier::PatchDataFactory const& pdf) const
    {
        NULL_USE(node_fill_boxes);

        /*
         * For this (default) case, the overlap is simply the intersection of
         * fill_boxes and data_box.
         */
        SAMRAI::hier::Transformation transformation(
            SAMRAI::hier::IntVector::getZero(patch_box.getDim()));

        SAMRAI::hier::BoxContainer overlap_boxes(fill_boxes);
        overlap_boxes.intersectBoxes(data_box);

        return pdf.getBoxGeometry(patch_box)->setUpOverlap(overlap_boxes, transformation);
    }
};

} // namespace PHARE::amr

#endif /* PHARE_SRC_AMR_FIELD_FIELD_VARIABLE_FILL_PATTERN_H */
