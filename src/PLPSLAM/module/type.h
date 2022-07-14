/**
 * This file is part of Structure PLP-SLAM, originally from OpenVSLAM.
 *
 * Copyright 2022 DFKI (German Research Center for Artificial Intelligence)
 * Modified by Fangwen Shu <Fangwen.Shu@dfki.de>
 *
 * If you use this code, please cite the respective publications as
 * listed on the github repository.
 *
 * Structure PLP-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Structure PLP-SLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Structure PLP-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef PLPSLAM_MODULE_TYPE_H
#define PLPSLAM_MODULE_TYPE_H

#include <g2o/types/sim3/types_seven_dof_expmap.h>

namespace PLPSLAM
{
    namespace module
    {

        // Typedef here to avoid cross-references
        typedef std::map<data::keyframe *, g2o::Sim3, std::less<data::keyframe *>,
                         Eigen::aligned_allocator<std::pair<data::keyframe *const, g2o::Sim3>>>
            keyframe_Sim3_pairs_t;

        // A structure that combines the set of keyframes, the central keyframe, and the number of consecutive detections.
        struct keyframe_set
        {
            keyframe_set(const std::set<data::keyframe *> &keyfrm_set, data::keyframe *lead_keyfrm, const unsigned int continuity)
                : keyfrm_set_(keyfrm_set), lead_keyfrm_(lead_keyfrm), continuity_(continuity)
            {
            }

            std::set<data::keyframe *> keyfrm_set_;
            data::keyframe *lead_keyfrm_ = nullptr;
            unsigned int continuity_ = 0;

            bool intersection_is_empty(const std::set<data::keyframe *> &other_set) const
            {
                for (const auto &this_keyfrm : keyfrm_set_)
                {
                    if (static_cast<bool>(other_set.count(this_keyfrm)))
                    {
                        return false;
                    }
                }
                return true;
            }

            bool intersection_is_empty(const keyframe_set &other_set) const
            {
                return intersection_is_empty(other_set.keyfrm_set_);
            }
        };

        using keyframe_sets = eigen_alloc_vector<keyframe_set>;

    } // namespace module
} // namespace PLPSLAM

#endif // PLPSLAM_MODULE_TYPE_H
