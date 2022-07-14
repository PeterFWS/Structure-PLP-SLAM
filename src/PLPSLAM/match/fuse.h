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

#ifndef PLPSLAM_MATCH_FUSE_H
#define PLPSLAM_MATCH_FUSE_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/match/base.h"

namespace PLPSLAM
{

    namespace data
    {
        class keyframe;
        class landmark;

        // FW:
        class Line;
    } // namespace data

    namespace match
    {

        class fuse final : public base
        {
        public:
            explicit fuse(const float lowe_ratio = 0.6)
                : base(lowe_ratio, true) {}

            ~fuse() final = default;

            //! Reproject the 3D points (landmarks_to_check) to the keyframe and search for the ones that overlap with the 3D points observed in the keyframe.
            //! Duplicates are recorded in duplicated_lms_in_keyfrm with the same index
            //! Unlike replace_duplication (), do not replace in the function
            //! NOTE: landmarks_to_check.size () == duplicated_lms_in_keyfrm.size ()
            unsigned int detect_duplication(data::keyframe *keyfrm, const Mat44_t &Sim3_cw, const std::vector<data::landmark *> &landmarks_to_check,
                                            const float margin, std::vector<data::landmark *> &duplicated_lms_in_keyfrm);

            //! Reproject the 3D points (landmarks_to_check) to the keyframe and search for the ones that overlap with the 3D points observed in the keyframe.
            //! Select and replace more reliable 3D points for duplicates
            //! Unlike detect_duplication (), replace in a function
            template <typename T>
            unsigned int replace_duplication(data::keyframe *keyfrm, const T &landmarks_to_check, const float margin = 3.0);

            // FW: similar function for 3D line
            template <typename T>
            unsigned int replace_duplication_line(data::keyframe *keyfrm, const T &landmarks_to_check, const float margin = 3.0);
        };

    } // namespace match
} // namespace PLPSLAM

#endif // PLPSLAM_MATCH_FUSE_H
