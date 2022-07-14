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

#ifndef PLPSLAM_MATCH_BOW_TREE_H
#define PLPSLAM_MATCH_BOW_TREE_H

#include "PLPSLAM/match/base.h"

namespace PLPSLAM
{

    namespace data
    {
        class frame;
        class keyframe;
        class landmark;
    } // namespace data

    namespace match
    {

        class bow_tree final : public base
        {
        public:
            explicit bow_tree(const float lowe_ratio = 0.6, const bool check_orientation = true)
                : base(lowe_ratio, check_orientation) {}

            ~bow_tree() final = default;

            //! Find the correspondence between the feature points observed in the frame and the feature points observed in the keyframe,
            //  and obtain the correspondence information between the feature points in the frame and the 3D points based on it.
            //! matched_lms_in_frm stores the 3D points (observed in the keyframe) corresponding to each feature point in the frame.
            //! NOTE: matched_lms_in_frm.size () matches the feature score of frame
            unsigned int match_frame_and_keyframe(data::keyframe *keyfrm, data::frame &frm, std::vector<data::landmark *> &matched_lms_in_frm) const;

            //! Find the correspondence between the feature points observed by keyframe1 and the feature points observed by keyframe2,
            //  and obtain the correspondence information between the feature points of keyframe1 and the 3D points based on it.
            //! matched_lms_in_keyfrm_1 stores the 3D points (observed in keyframe2) corresponding to each feature point in keyframe1.
            //! NOTE: matched_lms_in_keyfrm_1.size () matches the feature score of keyframe1
            unsigned int match_keyframes(data::keyframe *keyfrm_1, data::keyframe *keyfrm_2, std::vector<data::landmark *> &matched_lms_in_keyfrm_1) const;
        };

    } // namespace match
} // namespace PLPSLAM

#endif // PLPSLAM_MATCH_BOW_TREE_H
