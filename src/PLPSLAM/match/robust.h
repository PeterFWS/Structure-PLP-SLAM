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

#ifndef PLPSLAM_MATCH_ROBUST_H
#define PLPSLAM_MATCH_ROBUST_H

#include "PLPSLAM/type.h"
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

        class robust final : public base
        {
        public:
            explicit robust(const float lowe_ratio, const bool check_orientation)
                : base(lowe_ratio, check_orientation) {}

            ~robust() final = default;

            unsigned int match_for_triangulation(data::keyframe *keyfrm_1, data::keyframe *keyfrm_2, const Mat33_t &E_12,
                                                 std::vector<std::pair<unsigned int, unsigned int>> &matched_idx_pairs);

            unsigned int match_frame_and_keyframe(data::frame &frm, data::keyframe *keyfrm,
                                                  std::vector<data::landmark *> &matched_lms_in_frm);

            unsigned int brute_force_match(data::frame &frm, data::keyframe *keyfrm, std::vector<std::pair<int, int>> &matches);

        private:
            bool check_epipolar_constraint(const Vec3_t &bearing_1, const Vec3_t &bearing_2,
                                           const Mat33_t &E_12, const float bearing_1_scale_factor = 1.0);
        };

    } // namespace match
} // namespace PLPSLAM

#endif // PLPSLAM_MATCH_ROBUST_H
