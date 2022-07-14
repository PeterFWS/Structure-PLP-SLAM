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

#ifndef PLPSLAM_MATCH_AREA_H
#define PLPSLAM_MATCH_AREA_H

#include "PLPSLAM/match/base.h"

namespace PLPSLAM
{

    namespace data
    {
        class frame;
    } // namespace data

    namespace match
    {

        class area final : public base
        {
        public:
            area(const float lowe_ratio, const bool check_orientation)
                : base(lowe_ratio, check_orientation) {}

            ~area() final = default;

            unsigned int match_in_consistent_area(data::frame &frm_1, data::frame &frm_2, std::vector<cv::Point2f> &prev_matched_pts,
                                                  std::vector<int> &matched_indices_2_in_frm_1, int margin = 10);
        };

    } // namespace match
} // namespace PLPSLAM

#endif // PLPSLAM_MATCH_AREA_H
