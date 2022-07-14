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

#ifndef PLPSLAM_SOLVE_UTIL_H
#define PLPSLAM_SOLVE_UTIL_H

#include "PLPSLAM/type.h"

#include <vector>

#include <opencv2/core.hpp>

namespace PLPSLAM
{
    namespace solve
    {

        void normalize(const std::vector<cv::KeyPoint> &keypts, std::vector<cv::Point2f> &normalized_pts, Mat33_t &transform);

    } // namespace solve
} // namespace PLPSLAM

#endif // PLPSLAM_SOLVE_UTIL_H
