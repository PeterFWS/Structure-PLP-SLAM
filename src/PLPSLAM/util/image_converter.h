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

#ifndef PLPSLAM_UTIL_IMAGE_CONVERTER_H
#define PLPSLAM_UTIL_IMAGE_CONVERTER_H

#include "PLPSLAM/camera/base.h"

#include <opencv2/core.hpp>

namespace PLPSLAM
{
    namespace util
    {

        void convert_to_grayscale(cv::Mat &img, const camera::color_order_t in_color_order);

        void convert_to_true_depth(cv::Mat &img, const double depthmap_factor);

        void equalize_histogram(cv::Mat &img);

    } // namespace util
} // namespace PLPSLAM

#endif // PLPSLAM_UTIL_IMAGE_CONVERTER_H
