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

#include "PLPSLAM/solve/common.h"

namespace PLPSLAM
{
    namespace solve
    {

        void normalize(const std::vector<cv::KeyPoint> &keypts, std::vector<cv::Point2f> &normalized_pts, Mat33_t &transform)
        {
            float mean_x = 0;
            float mean_y = 0;
            const auto num_keypts = keypts.size();

            normalized_pts.resize(num_keypts);

            for (const auto &keypt : keypts)
            {
                mean_x += keypt.pt.x;
                mean_y += keypt.pt.y;
            }
            mean_x = mean_x / num_keypts;
            mean_y = mean_y / num_keypts;

            float mean_l1_dev_x = 0;
            float mean_l1_dev_y = 0;

            for (unsigned int index = 0; index < num_keypts; ++index)
            {
                normalized_pts.at(index).x = keypts.at(index).pt.x - mean_x;
                normalized_pts.at(index).y = keypts.at(index).pt.y - mean_y;

                mean_l1_dev_x += std::abs(normalized_pts.at(index).x);
                mean_l1_dev_y += std::abs(normalized_pts.at(index).y);
            }

            mean_l1_dev_x = mean_l1_dev_x / num_keypts;
            mean_l1_dev_y = mean_l1_dev_y / num_keypts;

            const float mean_l1_dev_x_inv = static_cast<float>(1.0) / mean_l1_dev_x;
            const float mean_l1_dev_y_inv = static_cast<float>(1.0) / mean_l1_dev_y;

            for (auto &normalized_pt : normalized_pts)
            {
                normalized_pt.x *= mean_l1_dev_x_inv;
                normalized_pt.y *= mean_l1_dev_y_inv;
            }

            transform = Mat33_t::Identity();
            transform(0, 0) = mean_l1_dev_x_inv;
            transform(1, 1) = mean_l1_dev_y_inv;
            transform(0, 2) = -mean_x * mean_l1_dev_x_inv;
            transform(1, 2) = -mean_y * mean_l1_dev_y_inv;
        }

    } // namespace solve
} // namespace PLPSLAM
