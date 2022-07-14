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

#ifndef PLPSLAM_UTIL_CONVERTER_H
#define PLPSLAM_UTIL_CONVERTER_H

#include "PLPSLAM/type.h"

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

namespace PLPSLAM
{
    namespace util
    {

        class converter
        {
        public:
            //! descriptor vector
            static std::vector<cv::Mat> to_desc_vec(const cv::Mat &desc);

            //! to SE3 of g2o
            static g2o::SE3Quat to_g2o_SE3(const Mat44_t &cam_pose);

            //! to Eigen::Mat/Vec
            static Mat44_t to_eigen_mat(const g2o::SE3Quat &g2o_SE3);
            static Mat44_t to_eigen_mat(const g2o::Sim3 &g2o_Sim3);
            static Mat44_t to_eigen_cam_pose(const Mat33_t &rot, const Vec3_t &trans);

            //! from/to angle axis
            static Vec3_t to_angle_axis(const Mat33_t &rot_mat);
            static Mat33_t to_rot_mat(const Vec3_t &angle_axis);

            //! to homogeneous coordinates
            template <typename T>
            static Vec3_t to_homogeneous(const cv::Point_<T> &pt)
            {
                return Vec3_t{pt.x, pt.y, 1.0};
            }

            //! to skew symmetric matrix
            static Mat33_t to_skew_symmetric_mat(const Vec3_t &vec);
        };

    } // namespace util
} // namespace PLPSLAM

#endif // PLPSLAM_UTIL_CONVERTER_H
