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

#include "PLPSLAM/optimize/g2o/se3/perspective_pose_opt_edge.h"

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            namespace se3
            {

                mono_perspective_pose_opt_edge::mono_perspective_pose_opt_edge()
                    : ::g2o::BaseUnaryEdge<2, Vec2_t, shot_vertex>()
                {
                }

                bool mono_perspective_pose_opt_edge::read(std::istream &is)
                {
                    for (unsigned int i = 0; i < 2; ++i)
                    {
                        is >> _measurement(i);
                    }
                    for (unsigned int i = 0; i < 2; ++i)
                    {
                        for (unsigned int j = i; j < 2; ++j)
                        {
                            is >> information()(i, j);
                            if (i != j)
                            {
                                information()(j, i) = information()(i, j);
                            }
                        }
                    }
                    return true;
                }

                bool mono_perspective_pose_opt_edge::write(std::ostream &os) const
                {
                    for (unsigned int i = 0; i < 2; ++i)
                    {
                        os << measurement()(i) << " ";
                    }
                    for (unsigned int i = 0; i < 2; ++i)
                    {
                        for (unsigned int j = i; j < 2; ++j)
                        {
                            os << " " << information()(i, j);
                        }
                    }
                    return os.good();
                }

                void mono_perspective_pose_opt_edge::linearizeOplus()
                {
                    auto vi = static_cast<shot_vertex *>(_vertices.at(0));
                    const ::g2o::SE3Quat &cam_pose_cw = vi->shot_vertex::estimate();
                    const Vec3_t pos_c = cam_pose_cw.map(pos_w_);

                    const auto x = pos_c(0);
                    const auto y = pos_c(1);
                    const auto z = pos_c(2);
                    const auto z_sq = z * z;

                    // jacobian regarding to camera pose 6DOF
                    _jacobianOplusXi(0, 0) = x * y / z_sq * fx_;
                    _jacobianOplusXi(0, 1) = -(1.0 + (x * x / z_sq)) * fx_;
                    _jacobianOplusXi(0, 2) = y / z * fx_;
                    _jacobianOplusXi(0, 3) = -1.0 / z * fx_;
                    _jacobianOplusXi(0, 4) = 0;
                    _jacobianOplusXi(0, 5) = x / z_sq * fx_;

                    _jacobianOplusXi(1, 0) = (1.0 + y * y / z_sq) * fy_;
                    _jacobianOplusXi(1, 1) = -x * y / z_sq * fy_;
                    _jacobianOplusXi(1, 2) = -x / z * fy_;
                    _jacobianOplusXi(1, 3) = 0.0;
                    _jacobianOplusXi(1, 4) = -1.0 / z * fy_;
                    _jacobianOplusXi(1, 5) = y / z_sq * fy_;
                }

                stereo_perspective_pose_opt_edge::stereo_perspective_pose_opt_edge()
                    : ::g2o::BaseUnaryEdge<3, Vec3_t, shot_vertex>() {}

                bool stereo_perspective_pose_opt_edge::read(std::istream &is)
                {
                    for (unsigned int i = 0; i < 3; ++i)
                    {
                        is >> _measurement(i);
                    }
                    for (unsigned int i = 0; i < 3; ++i)
                    {
                        for (unsigned int j = i; j < 3; ++j)
                        {
                            is >> information()(i, j);
                            if (i != j)
                            {
                                information()(j, i) = information()(i, j);
                            }
                        }
                    }
                    return true;
                }

                bool stereo_perspective_pose_opt_edge::write(std::ostream &os) const
                {
                    for (unsigned int i = 0; i < 3; ++i)
                    {
                        os << measurement()(i) << " ";
                    }
                    for (unsigned int i = 0; i < 3; ++i)
                    {
                        for (unsigned int j = i; j < 3; ++j)
                        {
                            os << " " << information()(i, j);
                        }
                    }
                    return os.good();
                }

                void stereo_perspective_pose_opt_edge::linearizeOplus()
                {
                    auto vi = static_cast<shot_vertex *>(_vertices.at(0));
                    const ::g2o::SE3Quat &cam_pose_cw = vi->shot_vertex::estimate();
                    const Vec3_t pos_c = cam_pose_cw.map(pos_w_);

                    const auto x = pos_c(0);
                    const auto y = pos_c(1);
                    const auto z = pos_c(2);
                    const auto z_sq = z * z;

                    _jacobianOplusXi(0, 0) = x * y / z_sq * fx_;
                    _jacobianOplusXi(0, 1) = -(1.0 + (x * x / z_sq)) * fx_;
                    _jacobianOplusXi(0, 2) = y / z * fx_;
                    _jacobianOplusXi(0, 3) = -1.0 / z * fx_;
                    _jacobianOplusXi(0, 4) = 0.0;
                    _jacobianOplusXi(0, 5) = x / z_sq * fx_;

                    _jacobianOplusXi(1, 0) = (1.0 + y * y / z_sq) * fy_;
                    _jacobianOplusXi(1, 1) = -x * y / z_sq * fy_;
                    _jacobianOplusXi(1, 2) = -x / z * fy_;
                    _jacobianOplusXi(1, 3) = 0.0;
                    _jacobianOplusXi(1, 4) = -1.0 / z * fy_;
                    _jacobianOplusXi(1, 5) = y / z_sq * fy_;

                    _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - focal_x_baseline_ * y / z_sq;
                    _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) + focal_x_baseline_ * x / z_sq;
                    _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2);
                    _jacobianOplusXi(2, 3) = _jacobianOplusXi(0, 3);
                    _jacobianOplusXi(2, 4) = 0.0;
                    _jacobianOplusXi(2, 5) = _jacobianOplusXi(0, 5) - focal_x_baseline_ / z_sq;
                }

            } // namespace se3
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM
