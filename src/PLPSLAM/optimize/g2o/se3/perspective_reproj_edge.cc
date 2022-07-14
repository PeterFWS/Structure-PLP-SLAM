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

#include "PLPSLAM/optimize/g2o/se3/perspective_reproj_edge.h"

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            namespace se3
            {
                mono_perspective_reproj_edge::mono_perspective_reproj_edge()
                    : ::g2o::BaseBinaryEdge<2, Vec2_t, landmark_vertex, shot_vertex>()
                {
                }

                bool mono_perspective_reproj_edge::read(std::istream &is)
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

                bool mono_perspective_reproj_edge::write(std::ostream &os) const
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

                // Define the jacobian matrix of the reprojection error
                void mono_perspective_reproj_edge::linearizeOplus()
                {
                    auto vj = static_cast<shot_vertex *>(_vertices.at(1));           // data::keyframe
                    const ::g2o::SE3Quat &cam_pose_cw = vj->shot_vertex::estimate(); // get camera pose in Quat

                    auto vi = static_cast<landmark_vertex *>(_vertices.at(0)); // data::landmark (3D point)
                    const Vec3_t &pos_w = vi->landmark_vertex::estimate();     // get position of the 3D point in world coordinate
                    const Vec3_t pos_c = cam_pose_cw.map(pos_w);               // transform position of the 3D point: world coordinate -> camera coordinate (R*X_w + t)

                    const auto x = pos_c(0);
                    const auto y = pos_c(1);
                    const auto z = pos_c(2);
                    const auto z_sq = z * z;

                    const Mat33_t rot_cw = cam_pose_cw.rotation().toRotationMatrix(); // rotation matrix

                    // FW:
                    // rotation and translation matrix (w->c)
                    // coordinates of the 3D points (c)
                    // The Jacobian matrix -> the partial derivative to the landmark and to the camera pose
                    // --------------------------------------------------------------------------------------------------------------------
                    // [1] the jacobian matrix (2x3) of vertex Xi -> landmark (3D point)
                    // error has 2 dimension, a 3D point has 3 dimension -> 2x3
                    _jacobianOplusXi(0, 0) = -fx_ * rot_cw(0, 0) / z + fx_ * x * rot_cw(2, 0) / z_sq;
                    _jacobianOplusXi(0, 1) = -fx_ * rot_cw(0, 1) / z + fx_ * x * rot_cw(2, 1) / z_sq;
                    _jacobianOplusXi(0, 2) = -fx_ * rot_cw(0, 2) / z + fx_ * x * rot_cw(2, 2) / z_sq;

                    _jacobianOplusXi(1, 0) = -fy_ * rot_cw(1, 0) / z + fy_ * y * rot_cw(2, 0) / z_sq;
                    _jacobianOplusXi(1, 1) = -fy_ * rot_cw(1, 1) / z + fy_ * y * rot_cw(2, 1) / z_sq;
                    _jacobianOplusXi(1, 2) = -fy_ * rot_cw(1, 2) / z + fy_ * y * rot_cw(2, 2) / z_sq;

                    // [2] the jacobian matrix (2x6) of vertex Xj -> keyframe
                    // error has 2 dimension, a keyframe has 6 dimension (SE3, lie group representation)
                    _jacobianOplusXj(0, 0) = x * y / z_sq * fx_;
                    _jacobianOplusXj(0, 1) = -(1.0 + (x * x / z_sq)) * fx_;
                    _jacobianOplusXj(0, 2) = y / z * fx_;
                    _jacobianOplusXj(0, 3) = -1.0 / z * fx_;
                    _jacobianOplusXj(0, 4) = 0.0;
                    _jacobianOplusXj(0, 5) = x / z_sq * fx_;

                    _jacobianOplusXj(1, 0) = (1.0 + y * y / z_sq) * fy_;
                    _jacobianOplusXj(1, 1) = -x * y / z_sq * fy_;
                    _jacobianOplusXj(1, 2) = -x / z * fy_;
                    _jacobianOplusXj(1, 3) = 0.0;
                    _jacobianOplusXj(1, 4) = -1.0 / z * fy_;
                    _jacobianOplusXj(1, 5) = y / z_sq * fy_;
                    // --------------------------------------------------------------------------------------------------------------------
                }

                stereo_perspective_reproj_edge::stereo_perspective_reproj_edge()
                    : BaseBinaryEdge<3, Vec3_t, landmark_vertex, shot_vertex>() {}

                bool stereo_perspective_reproj_edge::read(std::istream &is)
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

                bool stereo_perspective_reproj_edge::write(std::ostream &os) const
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

                void stereo_perspective_reproj_edge::linearizeOplus()
                {
                    auto vj = static_cast<shot_vertex *>(_vertices.at(1));
                    const ::g2o::SE3Quat &cam_pose_cw = vj->shot_vertex::estimate();

                    auto vi = static_cast<landmark_vertex *>(_vertices.at(0));
                    const Vec3_t &pos_w = vi->landmark_vertex::estimate();
                    const Vec3_t pos_c = cam_pose_cw.map(pos_w);

                    const auto x = pos_c(0);
                    const auto y = pos_c(1);
                    const auto z = pos_c(2);
                    const auto z_sq = z * z;

                    const Mat33_t rot_cw = cam_pose_cw.rotation().toRotationMatrix();

                    _jacobianOplusXi(0, 0) = -fx_ * rot_cw(0, 0) / z + fx_ * x * rot_cw(2, 0) / z_sq;
                    _jacobianOplusXi(0, 1) = -fx_ * rot_cw(0, 1) / z + fx_ * x * rot_cw(2, 1) / z_sq;
                    _jacobianOplusXi(0, 2) = -fx_ * rot_cw(0, 2) / z + fx_ * x * rot_cw(2, 2) / z_sq;

                    _jacobianOplusXi(1, 0) = -fy_ * rot_cw(1, 0) / z + fy_ * y * rot_cw(2, 0) / z_sq;
                    _jacobianOplusXi(1, 1) = -fy_ * rot_cw(1, 1) / z + fy_ * y * rot_cw(2, 1) / z_sq;
                    _jacobianOplusXi(1, 2) = -fy_ * rot_cw(1, 2) / z + fy_ * y * rot_cw(2, 2) / z_sq;

                    _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - focal_x_baseline_ * rot_cw(2, 0) / z_sq;
                    _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) - focal_x_baseline_ * rot_cw(2, 1) / z_sq;
                    _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2) - focal_x_baseline_ * rot_cw(2, 2) / z_sq;

                    _jacobianOplusXj(0, 0) = x * y / z_sq * fx_;
                    _jacobianOplusXj(0, 1) = -(1.0 + (x * x / z_sq)) * fx_;
                    _jacobianOplusXj(0, 2) = y / z * fx_;
                    _jacobianOplusXj(0, 3) = -1.0 / z * fx_;
                    _jacobianOplusXj(0, 4) = 0.0;
                    _jacobianOplusXj(0, 5) = x / z_sq * fx_;

                    _jacobianOplusXj(1, 0) = (1.0 + y * y / z_sq) * fy_;
                    _jacobianOplusXj(1, 1) = -x * y / z_sq * fy_;
                    _jacobianOplusXj(1, 2) = -x / z * fy_;
                    _jacobianOplusXj(1, 3) = 0.0;
                    _jacobianOplusXj(1, 4) = -1.0 / z * fy_;
                    _jacobianOplusXj(1, 5) = y / z_sq * fy_;

                    _jacobianOplusXj(2, 0) = _jacobianOplusXj(0, 0) - focal_x_baseline_ * y / z_sq;
                    _jacobianOplusXj(2, 1) = _jacobianOplusXj(0, 1) + focal_x_baseline_ * x / z_sq;
                    _jacobianOplusXj(2, 2) = _jacobianOplusXj(0, 2);
                    _jacobianOplusXj(2, 3) = _jacobianOplusXj(0, 3);
                    _jacobianOplusXj(2, 4) = 0;
                    _jacobianOplusXj(2, 5) = _jacobianOplusXj(0, 5) - focal_x_baseline_ / z_sq;
                }

            } // namespace se3
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM
