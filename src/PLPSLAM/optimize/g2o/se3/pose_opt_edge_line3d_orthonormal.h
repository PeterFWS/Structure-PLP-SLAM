/**
 * This file is part of Structure PLP-SLAM.
 *
 * Copyright 2022 DFKI (German Research Center for Artificial Intelligence)
 * Developed by Fangwen Shu <Fangwen.Shu@dfki.de>
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

#ifndef PLPSLAM_OPTIMIZER_G2O_SE3_POSE_OPT_EDGE_LINE3D_ORTHONORMAL_H
#define PLPSLAM_OPTIMIZER_G2O_SE3_POSE_OPT_EDGE_LINE3D_ORTHONORMAL_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex.h"
#include "PLPSLAM/optimize/g2o/se3/shot_vertex.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>

// FW:
#include "PLPSLAM/optimize/g2o/line3d.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex_line3d.h"
#include "PLPSLAM/util/converter.h"
#include <g2o/types/slam3d/parameter_se3_offset.h>
#include <g2o/types/slam3d/isometry3d_mappings.h>

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            namespace se3
            {

                class pose_opt_edge_line3d final
                    : public ::g2o::BaseUnaryEdge<2, Vec4_t, shot_vertex>
                {
                public:
                    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                    pose_opt_edge_line3d(); // Unary edge

                    bool read(std::istream &is) override;

                    bool write(std::ostream &os) const override;

                    void computeError() override
                    {
                        const shot_vertex *se3Vertex = static_cast<const shot_vertex *>(_vertices[0]);
                        auto pose_keyframe_SE3Quat = se3Vertex->estimate();
                        Vec3_t proj = cam_project(pose_keyframe_SE3Quat, _pos_w); // re-projected line function (ax+by+c=0)

                        Vec4_t obs(_measurement); // (xs,ys, xe, ye) the two endpoints of the detected line segment
                        _error(0) = (obs(0) * proj(0) + obs(1) * proj(1) + proj(2)) / sqrt(proj(0) * proj(0) + proj(1) * proj(1));
                        _error(1) = (obs(2) * proj(0) + obs(3) * proj(1) + proj(2)) / sqrt(proj(0) * proj(0) + proj(1) * proj(1));
                    }

                    // return the re-projected line function (ax+by+c=0)
                    inline Vec3_t cam_project(const ::g2o::SE3Quat &cam_pose_cw, const Vec6_t &plucker_coord)
                    {
                        Mat44_t pose_cw = util::converter::to_eigen_mat(cam_pose_cw);

                        const Mat33_t rot_cw = pose_cw.block<3, 3>(0, 0);
                        const Vec3_t trans_cw = pose_cw.block<3, 1>(0, 3);

                        Mat66_t transformation_line_cw = Eigen::Matrix<double, 6, 6>::Zero();
                        transformation_line_cw.block<3, 3>(0, 0) = rot_cw;
                        transformation_line_cw.block<3, 3>(3, 3) = rot_cw;
                        transformation_line_cw.block<3, 3>(0, 3) = skew(trans_cw) * rot_cw;

                        return _K * (transformation_line_cw * plucker_coord).block<3, 1>(0, 0);
                    }

                    inline Mat33_t skew(const Vec3_t &t) const
                    {
                        Mat33_t S;
                        S << 0, -t.z(), t.y(), t.z(), 0, -t.x(), -t.y(), t.x(), 0;
                        return S;
                    }

                    Vec6_t _pos_w; // Plücker coordinates of the 3D line

                    number_t _fx, _fy, _cx, _cy;
                    Mat33_t _K;
                };

            } // namespace se3
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM

#endif
