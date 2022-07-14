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

#ifndef PLPSLAM_OPTIMIZER_G2O_SE3_REPROJ_EDGE_LINE3D_ORTHONORMAL_H
#define PLPSLAM_OPTIMIZER_G2O_SE3_REPROJ_EDGE_LINE3D_ORTHONORMAL_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/optimize/g2o/line3d.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex_line3d.h"
#include "PLPSLAM/optimize/g2o/se3/shot_vertex.h"
#include "PLPSLAM/util/converter.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
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
                // FW:
                // error function is the 2D-3D line reprojection error
                // Plücker coordinates, and orthonormal representation
                class reproj_edge_line3d
                    : public ::g2o::BaseBinaryEdge<2, Vec4_t, shot_vertex, VertexLine3D>
                {
                public:
                    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

                    reproj_edge_line3d();

                    virtual bool read(std::istream &is);
                    virtual bool write(std::ostream &os) const;

                    void computeError() override
                    {
                        const shot_vertex *se3Vertex = static_cast<const shot_vertex *>(_vertices[0]);
                        const VertexLine3D *lineVertex = static_cast<const VertexLine3D *>(_vertices[1]);

                        const Line3D &line = lineVertex->estimate();
                        auto pose_keyframe_SE3Quat = se3Vertex->estimate();
                        Vec3_t proj = cam_project(pose_keyframe_SE3Quat, line); // re-projected line function (ax+by+c=0)

                        Vec4_t obs(_measurement); // (xs,ys, xe, ye) the two endpoints of the detected line segment
                        _error(0) = (obs(0) * proj(0) + obs(1) * proj(1) + proj(2)) / sqrt(proj(0) * proj(0) + proj(1) * proj(1));
                        _error(1) = (obs(2) * proj(0) + obs(3) * proj(1) + proj(2)) / sqrt(proj(0) * proj(0) + proj(1) * proj(1));
                    }

                    // return the re-projected line function (ax+by+c=0)
                    inline Vec3_t cam_project(const ::g2o::SE3Quat &cam_pose_cw, const Vec6_t &plucker_coord) const
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

                    bool depth_is_positive_via_endpoints_trimming() const
                    {
                        // endpoints of line segment
                        Vec2_t sp{_measurement(0), _measurement(1)};
                        Vec2_t ep{_measurement(2), _measurement(3)};

                        // get the line function of re-projected 3D line
                        const shot_vertex *se3Vertex = static_cast<const shot_vertex *>(_vertices[0]);
                        const VertexLine3D *lineVertex = static_cast<const VertexLine3D *>(_vertices[1]);
                        const Line3D &line = lineVertex->estimate();
                        auto pose_keyframe_SE3Quat = se3Vertex->estimate();
                        Vec3_t proj = cam_project(pose_keyframe_SE3Quat, line); // re-projected line function (ax+by+c=0)
                        double l1 = proj(0);
                        double l2 = proj(1);
                        double l3 = proj(2);

                        // calculate closet point on the re-projected line
                        double x_sp_closet = -(sp(1) - (l2 / l1) * sp(0) + (l3 / l2)) * ((l1 * l2) / (l1 * l1 + l2 * l2));
                        double y_sp_closet = -(l1 / l2) * x_sp_closet - (l3 / l2);

                        double x_ep_closet = -(ep(1) - (l2 / l1) * ep(0) + (l3 / l2)) * ((l1 * l2) / (l1 * l1 + l2 * l2));
                        double y_ep_closet = -(l1 / l2) * x_ep_closet - (l3 / l2);

                        // calculate another point
                        double x_0sp = 0;
                        double y_0sp = sp(1) - (l2 / l1) * sp(0);

                        double x_0ep = 0;
                        double y_0ep = ep(1) - (l2 / l1) * ep(0);

                        // calculate 3D plane
                        Mat44_t pose_cw = util::converter::to_eigen_mat(pose_keyframe_SE3Quat);
                        const Mat33_t rot_cw = pose_cw.block<3, 3>(0, 0);
                        const Vec3_t trans_cw = pose_cw.block<3, 1>(0, 3);

                        Mat34_t P;
                        Mat34_t rotation_translation_combined = Eigen::Matrix<double, 3, 4>::Zero();
                        rotation_translation_combined.block<3, 3>(0, 0) = rot_cw;
                        rotation_translation_combined.block<3, 1>(0, 3) = trans_cw;
                        P = _cam_matrix * rotation_translation_combined;

                        Vec3_t point2d_sp_closet{x_sp_closet, y_sp_closet, 1.0};
                        Vec3_t point2d_0sp{x_0sp, y_0sp, 1.0};
                        Vec3_t line_temp_sp = point2d_sp_closet.cross(point2d_0sp);
                        Vec4_t plane3d_temp_sp = P.transpose() * line_temp_sp;

                        Vec3_t point2d_ep_closet{x_ep_closet, y_ep_closet, 1.0};
                        Vec3_t point2d_0ep{x_0ep, y_0ep, 1.0};
                        Vec3_t line_temp_ep = point2d_ep_closet.cross(point2d_0ep);
                        Vec4_t plane3d_temp_ep = P.transpose() * line_temp_ep;

                        // calculate intersection of the 3D plane and 3d line
                        Mat44_t line3d_pluecker_matrix = Eigen::Matrix<double, 4, 4>::Zero();
                        Vec3_t m = line.head<3>();
                        Vec3_t d = line.tail<3>();
                        line3d_pluecker_matrix.block<3, 3>(0, 0) = skew(m);
                        line3d_pluecker_matrix.block<3, 1>(0, 3) = d;
                        line3d_pluecker_matrix.block<1, 3>(3, 0) = -d.transpose();

                        Vec4_t intersect_endpoint_sp, intersect_endpoint_ep;
                        intersect_endpoint_sp = line3d_pluecker_matrix * plane3d_temp_sp;
                        intersect_endpoint_ep = line3d_pluecker_matrix * plane3d_temp_ep;

                        // check positive depth
                        Vec4_t pos_w_sp;
                        pos_w_sp << intersect_endpoint_sp(0) / intersect_endpoint_sp(3),
                            intersect_endpoint_sp(1) / intersect_endpoint_sp(3),
                            intersect_endpoint_sp(2) / intersect_endpoint_sp(3),
                            1.0;

                        Vec4_t pos_w_ep;
                        pos_w_ep << intersect_endpoint_ep(0) / intersect_endpoint_ep(3),
                            intersect_endpoint_ep(1) / intersect_endpoint_ep(3),
                            intersect_endpoint_ep(2) / intersect_endpoint_ep(3),
                            1.0;

                        Vec3_t pos_c_sp = rotation_translation_combined * pos_w_sp;
                        Vec3_t pos_c_ep = rotation_translation_combined * pos_w_ep;

                        return 0 < pos_c_sp(2) && 0 < pos_c_ep(2);
                    }

                    double _fx, _fy, _cx, _cy;
                    Mat33_t _K;
                    Mat33_t _cam_matrix;
                };
            }
        }
    }
}

#endif