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

#ifndef PLPSLAM_OPTIMIZER_G2O_SE3_PERSPECTIVE_REPROJ_EDGE_H
#define PLPSLAM_OPTIMIZER_G2O_SE3_PERSPECTIVE_REPROJ_EDGE_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex.h"
#include "PLPSLAM/optimize/g2o/se3/shot_vertex.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            namespace se3
            {
                // FW: a binary edge, inherited from BaseBinaryEdge
                //  - the dimension of the error -> 2,
                //  - the datatype of measurement -> vector2d,
                //  - the datatype of vertex connected -> landmark_vertex + keyframe_vertex
                class mono_perspective_reproj_edge final : public ::g2o::BaseBinaryEdge<2, Vec2_t, landmark_vertex, shot_vertex>
                {
                public:
                    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                    mono_perspective_reproj_edge(); // Binary edge

                    bool read(std::istream &is) override;

                    bool write(std::ostream &os) const override;

                    // reprojection error
                    void computeError() override
                    {
                        const auto v1 = static_cast<const shot_vertex *>(_vertices.at(1));
                        const auto v2 = static_cast<const landmark_vertex *>(_vertices.at(0));
                        const Vec2_t obs(_measurement);
                        _error = obs - cam_project(v1->estimate().map(v2->estimate()));
                    }

                    // jacobian matrix (not mandotary to code by hand)
                    void linearizeOplus() override;

                    bool depth_is_positive() const
                    {
                        const auto v1 = static_cast<const shot_vertex *>(_vertices.at(1));
                        const auto v2 = static_cast<const landmark_vertex *>(_vertices.at(0));
                        return 0.0 < (v1->estimate().map(v2->estimate()))(2);
                    }

                    inline Vec2_t cam_project(const Vec3_t &pos_c) const
                    {
                        // project 3D point to 2D image pixel coordinates
                        return {fx_ * pos_c(0) / pos_c(2) + cx_, fy_ * pos_c(1) / pos_c(2) + cy_};
                    }

                    double fx_, fy_, cx_, cy_;
                };

                class stereo_perspective_reproj_edge final : public ::g2o::BaseBinaryEdge<3, Vec3_t, landmark_vertex, shot_vertex>
                {
                public:
                    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                    stereo_perspective_reproj_edge();

                    bool read(std::istream &is) override;

                    bool write(std::ostream &os) const override;

                    void computeError() override
                    {
                        const auto v1 = static_cast<const shot_vertex *>(_vertices.at(1));
                        const auto v2 = static_cast<const landmark_vertex *>(_vertices.at(0));
                        const Vec3_t obs(_measurement);
                        _error = obs - cam_project(v1->estimate().map(v2->estimate()));
                    }

                    void linearizeOplus() override;

                    bool depth_is_positive() const
                    {
                        const auto v1 = static_cast<const shot_vertex *>(_vertices.at(1));
                        const auto v2 = static_cast<const landmark_vertex *>(_vertices.at(0));
                        return 0 < (v1->estimate().map(v2->estimate()))(2);
                    }

                    inline Vec3_t cam_project(const Vec3_t &pos_c) const
                    {
                        const double reproj_x = fx_ * pos_c(0) / pos_c(2) + cx_;
                        return {reproj_x, fy_ * pos_c(1) / pos_c(2) + cy_, reproj_x - focal_x_baseline_ / pos_c(2)};
                    }

                    double fx_, fy_, cx_, cy_, focal_x_baseline_;
                };

            } // namespace se3
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_OPTIMIZER_G2O_SE3_PERSPECTIVE_REPROJ_EDGE_H
