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

#ifndef PLPSLAM_OPTIMIZE_G2O_BACKWARD_REPROJ_EDGE_H
#define PLPSLAM_OPTIMIZE_G2O_BACKWARD_REPROJ_EDGE_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/optimize/g2o/sim3/transform_vertex.h"

#include <g2o/core/base_unary_edge.h>

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            namespace sim3
            {

                class base_backward_reproj_edge : public ::g2o::BaseUnaryEdge<2, Vec2_t, transform_vertex>
                {
                public:
                    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                    base_backward_reproj_edge();

                    bool read(std::istream &is) override;

                    bool write(std::ostream &os) const override;

                    void computeError() final
                    {
                        // Extract Sim3 (2-> 1) from vertex and convert it to Sim3 (1-> 2)
                        const auto v1 = static_cast<const transform_vertex *>(_vertices.at(0));
                        const ::g2o::Sim3 &Sim3_12 = v1->estimate();
                        const ::g2o::Sim3 Sim3_21 = Sim3_12.inverse();
                        // Extract SE3 (world-> 1) from vertex
                        const Mat33_t &rot_1w = v1->rot_1w_;
                        const Vec3_t &trans_1w = v1->trans_1w_;

                        // Convert point coordinate system (world-> 1)
                        const Vec3_t pos_1 = rot_1w * pos_w_ + trans_1w;
                        // Further conversion using Sim3 (1-> 2)
                        const Vec3_t pos_2 = Sim3_21.map(pos_1);
                        // Calculate reprojection error
                        const Vec2_t obs(_measurement);
                        _error = obs - cam_project(pos_2);
                    }

                    virtual Vec2_t cam_project(const Vec3_t &pos_c) const = 0;

                    Vec3_t pos_w_;
                };

                class perspective_backward_reproj_edge final : public base_backward_reproj_edge
                {
                public:
                    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                    perspective_backward_reproj_edge();

                    inline Vec2_t cam_project(const Vec3_t &pos_c) const override
                    {
                        return {fx_ * pos_c(0) / pos_c(2) + cx_, fy_ * pos_c(1) / pos_c(2) + cy_};
                    }

                    double fx_, fy_, cx_, cy_;
                };

                class equirectangular_backward_reproj_edge final : public base_backward_reproj_edge
                {
                public:
                    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                    equirectangular_backward_reproj_edge();

                    inline Vec2_t cam_project(const Vec3_t &pos_c) const override
                    {
                        const double theta = std::atan2(pos_c(0), pos_c(2));
                        const double phi = -std::asin(pos_c(1) / pos_c.norm());
                        return {cols_ * (0.5 + theta / (2 * M_PI)), rows_ * (0.5 - phi / M_PI)};
                    }

                    double cols_, rows_;
                };

            } // namespace sim3
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_OPTIMIZE_G2O_BACKWARD_REPROJ_EDGE_H
