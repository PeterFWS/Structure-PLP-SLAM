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

#ifndef PLPSLAM_OPTIMIZER_G2O_SE3_EQUIRECTANGULAR_POSE_OPT_EDGE_H
#define PLPSLAM_OPTIMIZER_G2O_SE3_EQUIRECTANGULAR_POSE_OPT_EDGE_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex.h"
#include "PLPSLAM/optimize/g2o/se3/shot_vertex.h"

#include <g2o/core/base_unary_edge.h>

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            namespace se3
            {

                class equirectangular_pose_opt_edge final : public ::g2o::BaseUnaryEdge<2, Vec2_t, shot_vertex>
                {
                public:
                    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                    equirectangular_pose_opt_edge();

                    bool read(std::istream &is) override;

                    bool write(std::ostream &os) const override;

                    void computeError() override
                    {
                        const auto v1 = static_cast<const shot_vertex *>(_vertices.at(0));
                        const Vec2_t obs(_measurement);
                        _error = obs - cam_project(v1->estimate().map(pos_w_));
                    }

                    void linearizeOplus() override;

                    inline Vec2_t cam_project(const Vec3_t &pos_c) const
                    {
                        const double theta = std::atan2(pos_c(0), pos_c(2));
                        const double phi = -std::asin(pos_c(1) / pos_c.norm());
                        return {cols_ * (0.5 + theta / (2 * M_PI)), rows_ * (0.5 - phi / M_PI)};
                    }

                    Vec3_t pos_w_;
                    double cols_, rows_;
                };

            } // namespace se3
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_EQUIRECTANGULAR_POSE_OPT_EDGE_H
