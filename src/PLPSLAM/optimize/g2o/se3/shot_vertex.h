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

#ifndef PLPSLAM_OPTIMIZER_G2O_SE3_SHOT_VERTEX_H
#define PLPSLAM_OPTIMIZER_G2O_SE3_SHOT_VERTEX_H

#include "PLPSLAM/type.h"

#include <g2o/core/base_vertex.h>
#include <g2o/types/slam3d/se3quat.h>

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            namespace se3
            {

                class shot_vertex final : public ::g2o::BaseVertex<6, ::g2o::SE3Quat>
                {
                    // vertex for frame/keyframe
                public:
                    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                    shot_vertex();

                    bool read(std::istream &is) override;

                    bool write(std::ostream &os) const override;

                    void setToOriginImpl() override
                    {
                        _estimate = ::g2o::SE3Quat();
                    }

                    void oplusImpl(const number_t *update_) override
                    {
                        Eigen::Map<const Vec6_t> update(update_);
                        setEstimate(::g2o::SE3Quat::exp(update) * estimate());
                    }
                };

            } // namespace se3
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_OPTIMIZER_G2O_SE3_SHOT_VERTEX_H
