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

#ifndef PLPSLAM_OPTIMIZER_G2O_SIM3_SHOT_VERTEX_H
#define PLPSLAM_OPTIMIZER_G2O_SIM3_SHOT_VERTEX_H

#include "PLPSLAM/type.h"

#include <g2o/core/base_vertex.h>
#include <g2o/types/sim3/sim3.h>

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            namespace sim3
            {

                class shot_vertex final : public ::g2o::BaseVertex<7, ::g2o::Sim3>
                {
                public:
                    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                    shot_vertex();

                    bool read(std::istream &is) override;

                    bool write(std::ostream &os) const override;

                    void setToOriginImpl() override
                    {
                        _estimate = ::g2o::Sim3();
                    }

                    void oplusImpl(const number_t *update_) override
                    {
                        Eigen::Map<Vec7_t> update(const_cast<number_t *>(update_));

                        if (fix_scale_)
                        {
                            update(6) = 0;
                        }

                        const ::g2o::Sim3 s(update);
                        setEstimate(s * estimate());
                    }

                    bool fix_scale_;
                };

            } // namespace sim3
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_OPTIMIZER_G2O_SIM3_SHOT_VERTEX_H
