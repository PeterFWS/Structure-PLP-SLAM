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

#ifndef PLPSLAM_OPTIMIZER_G2O_LANDMARK_VERTEX_PLANE_H
#define PLPSLAM_OPTIMIZER_G2O_LANDMARK_VERTEX_PLANE_H

#include "PLPSLAM/type.h"
#include <g2o/core/base_vertex.h>
#include "PLPSLAM/data/landmark_plane.h"
#include "PLPSLAM/optimize/g2o/Plane3D.h"

#include <g2o/config.h>
#include <g2o/stuff/misc.h>
#include <g2o/core/eigen_types.h>

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            class landmark_vertex_plane
                : public ::g2o::BaseVertex<3, Plane3D>
            {
                // A vertex for 3D plane landmark:
                // The DOF of variables to be optimized is 3 -> (phi,psi,d), the datatype is Plane3D
            public:
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                landmark_vertex_plane();

                bool read(std::istream &is) override;
                bool write(std::ostream &os) const override;

                // reset function
                void setToOriginImpl() override
                {
                    _estimate = Plane3D(); // initialize the variables to be optimized
                }

                // update function (important), this is the ⊞ operater in mainfold
                void oplusImpl(const double *update_) override
                {
                    Eigen::Map<const ::g2o::Vector3> update(update_);
                    _estimate.oplus(update);
                }

                bool setEstimateDataImpl(const double *est) override
                {
                    Eigen::Map<const ::g2o::Vector4> _est(est);
                    _estimate.fromVector(_est);
                    return true;
                }

                bool getEstimateData(double *est) const override
                {
                    Eigen::Map<::g2o::Vector4> _est(est);
                    _est = _estimate.toVector();
                    return true;
                }

                int estimateDimension() const override
                {
                    return 4;
                }
            };

        } // namespace g2o
    }     // namespace optimize
} // namespace PLPSLAM

#endif