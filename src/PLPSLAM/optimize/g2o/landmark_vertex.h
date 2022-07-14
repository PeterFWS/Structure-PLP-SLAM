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

#ifndef PLPSLAM_OPTIMIZER_G2O_LANDMARK_VERTEX_H
#define PLPSLAM_OPTIMIZER_G2O_LANDMARK_VERTEX_H

#include "PLPSLAM/type.h"

#include <g2o/core/base_vertex.h>

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            class landmark_vertex final : public ::g2o::BaseVertex<3, Vec3_t>
            {
                // A vertex for 3D point landmark:
                // Inherited from the built-in class BaseVertex of g2o
                // The DOF of variables to be optimized is 3 -> (x,y,z), the datatype is Eigen::vector3d
            public:
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                landmark_vertex();

                // The read/write function, reset/update function are not called by us,
                // but they are called by the g2o solver during the iterative optimization process.
                bool read(std::istream &is) override;
                bool write(std::ostream &os) const override;

                // reset function
                void setToOriginImpl() override
                {
                    _estimate.fill(0); // initialize the variables to be optimized as (0,0,0)
                }

                // update function
                // the update function just takes the current value and add up a incremental value during iterative optimization.
                void oplusImpl(const double *update) override
                {
                    Eigen::Map<const Vec3_t> v(update); // map the array "update" to Eigen vector_3d, without any overhead cost
                    _estimate += v;
                }
            };

        } // namespace g2o
    }     // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_OPTIMIZER_G2O_LANDMARK_VERTEX_H
