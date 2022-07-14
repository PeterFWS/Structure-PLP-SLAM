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

#ifndef PLPSLAM_OPTIMIZER_G2O_EXTENDED_PLANE_POINT_DISTANCE_EDGE_H
#define PLPSLAM_OPTIMIZER_G2O_EXTENDED_PLANE_POINT_DISTANCE_EDGE_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex_plane.h"
#include "PLPSLAM/optimize/g2o/Plane3D.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            namespace extended
            {
                class point_plane_distance_edge final
                    : public ::g2o::BaseUnaryEdge<1, Vec4_t, landmark_vertex>
                {
                public:
                    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                    point_plane_distance_edge();

                    // minimizing distance error to zero
                    void computeError() override
                    {
                        const landmark_vertex *vt_pt = static_cast<const landmark_vertex *>(_vertices[0]);

                        Vec3_t pos_w = vt_pt->estimate();
                        Vec4_t plane3D_function(_measurement); // (n, d)
                        _error[0] = (pos_w.dot(plane3D_function.head<3>()) + plane3D_function(3)) /
                                    plane3D_function.head<3>().norm();
                    }

                    // no implementation needed?
                    virtual bool read(std::istream &is);

                    virtual bool write(std::ostream &os) const;
                };
            } // namespace extended
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM

#endif