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

#include "PLPSLAM/optimize/g2o/landmark_vertex_plane.h"

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            landmark_vertex_plane::landmark_vertex_plane()
                : ::g2o::BaseVertex<3, Plane3D>()
            {
            }

            bool landmark_vertex_plane::read(std::istream &is)
            {
                ::g2o::Vector4 lv;
                bool state = ::g2o::internal::readVector(is, lv);
                setEstimate(Plane3D(lv));
                return state;
            }

            bool landmark_vertex_plane::write(std::ostream &os) const
            {
                bool state = ::g2o::internal::writeVector(os, _estimate.toVector());
                return state;
            }
        } // namespace g2o
    }     // namespace optimize
} // namespace PLPSLAM
