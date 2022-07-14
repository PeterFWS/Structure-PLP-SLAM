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

#include "PLPSLAM/optimize/g2o/landmark_vertex_line3d.h"

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            VertexLine3D::VertexLine3D() : ::g2o::BaseVertex<4, Line3D>()
            {
            }

            bool VertexLine3D::read(std::istream &is)
            {
                Vector6 lv;
                bool state = ::g2o::internal::readVector(is, lv);
                setEstimate(Line3D(lv));
                return state;
            }

            bool VertexLine3D::write(std::ostream &os) const
            {
                return ::g2o::internal::writeVector(os, _estimate);
            }
        }
    }
}