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

#include "PLPSLAM/optimize/g2o/landmark_vertex.h"

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {

            landmark_vertex::landmark_vertex() : BaseVertex<3, Vec3_t>()
            {
            }

            bool landmark_vertex::read(std::istream &is)
            {
                Vec3_t lv;
                for (unsigned int i = 0; i < 3; ++i)
                {
                    is >> _estimate(i);
                }
                return true;
            }

            bool landmark_vertex::write(std::ostream &os) const
            {
                const Vec3_t pos_w = estimate();
                for (unsigned int i = 0; i < 3; ++i)
                {
                    os << pos_w(i) << " ";
                }
                return os.good();
            }

        } // namespace g2o
    }     // namespace optimize
} // namespace PLPSLAM
