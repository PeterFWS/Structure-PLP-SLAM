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

#include "PLPSLAM/optimize/g2o/extended/plane_point_distance_edge.h"

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            namespace extended
            {
                point_plane_distance_edge::point_plane_distance_edge()
                    : ::g2o::BaseUnaryEdge<1, Vec4_t, landmark_vertex>()
                {
                }

                bool point_plane_distance_edge::read(std::istream &is)
                {
                    for (unsigned int i = 0; i < 4; ++i)
                    {
                        is >> _measurement(i);
                    }

                    for (unsigned int i = 0; i < 2; ++i)
                    {
                        for (unsigned int j = i; j < 2; ++j)
                        {
                            is >> information()(i, j);
                            if (i != j)
                            {
                                information()(j, i) = information()(i, j);
                            }
                        }
                    }

                    return true;
                }

                bool point_plane_distance_edge::write(std::ostream &os) const
                {
                    for (unsigned int i = 0; i < 4; ++i)
                    {
                        os << measurement()(i) << " ";
                    }

                    for (unsigned int i = 0; i < 2; ++i)
                    {
                        for (unsigned int j = i; j < 2; ++j)
                        {
                            os << " " << information()(i, j);
                        }
                    }

                    return os.good();
                }
            } // namespace extended
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM