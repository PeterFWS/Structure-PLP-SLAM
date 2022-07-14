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

#include "PLPSLAM/optimize/g2o/sim3/forward_reproj_edge.h"

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            namespace sim3
            {

                base_forward_reproj_edge::base_forward_reproj_edge()
                    : ::g2o::BaseUnaryEdge<2, Vec2_t, transform_vertex>()
                {
                }

                bool base_forward_reproj_edge::read(std::istream &is)
                {
                    for (unsigned int i = 0; i < 2; ++i)
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

                bool base_forward_reproj_edge::write(std::ostream &os) const
                {
                    for (unsigned int i = 0; i < 2; ++i)
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

                perspective_forward_reproj_edge::perspective_forward_reproj_edge()
                    : base_forward_reproj_edge() {}

                equirectangular_forward_reproj_edge::equirectangular_forward_reproj_edge()
                    : base_forward_reproj_edge() {}

            } // namespace sim3
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM
