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

#include "PLPSLAM/optimize/g2o/se3/shot_vertex.h"

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            namespace se3
            {

                shot_vertex::shot_vertex()
                    : BaseVertex<6, ::g2o::SE3Quat>()
                {
                }

                bool shot_vertex::read(std::istream &is)
                {
                    Vec7_t estimate;
                    for (unsigned int i = 0; i < 7; ++i)
                    {
                        is >> estimate(i);
                    }
                    ::g2o::SE3Quat g2o_cam_pose_wc;
                    g2o_cam_pose_wc.fromVector(estimate);
                    setEstimate(g2o_cam_pose_wc.inverse());
                    return true;
                }

                bool shot_vertex::write(std::ostream &os) const
                {
                    ::g2o::SE3Quat g2o_cam_pose_wc(estimate().inverse());
                    for (unsigned int i = 0; i < 7; ++i)
                    {
                        os << g2o_cam_pose_wc[i] << " ";
                    }
                    return os.good();
                }

            } // namespace se3
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM
