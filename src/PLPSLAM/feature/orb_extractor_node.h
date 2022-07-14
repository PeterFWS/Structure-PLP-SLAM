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

#ifndef PLPSLAM_FEATURE_ORB_EXTRACTOR_NODE_H
#define PLPSLAM_FEATURE_ORB_EXTRACTOR_NODE_H

#include <list>

#include <opencv2/core/types.hpp>

namespace PLPSLAM
{
    namespace feature
    {

        class orb_extractor_node
        {
        public:
            //! Constructor
            orb_extractor_node() = default;

            //! Divide node to four child nodes
            std::array<orb_extractor_node, 4> divide_node();

            //! Keypoints which distributed into this node
            std::vector<cv::KeyPoint> keypts_;

            //! Begin and end of the allocated area on the image
            cv::Point2i pt_begin_, pt_end_;

            //! A iterator pointing to self, used for removal on list
            std::list<orb_extractor_node>::iterator iter_;

            //! A flag designating if this node is a leaf node
            bool is_leaf_node_ = false;
        };

    } // namespace feature
} // namespace PLPSLAM

#endif // PLPSLAM_FEATURE_ORB_EXTRACTOR_NODE_H
