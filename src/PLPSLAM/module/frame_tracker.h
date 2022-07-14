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

#ifndef PLPSLAM_MODULE_FRAME_TRACKER_H
#define PLPSLAM_MODULE_FRAME_TRACKER_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/optimize/pose_optimizer.h"
#include "PLPSLAM/optimize/pose_optimizer_extended_line.h"

#include "PLPSLAM/initialize/base.h"
#include "PLPSLAM/initialize/perspective.h"

namespace PLPSLAM
{

    namespace camera
    {
        class base;
    } // namespace camera

    namespace data
    {
        class frame;
        class keyframe;
        class Line;
    } // namespace data

    namespace module
    {

        class frame_tracker
        {
        public:
            explicit frame_tracker(camera::base *camera, const unsigned int num_matches_thr = 20);

            bool motion_based_track(data::frame &curr_frm, const data::frame &last_frm, const Mat44_t &velocity) const;

            bool bow_match_based_track(data::frame &curr_frm, const data::frame &last_frm, data::keyframe *ref_keyfrm) const;

            bool robust_match_based_track(data::frame &curr_frm, const data::frame &last_frm, data::keyframe *ref_keyfrm) const;

            // FW:
            void set_using_line_tracking();

        private:
            unsigned int discard_outliers(data::frame &curr_frm) const;

            // FW:
            unsigned int discard_outliers_line(data::frame &curr_frm) const;

            const camera::base *camera_;
            const unsigned int num_matches_thr_;

            const optimize::pose_optimizer pose_optimizer_;
            const optimize::pose_optimizer_extended_line _pose_optimizer_extended_line;

            // FW:
            bool _b_use_line_tracking = false;
        };

    } // namespace module
} // namespace PLPSLAM

#endif // PLPSLAM_MODULE_FRAME_TRACKER_H
