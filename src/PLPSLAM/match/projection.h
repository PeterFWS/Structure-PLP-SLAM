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

#ifndef PLPSLAM_MATCH_PROJECTION_H
#define PLPSLAM_MATCH_PROJECTION_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/match/base.h"

#include <set>

namespace PLPSLAM
{

    namespace data
    {
        class frame;
        class keyframe;
        class landmark;
        class Line; // FW:
    }

    namespace match
    {

        class projection final : public base
        {
        public:
            explicit projection(const float lowe_ratio = 0.6, const bool check_orientation = true)
                : base(lowe_ratio, check_orientation) {}

            ~projection() final = default;

            //-----------------------------------------
            //! Find the 2D point of frame and 3D correspondence, and record the correspondence information in frame.landmarks_
            // used in tracking_module::search_local_landmarks()
            unsigned int match_frame_and_landmarks(data::frame &frm, const std::vector<data::landmark *> &local_landmarks, const float margin = 5.0) const;

            // FW:
            // find the 2D and 3D line correspondence
            // used in tracking_module::search_local_landmarks_line()
            unsigned int match_frame_and_landmarks_line(data::frame &frm, const std::vector<data::Line *> &local_landmarks_line, const float margin = 5.0) const;

            //-----------------------------------------
            //! Reproject the 3D point observed in the last frame to the current frame and record the corresponding information in frame.landmarks_
            // used in frame::tracker::motion_based_track()
            unsigned int match_current_and_last_frames(data::frame &curr_frm, const data::frame &last_frm, const float margin) const;

            // FW:
            // find the 2D and 3D line correspondence
            // used in frame::tracker::motion_based_track()
            unsigned int match_current_and_last_frames_line(data::frame &curr_frm, const data::frame &last_frm, const float margin) const;

            //-----------------------------------------
            //! Reproject the 3D point observed by keyfarme to the current frame and record the corresponding information in frame.landmarks_
            //! If the current frame already corresponds, specify already_matched_lms so that it will not be reprojected.
            // used in see -> relocalizer::relocalize()
            unsigned int match_frame_and_keyframe(data::frame &curr_frm, data::keyframe *keyfrm, const std::set<data::landmark *> &already_matched_lms,
                                                  const float margin, const unsigned int hamm_dist_thr) const;

            // FW:
            // find the 2D and 3D line correspondence
            // used in see -> relocalizer::relocalize()
            unsigned int match_frame_and_keyframe_line(data::frame &curr_frm, data::keyframe *keyfrm, const std::set<data::Line *> &already_matched_lms,
                                                       const float margin, const unsigned int hamm_dist_thr) const;

            //-----------------------------------------
            //! After converting the coordinates of the 3D point with Sim3, reproject it to the keyframe and record the corresponding information in matched_lms_in_keyfrm.
            //! If the corresponding information is already recorded in matched_lms_in_keyfrm, it is excluded from the search.
            //! (NOTE: keyframe feature score and matched_lms_in_keyfrm.size () match)
            // used in loop_detector::validate_candidates()
            unsigned int match_by_Sim3_transform(data::keyframe *keyfrm, const Mat44_t &Sim3_cw, const std::vector<data::landmark *> &landmarks,
                                                 std::vector<data::landmark *> &matched_lms_in_keyfrm, const float margin) const;

            //! Using the specified Sim3, convert and reproject the 3D points observed in each keyframe to the other keyframe, and find the corresponding points.
            //! matched_lms_in_keyfrm_1 records the 3D points observed in keyframe2, which correspond to the feature points (index) in keyframe1.
            // used in loop_detector::select_loop_candidate_via_Sim3()
            unsigned int match_keyframes_mutually(data::keyframe *keyfrm_1, data::keyframe *keyfrm_2, std::vector<data::landmark *> &matched_lms_in_keyfrm_1,
                                                  const float &s_12, const Mat33_t &rot_12, const Vec3_t &trans_12, const float margin) const;
        };

    } // namespace match
} // namespace PLPSLAM

#endif // PLPSLAM_MATCH_PROJECTION_H
