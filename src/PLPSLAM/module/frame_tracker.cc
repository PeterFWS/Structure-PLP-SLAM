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

#include "PLPSLAM/camera/base.h"
#include "PLPSLAM/data/frame.h"
#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/match/bow_tree.h"
#include "PLPSLAM/match/projection.h"
#include "PLPSLAM/match/robust.h"
#include "PLPSLAM/module/frame_tracker.h"

#include "PLPSLAM/camera/perspective.h"
#include "PLPSLAM/solve/triangulator.h"

#include "PLPSLAM/data/landmark_line.h"

#include <opencv2/features2d.hpp>
#include "opencv2/highgui.hpp"
#include <spdlog/spdlog.h>

namespace PLPSLAM
{
    namespace module
    {
        frame_tracker::frame_tracker(camera::base *camera, const unsigned int num_matches_thr)
            : camera_(camera), num_matches_thr_(num_matches_thr), pose_optimizer_(), _pose_optimizer_extended_line()
        {
        }

        // FW: around 4 ms for tracking
        bool frame_tracker::motion_based_track(data::frame &curr_frm, const data::frame &last_frm, const Mat44_t &velocity) const
        {
            // auto before = std::chrono::high_resolution_clock::now();

            match::projection projection_matcher(0.9, true);

            // Set the initial value of pose using motion model
            curr_frm.set_cam_pose(velocity * last_frm.cam_pose_cw_);

            // Initialize 2D-3D support
            std::fill(curr_frm.landmarks_.begin(), curr_frm.landmarks_.end(), nullptr);

            // Reproject the 3D point visible in the last frame to find 2D-3D support
            const float margin = (camera_->setup_type_ != camera::setup_type_t::Stereo) ? 20 : 10;
            auto num_matches = projection_matcher.match_current_and_last_frames(curr_frm, last_frm, margin);

            if (num_matches < num_matches_thr_)
            {
                // Widen the margin and search again
                std::fill(curr_frm.landmarks_.begin(), curr_frm.landmarks_.end(), nullptr);
                num_matches = projection_matcher.match_current_and_last_frames(curr_frm, last_frm, 2 * margin);
            }

            if (num_matches < num_matches_thr_)
            {
                spdlog::debug("motion based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
                return false;
            }

            // FW: Motion-only BA after enough 3D-2D correspondences are found
            // pose optimization -> update pose
            if (_b_use_line_tracking)
            {
                // FW: need to find also the 2D-3D line support
                std::fill(curr_frm._landmarks_line.begin(), curr_frm._landmarks_line.end(), nullptr);
                unsigned int num_matches_line = projection_matcher.match_current_and_last_frames_line(curr_frm, last_frm, 20);

                // spdlog::info("motion based tracking, num_matches_line: {}", num_matches_line);

                _pose_optimizer_extended_line.optimize(curr_frm);
            }
            else
            {
                pose_optimizer_.optimize(curr_frm);
            }

            // Excluding outliers
            const auto num_valid_matches = discard_outliers(curr_frm);

            // FW: also exclude outlier of line
            if (_b_use_line_tracking)
            {
                const auto num_valid_matches_line = discard_outliers_line(curr_frm);
                // spdlog::info("motion based tracking, num_valid_matches_line: {}", num_valid_matches_line);
            }

            // spdlog::info("num_valid_matches_line: {}", num_valid_matches_line);

            if (num_valid_matches < num_matches_thr_)
            {
                spdlog::debug("motion based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
                return false;
            }
            else
            {
                // spdlog::info("Motion-model based tracking");
                // auto after = std::chrono::high_resolution_clock::now();
                // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count();
                // spdlog::info("\t \t | process time (Motion-model based tracking): {}ms", duration);

                return true;
            }
        }

        // FW: around 7 ms for tracking
        bool frame_tracker::bow_match_based_track(data::frame &curr_frm, const data::frame &last_frm, data::keyframe *ref_keyfrm) const
        {
            // auto before = std::chrono::high_resolution_clock::now();
            match::bow_tree bow_matcher(0.7, true);

            // Since BoW match is performed, calculate BoW
            curr_frm.compute_bow();

            // Search for 2D correspondence between keyframe and frame,
            // and obtain the correspondence between the feature points of frame and the 3D points observed by keyframe.
            std::vector<data::landmark *> matched_lms_in_curr;
            auto num_matches = bow_matcher.match_frame_and_keyframe(ref_keyfrm, curr_frm, matched_lms_in_curr);

            if (num_matches < num_matches_thr_)
            {
                spdlog::debug("bow match based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
                return false;
            }

            // Updated 2D-3D support information
            curr_frm.landmarks_ = matched_lms_in_curr;

            // pose optimization
            // The initial value is the pose of the previous frame
            curr_frm.set_cam_pose(last_frm.cam_pose_cw_);

            // FW: in BoW-based tracking case, we search 3D-2D line
            if (_b_use_line_tracking)
            {
                match::projection projection_matcher(0.9, true);
                std::fill(curr_frm._landmarks_line.begin(), curr_frm._landmarks_line.end(), nullptr);
                unsigned int num_matches_line = projection_matcher.match_current_and_last_frames_line(curr_frm, last_frm, 20);

                // spdlog::info("bow match based tracking, num_matches_line: {}", num_matches_line);
            }

            pose_optimizer_.optimize(curr_frm);

            // Excluding outliers
            const auto num_valid_matches = discard_outliers(curr_frm);

            // FW: also exclude outlier of line
            if (_b_use_line_tracking)
            {
                const auto num_valid_matches_line = discard_outliers_line(curr_frm);
                // spdlog::info("bow match based tracking, num_valid_matches_line: {}", num_valid_matches_line);
            }

            if (num_valid_matches < num_matches_thr_)
            {
                spdlog::debug("bow match based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
                return false;
            }
            else
            {
                // spdlog::info("BoW-match based tracking");
                // auto after = std::chrono::high_resolution_clock::now();
                // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count();
                // spdlog::info("\t \t | process time (BoW-match based tracking): {}ms", duration);

                return true;
            }
        }

        // FW: around 40 ms, 10 times slower than motion-model based tracking
        bool frame_tracker::robust_match_based_track(data::frame &curr_frm, const data::frame &last_frm, data::keyframe *ref_keyfrm) const
        {
            // auto before = std::chrono::high_resolution_clock::now();

            match::robust robust_matcher(0.8, false);

            // Search for 2D correspondence between keyframe and frame,
            // and obtain the correspondence between the feature points of frame and the 3D points observed by keyframe.
            std::vector<data::landmark *> matched_lms_in_curr;
            auto num_matches = robust_matcher.match_frame_and_keyframe(curr_frm, ref_keyfrm, matched_lms_in_curr);

            if (num_matches < num_matches_thr_)
            {
                spdlog::debug("robust match based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
                return false;
            }

            // Updated 2D-3D support information
            curr_frm.landmarks_ = matched_lms_in_curr;

            // pose optimization
            // The initial value is the pose of the previous frame
            curr_frm.set_cam_pose(last_frm.cam_pose_cw_);

            // FW: in robust-match-based tracking case, we search 3D-2D line correspondence
            if (_b_use_line_tracking)
            {
                match::projection projection_matcher(0.9, true);
                std::fill(curr_frm._landmarks_line.begin(), curr_frm._landmarks_line.end(), nullptr);
                unsigned int num_matches_line = projection_matcher.match_current_and_last_frames_line(curr_frm, last_frm, 20);

                // spdlog::info("robust match based tracking, num_matches_line: {}", num_matches_line);
            }

            pose_optimizer_.optimize(curr_frm);

            // Excluding outliers
            const auto num_valid_matches = discard_outliers(curr_frm);

            // FW: also exclude outlier of line
            if (_b_use_line_tracking)
            {
                const auto num_valid_matches_line = discard_outliers_line(curr_frm);
            }

            if (num_valid_matches < num_matches_thr_)
            {
                spdlog::debug("robust match based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
                return false;
            }
            else
            {
                // spdlog::info("Robust-match based tracking");
                // auto after = std::chrono::high_resolution_clock::now();
                // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count();
                // spdlog::info("\t \t | process time (Robust-match based tracking): {}ms", duration);

                return true;
            }
        }

        // FW:
        void frame_tracker::set_using_line_tracking()
        {
            _b_use_line_tracking = true;
        }

        unsigned int frame_tracker::discard_outliers(data::frame &curr_frm) const
        {
            unsigned int num_valid_matches = 0;

            for (unsigned int idx = 0; idx < curr_frm.num_keypts_; ++idx)
            {
                if (!curr_frm.landmarks_.at(idx))
                {
                    continue;
                }

                auto lm = curr_frm.landmarks_.at(idx);

                if (curr_frm.outlier_flags_.at(idx))
                {
                    curr_frm.landmarks_.at(idx) = nullptr;
                    curr_frm.outlier_flags_.at(idx) = false;
                    lm->is_observable_in_tracking_ = false;
                    lm->identifier_in_local_lm_search_ = curr_frm.id_;
                    continue;
                }

                ++num_valid_matches;
            }

            return num_valid_matches;
        }

        // FW:
        unsigned int frame_tracker::discard_outliers_line(data::frame &curr_frm) const
        {
            unsigned int num_valid_matches = 0;

            for (unsigned int idx = 0; idx < curr_frm._num_keylines; ++idx)
            {
                if (!curr_frm._landmarks_line.at(idx))
                {
                    continue;
                }

                auto lm_line = curr_frm._landmarks_line.at(idx);

                if (curr_frm._outlier_flags_line.at(idx))
                {
                    curr_frm._landmarks_line.at(idx) = nullptr;
                    curr_frm._outlier_flags_line.at(idx) = false;
                    lm_line->_is_observable_in_tracking = false;
                    lm_line->_identifier_in_local_lm_search = curr_frm.id_;
                    continue;
                }

                ++num_valid_matches;
            }

            return num_valid_matches;
        }

    } // namespace module
} // namespace PLPSLAM
