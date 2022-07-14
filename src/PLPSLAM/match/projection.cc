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
#include "PLPSLAM/data/landmark_line.h"
#include "PLPSLAM/match/projection.h"
#include "PLPSLAM/match/angle_checker.h"

namespace PLPSLAM
{
    namespace match
    {

        unsigned int projection::match_frame_and_landmarks(data::frame &frm, const std::vector<data::landmark *> &local_landmarks, const float margin) const
        {
            unsigned int num_matches = 0;

            for (auto local_lm : local_landmarks)
            {
                if (!local_lm->is_observable_in_tracking_)
                {
                    continue;
                }
                if (local_lm->will_be_erased())
                {
                    continue;
                }

                const auto pred_scale_level = local_lm->scale_level_in_tracking_;

                // Get the feature point of the cell where the point that reprojected the 3D point exists
                const auto indices_in_cell = frm.get_keypoints_in_cell(local_lm->reproj_in_tracking_(0), local_lm->reproj_in_tracking_(1),
                                                                       margin * frm.scale_factors_.at(pred_scale_level),
                                                                       pred_scale_level - 1, pred_scale_level);
                if (indices_in_cell.empty())
                {
                    continue;
                }

                const cv::Mat lm_desc = local_lm->get_descriptor();

                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                int best_scale_level = -1;
                unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;
                int second_best_scale_level = -1;
                int best_idx = -1;

                for (const auto idx : indices_in_cell)
                {
                    if (frm.landmarks_.at(idx) && frm.landmarks_.at(idx)->has_observation())
                    {
                        continue;
                    }

                    if (0 < frm.stereo_x_right_.at(idx))
                    {
                        const auto reproj_error = std::abs(local_lm->x_right_in_tracking_ - frm.stereo_x_right_.at(idx));
                        if (margin * frm.scale_factors_.at(pred_scale_level) < reproj_error)
                        {
                            continue;
                        }
                    }

                    const cv::Mat &desc = frm.descriptors_.row(idx);

                    const auto dist = compute_descriptor_distance_32(lm_desc, desc);

                    if (dist < best_hamm_dist)
                    {
                        second_best_hamm_dist = best_hamm_dist;
                        best_hamm_dist = dist;
                        second_best_scale_level = best_scale_level;
                        best_scale_level = frm.undist_keypts_.at(idx).octave;
                        best_idx = idx;
                    }
                    else if (dist < second_best_hamm_dist)
                    {
                        second_best_scale_level = frm.undist_keypts_.at(idx).octave;
                        second_best_hamm_dist = dist;
                    }
                }

                if (best_hamm_dist <= HAMMING_DIST_THR_HIGH)
                {
                    // lowe's ratio test
                    if (best_scale_level == second_best_scale_level && best_hamm_dist > lowe_ratio_ * second_best_hamm_dist)
                    {
                        continue;
                    }

                    // Add support information
                    frm.landmarks_.at(best_idx) = local_lm;
                    ++num_matches;
                }
            }

            return num_matches;
        }

        // FW:
        unsigned int projection::match_frame_and_landmarks_line(data::frame &frm,
                                                                const std::vector<data::Line *> &local_landmarks_line,
                                                                const float margin) const
        {
            unsigned int num_matches = 0;

            if (local_landmarks_line.empty())
            {
                return 0;
            }

            for (auto local_lm_line : local_landmarks_line)
            {
                if (!local_lm_line->_is_observable_in_tracking)
                {
                    continue;
                }
                if (local_lm_line->will_be_erased())
                {
                    continue;
                }

                const auto pred_scale_level = local_lm_line->_scale_level_in_tracking;

                // Get the feature line of the cell where the line that reprojected the 3D line exists
                const auto indices_in_cell = frm.get_keylines_in_cell(local_lm_line->_reproj_in_tracking_sp(0),
                                                                      local_lm_line->_reproj_in_tracking_sp(1),
                                                                      local_lm_line->_reproj_in_tracking_ep(0),
                                                                      local_lm_line->_reproj_in_tracking_ep(1),
                                                                      margin * frm._scale_factors_lsd.at(pred_scale_level),
                                                                      pred_scale_level - 1, pred_scale_level);

                if (indices_in_cell.empty())
                {
                    continue;
                }

                // std::cout << "indices_in_cell number: " << indices_in_cell.size() << std::endl;

                const cv::Mat lm_line_desc = local_lm_line->get_descriptor();

                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                int best_scale_level = -1;
                unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;
                int second_best_scale_level = -1;
                int best_idx = -1;

                for (const auto idx : indices_in_cell)
                {
                    if (frm._landmarks_line.at(idx) && frm._landmarks_line.at(idx)->has_observation())
                    {
                        continue;
                    }

                    const cv::Mat &desc = frm._lbd_descr.row(idx);

                    const auto dist = compute_descriptor_distance_32(lm_line_desc, desc);

                    if (dist < best_hamm_dist)
                    {
                        second_best_hamm_dist = best_hamm_dist;
                        best_hamm_dist = dist;
                        second_best_scale_level = best_scale_level;
                        best_scale_level = frm.undist_keypts_.at(idx).octave;
                        best_idx = idx;
                    }
                    else if (dist < second_best_hamm_dist)
                    {
                        second_best_scale_level = frm.undist_keypts_.at(idx).octave;
                        second_best_hamm_dist = dist;
                    }
                }

                if (best_hamm_dist <= HAMMING_DIST_THR_HIGH)
                {
                    // lowe's ratio test
                    if (best_scale_level == second_best_scale_level && best_hamm_dist > lowe_ratio_ * second_best_hamm_dist)
                    {
                        continue;
                    }

                    // Add support information
                    frm._landmarks_line.at(best_idx) = local_lm_line;
                    ++num_matches;
                }
            }

            return num_matches;
        }

        unsigned int projection::match_current_and_last_frames(data::frame &curr_frm, const data::frame &last_frm, const float margin) const
        {
            unsigned int num_matches = 0;

            angle_checker<int> angle_checker;

            const Mat33_t rot_cw = curr_frm.cam_pose_cw_.block<3, 3>(0, 0);
            const Vec3_t trans_cw = curr_frm.cam_pose_cw_.block<3, 1>(0, 3);

            const Vec3_t trans_wc = -rot_cw.transpose() * trans_cw;

            const Mat33_t rot_lw = last_frm.cam_pose_cw_.block<3, 3>(0, 0);
            const Vec3_t trans_lw = last_frm.cam_pose_cw_.block<3, 1>(0, 3);

            const Vec3_t trans_lc = rot_lw * trans_wc + trans_lw;

            // If it is not monocular, it is judged whether it is advancing or not by the z component of the translation vector of current-> last.
            // The z component is swinging positively-> moving forward
            const bool assume_forward = (curr_frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                            ? false
                                            : trans_lc(2) > curr_frm.camera_->true_baseline_;
            // z component swings negative-> recedes
            const bool assume_backward = (curr_frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                             ? false
                                             : -trans_lc(2) > curr_frm.camera_->true_baseline_;

            // Back-project the 3D point that corresponds to the feature point of the last frame onto the current frame to find the correspondence.
            for (unsigned int idx_last = 0; idx_last < last_frm.num_keypts_; ++idx_last)
            {
                auto *lm = last_frm.landmarks_.at(idx_last);
                // Not compatible with 3D points
                if (!lm)
                {
                    continue;
                }

                // Does not correspond to the outlier in pose optimization
                if (last_frm.outlier_flags_.at(idx_last))
                {
                    continue;
                }

                // Global standard 3D point coordinates
                const Vec3_t pos_w = lm->get_pos_in_world();

                // Reproject for visibility
                Vec2_t reproj;
                float x_right;
                const bool in_image = curr_frm.camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);

                // Through if reprojected outside the image
                if (!in_image)
                {
                    continue;
                }

                // The search range is set assuming that the scale of the corresponding feature points is constant between adjacent frames.
                const auto last_scale_level = last_frm.keypts_.at(idx_last).octave;

                // Acquire the feature point of the cell where the point that reprojected the 3D point exists
                std::vector<unsigned int> indices;
                if (assume_forward)
                {
                    indices = curr_frm.get_keypoints_in_cell(reproj(0), reproj(1),
                                                             margin * curr_frm.scale_factors_.at(last_scale_level),
                                                             last_scale_level, last_frm.num_scale_levels_ - 1);
                }
                else if (assume_backward)
                {
                    indices = curr_frm.get_keypoints_in_cell(reproj(0), reproj(1),
                                                             margin * curr_frm.scale_factors_.at(last_scale_level),
                                                             0, last_scale_level);
                }
                else
                { // FW: neither forward nor backwork?
                    indices = curr_frm.get_keypoints_in_cell(reproj(0), reproj(1),
                                                             margin * curr_frm.scale_factors_.at(last_scale_level),
                                                             last_scale_level - 1, last_scale_level + 1);
                }

                if (indices.empty())
                {
                    continue;
                }

                const auto lm_desc = lm->get_descriptor();

                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                int best_idx = -1;

                for (const auto curr_idx : indices)
                {
                    if (curr_frm.landmarks_.at(curr_idx) && curr_frm.landmarks_[curr_idx]->has_observation())
                    {
                        continue;
                    }

                    if (curr_frm.stereo_x_right_.at(curr_idx) > 0)
                    {
                        const float reproj_error = std::fabs(x_right - curr_frm.stereo_x_right_.at(curr_idx));
                        if (margin * curr_frm.scale_factors_.at(last_scale_level) < reproj_error)
                        {
                            continue;
                        }
                    }

                    const auto &desc = curr_frm.descriptors_.row(curr_idx);

                    const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

                    if (hamm_dist < best_hamm_dist)
                    {
                        best_hamm_dist = hamm_dist;
                        best_idx = curr_idx;
                    }
                }

                if (HAMMING_DIST_THR_HIGH < best_hamm_dist)
                {
                    continue;
                }

                // Make it a valid matching
                curr_frm.landmarks_.at(best_idx) = lm;
                ++num_matches;

                if (check_orientation_)
                {
                    const auto delta_angle = last_frm.undist_keypts_.at(idx_last).angle - curr_frm.undist_keypts_.at(best_idx).angle;
                    angle_checker.append_delta_angle(delta_angle, best_idx);
                }
            }

            if (check_orientation_)
            {
                const auto invalid_matches = angle_checker.get_invalid_matches();
                for (const auto invalid_idx : invalid_matches)
                {
                    curr_frm.landmarks_.at(invalid_idx) = nullptr;
                    --num_matches;
                }
            }

            return num_matches;
        }

        // FW:
        unsigned int projection::match_current_and_last_frames_line(data::frame &curr_frm, const data::frame &last_frm, const float margin) const
        {
            unsigned int num_matches = 0;

            const Mat33_t rot_cw = curr_frm.cam_pose_cw_.block<3, 3>(0, 0);
            const Vec3_t trans_cw = curr_frm.cam_pose_cw_.block<3, 1>(0, 3);

            const Vec3_t trans_wc = -rot_cw.transpose() * trans_cw;

            const Mat33_t rot_lw = last_frm.cam_pose_cw_.block<3, 3>(0, 0);
            const Vec3_t trans_lw = last_frm.cam_pose_cw_.block<3, 1>(0, 3);

            const Vec3_t trans_lc = rot_lw * trans_wc + trans_lw;

            // If it is not monocular, it is judged whether it is advancing or not by the z component of the translation vector of current-> last.
            // The z component is swinging positively-> moving forward
            const bool assume_forward = (curr_frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                            ? false
                                            : trans_lc(2) > curr_frm.camera_->true_baseline_;
            // z component swings negative-> recedes
            const bool assume_backward = (curr_frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                             ? false
                                             : -trans_lc(2) > curr_frm.camera_->true_baseline_;

            // Back-project the 3D line that corresponds to the feature lines of the last frame onto the current frame to find the correspondence.
            for (unsigned int idx_last = 0; idx_last < last_frm._num_keylines; ++idx_last)
            {
                auto *lm_line = last_frm._landmarks_line.at(idx_last);

                // Not compatible with 3D lines
                if (!lm_line)
                {
                    continue;
                }

                // Does not correspond to the outlier in pose optimization
                if (last_frm._outlier_flags_line.at(idx_last))
                {
                    continue;
                }

                // Global standard endpoints coordinates
                const Vec6_t pos_w = lm_line->get_pos_in_world();

                // Reproject for visibility
                Vec2_t reproj_sp;
                float x_right_sp;
                const bool in_image_sp = curr_frm.camera_->reproject_to_image(rot_cw, trans_cw, pos_w.head<3>(), reproj_sp, x_right_sp);

                Vec2_t reproj_ep;
                float x_right_ep;
                const bool in_image_ep = curr_frm.camera_->reproject_to_image(rot_cw, trans_cw, pos_w.tail<3>(), reproj_ep, x_right_ep);

                // Through if reprojected outside the image
                if (!in_image_sp && !in_image_ep)
                {
                    continue;
                }

                // check if the 3D line is partially occluded
                bool partial_occlusion = false;
                if (!in_image_sp || !in_image_ep)
                {
                    // check if the middle point are within the image frustum
                    Vec3_t pos_w_mp = 0.5 * (pos_w.head<3>() + pos_w.tail<3>());
                    Vec2_t reproj_mp;
                    float x_right_mp;
                    const bool in_image_mp = curr_frm.camera_->reproject_to_image(rot_cw, trans_cw, pos_w_mp, reproj_mp, x_right_mp);

                    if (in_image_mp)
                    {
                        // partially occluded is ok, it would be good, if the middle point is visible
                        partial_occlusion = true;
                    }
                    else
                    {
                        // if the middle point is not even visible, discard this landmark
                        continue;
                    }
                }

                // The search range is set assuming that the scale of the corresponding feature points is constant between adjacent frames.
                const auto last_scale_level = last_frm._keylsd.at(idx_last).octave;

                // std::cout << "last_scale_level: " << last_scale_level << std::endl;                                 // 0
                // std::cout << "last_frm.num_scale_levels_ - 1: " << last_frm._num_scale_levels_lsd - 1 << std::endl; // 0
                // std::cout << std::endl;

                // Acquire the feature point of the cell where the point that reprojected the 3D point exists
                std::vector<unsigned int> indices;
                if (assume_forward)
                {
                    indices = curr_frm.get_keylines_in_cell(reproj_sp(0), reproj_sp(1),
                                                            reproj_ep(0), reproj_ep(1),
                                                            margin * curr_frm._scale_factors_lsd.at(last_scale_level),
                                                            last_scale_level, last_frm._num_scale_levels_lsd);
                }
                else if (assume_backward)
                {
                    indices = curr_frm.get_keylines_in_cell(reproj_sp(0), reproj_sp(1),
                                                            reproj_ep(0), reproj_ep(1),
                                                            margin * curr_frm._scale_factors_lsd.at(last_scale_level),
                                                            0, last_scale_level + 1);
                }
                else
                { // FW: neither forward nor backwork?
                    indices = curr_frm.get_keylines_in_cell(reproj_sp(0), reproj_sp(1),
                                                            reproj_ep(0), reproj_ep(1),
                                                            margin * curr_frm._scale_factors_lsd.at(last_scale_level),
                                                            last_scale_level - 1, last_scale_level + 1);
                }

                if (indices.empty())
                {
                    continue;
                }

                const auto lm_line_desc = lm_line->get_descriptor();

                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                int best_idx = -1;

                for (const auto curr_idx : indices)
                {
                    if (curr_frm._landmarks_line.at(curr_idx) && curr_frm._landmarks_line[curr_idx]->has_observation())
                    {
                        continue;
                    }

                    if (curr_frm.camera_->setup_type_ == camera::setup_type_t::RGBD)
                    {
                        if (curr_frm._stereo_x_right_cooresponding_to_keylines.at(curr_idx).first > 0 &&
                            curr_frm._stereo_x_right_cooresponding_to_keylines.at(curr_idx).second > 0)
                        {
                            const float reproj_error_sp = std::fabs(x_right_sp - curr_frm._stereo_x_right_cooresponding_to_keylines.at(curr_idx).first);
                            const float reproj_error_ep = std::fabs(x_right_ep - curr_frm._stereo_x_right_cooresponding_to_keylines.at(curr_idx).second);
                            if (margin * curr_frm._scale_factors_lsd.at(last_scale_level) < reproj_error_sp ||
                                margin * curr_frm._scale_factors_lsd.at(last_scale_level) < reproj_error_ep)
                            {
                                continue;
                            }
                        }
                    }

                    const auto &desc = curr_frm._lbd_descr.row(curr_idx);

                    const auto hamm_dist = compute_descriptor_distance_32(lm_line_desc, desc);

                    if (hamm_dist < best_hamm_dist)
                    {
                        best_hamm_dist = hamm_dist;
                        best_idx = curr_idx;
                    }
                }

                if (HAMMING_DIST_THR_HIGH < best_hamm_dist)
                {
                    continue;
                }

                // Make it a valid matching
                curr_frm._landmarks_line.at(best_idx) = lm_line;
                ++num_matches;
            }

            return num_matches;
        }

        unsigned int projection::match_frame_and_keyframe(data::frame &curr_frm, data::keyframe *keyfrm, const std::set<data::landmark *> &already_matched_lms,
                                                          const float margin, const unsigned int hamm_dist_thr) const
        {
            unsigned int num_matches = 0;

            angle_checker<int> angle_checker;

            const Mat33_t rot_cw = curr_frm.cam_pose_cw_.block<3, 3>(0, 0);
            const Vec3_t trans_cw = curr_frm.cam_pose_cw_.block<3, 1>(0, 3);
            const Vec3_t cam_center = -rot_cw.transpose() * trans_cw;

            const auto landmarks = keyfrm->get_landmarks();

            for (unsigned int idx = 0; idx < landmarks.size(); idx++)
            {
                auto *lm = landmarks.at(idx);
                if (!lm)
                {
                    continue;
                }
                if (lm->will_be_erased())
                {
                    continue;
                }
                if (already_matched_lms.count(lm))
                {
                    continue;
                }

                // Global standard 3D point coordinates
                const Vec3_t pos_w = lm->get_pos_in_world();

                // Reproject for visibility
                Vec2_t reproj;
                float x_right;
                const bool in_image = curr_frm.camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);

                // Through if reprojected outside the image
                if (!in_image)
                {
                    continue;
                }

                // Make sure it is within the ORB scale
                const Vec3_t cam_to_lm_vec = pos_w - cam_center;
                const auto cam_to_lm_dist = cam_to_lm_vec.norm();
                const auto max_cam_to_lm_dist = lm->get_max_valid_distance();
                const auto min_cam_to_lm_dist = lm->get_min_valid_distance();

                if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist)
                {
                    continue;
                }

                // Get the feature point of the cell where the point that reprojected the 3D point exists
                const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, &curr_frm);

                const auto indices = curr_frm.get_keypoints_in_cell(reproj(0), reproj(1),
                                                                    margin * curr_frm.scale_factors_.at(pred_scale_level),
                                                                    pred_scale_level - 1, pred_scale_level + 1);

                if (indices.empty())
                {
                    continue;
                }

                const auto lm_desc = lm->get_descriptor();

                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                int best_idx = -1;

                for (unsigned long curr_idx : indices)
                {
                    if (curr_frm.landmarks_.at(curr_idx))
                    {
                        continue;
                    }

                    const auto &desc = curr_frm.descriptors_.row(curr_idx);

                    const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

                    if (hamm_dist < best_hamm_dist)
                    {
                        best_hamm_dist = hamm_dist;
                        best_idx = curr_idx;
                    }
                }

                // Make it a valid matching
                if (hamm_dist_thr < best_hamm_dist)
                {
                    continue;
                }

                curr_frm.landmarks_.at(best_idx) = lm;
                num_matches++;

                if (check_orientation_)
                {
                    const auto delta_angle = keyfrm->undist_keypts_.at(idx).angle - curr_frm.undist_keypts_.at(best_idx).angle;
                    angle_checker.append_delta_angle(delta_angle, best_idx);
                }
            }

            if (check_orientation_)
            {
                const auto invalid_matches = angle_checker.get_invalid_matches();
                for (const auto invalid_idx : invalid_matches)
                {
                    curr_frm.landmarks_.at(invalid_idx) = nullptr;
                    --num_matches;
                }
            }

            return num_matches;
        }

        // FW:
        unsigned int projection::match_frame_and_keyframe_line(data::frame &curr_frm,
                                                               data::keyframe *keyfrm,
                                                               const std::set<data::Line *> &already_matched_lms,
                                                               const float margin,
                                                               const unsigned int hamm_dist_thr) const
        {
            unsigned int num_matches = 0;

            const Mat33_t rot_cw = curr_frm.cam_pose_cw_.block<3, 3>(0, 0);
            const Vec3_t trans_cw = curr_frm.cam_pose_cw_.block<3, 1>(0, 3);
            const Vec3_t cam_center = -rot_cw.transpose() * trans_cw;

            const auto landmarks_line = keyfrm->get_landmarks_line();

            for (unsigned int idx = 0; idx < landmarks_line.size(); idx++)
            {
                auto *lm_line = landmarks_line.at(idx);
                if (!lm_line)
                {
                    continue;
                }
                if (lm_line->will_be_erased())
                {
                    continue;
                }
                if (already_matched_lms.count(lm_line))
                {
                    continue;
                }

                // Global standard endpoints coordinates
                const Vec6_t pos_w = lm_line->get_pos_in_world();

                // Reproject for visibility
                Vec2_t reproj_sp;
                float x_right_sp;
                const bool in_image_sp = curr_frm.camera_->reproject_to_image(rot_cw, trans_cw, pos_w.head<3>(), reproj_sp, x_right_sp);

                Vec2_t reproj_ep;
                float x_right_ep;
                const bool in_image_ep = curr_frm.camera_->reproject_to_image(rot_cw, trans_cw, pos_w.tail<3>(), reproj_ep, x_right_ep);

                // Through if reprojected outside the image
                if (!in_image_sp && !in_image_ep)
                {
                    continue;
                }

                Vec3_t pos_w_sp = pos_w.head(3);
                Vec3_t pos_w_ep = pos_w.tail(3);

                // check if the 3D line is partially occluded
                bool partial_occlusion = false;
                if (!in_image_sp || !in_image_ep)
                {
                    // check if the middle point are within the image frustum
                    Vec3_t pos_w_mp = 0.5 * (pos_w_sp + pos_w_ep);
                    Vec2_t reproj_mp;
                    float x_right_mp;
                    const bool in_image_mp = curr_frm.camera_->reproject_to_image(rot_cw, trans_cw, pos_w_mp, reproj_mp, x_right_mp);

                    if (in_image_mp)
                    {
                        // partially occluded is ok, it would be good, if the middle point is visible
                        partial_occlusion = true;
                    }
                    else
                    {
                        // if the middle point is not even visible, discard this landmark
                        continue;
                    }
                }

                // Make sure it is within the scale
                const Vec3_t cam_to_lm_vec = 0.5 * (pos_w_sp + pos_w_ep) - cam_center;
                const auto cam_to_lm_dist = cam_to_lm_vec.norm();
                const auto max_cam_to_lm_dist = lm_line->get_max_valid_distance();
                const auto min_cam_to_lm_dist = lm_line->get_min_valid_distance();

                if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist)
                {
                    continue;
                }

                // Get the feature line of the cell where the line that reprojected the 3D line exists
                const auto pred_scale_level = lm_line->predict_scale_level(cam_to_lm_dist, curr_frm._log_scale_factor_lsd, curr_frm._num_scale_levels_lsd);

                const auto indices = curr_frm.get_keylines_in_cell(reproj_sp(0), reproj_sp(1),
                                                                   reproj_ep(0), reproj_ep(1),
                                                                   margin * curr_frm._scale_factors_lsd.at(pred_scale_level),
                                                                   pred_scale_level - 1, pred_scale_level + 1);

                if (indices.empty())
                {
                    continue;
                }

                const auto lm_line_desc = lm_line->get_descriptor();

                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                int best_idx = -1;

                for (unsigned long curr_idx : indices)
                {
                    if (curr_frm._landmarks_line.at(curr_idx))
                    {
                        continue;
                    }

                    const auto &desc = curr_frm._lbd_descr.row(curr_idx);

                    const auto hamm_dist = compute_descriptor_distance_32(lm_line_desc, desc);

                    if (hamm_dist < best_hamm_dist)
                    {
                        best_hamm_dist = hamm_dist;
                        best_idx = curr_idx;
                    }
                }

                // Make it a valid matching
                if (hamm_dist_thr < best_hamm_dist)
                {
                    continue;
                }

                curr_frm._landmarks_line.at(best_idx) = lm_line;
                num_matches++;
            }

            return num_matches;
        }

        unsigned int projection::match_by_Sim3_transform(data::keyframe *keyfrm, const Mat44_t &Sim3_cw, const std::vector<data::landmark *> &landmarks,
                                                         std::vector<data::landmark *> &matched_lms_in_keyfrm, const float margin) const
        {
            unsigned int num_matches = 0;

            // Disassemble Sim3 into SE3
            const Mat33_t s_rot_cw = Sim3_cw.block<3, 3>(0, 0);
            const auto s_cw = std::sqrt(s_rot_cw.block<1, 3>(0, 0).dot(s_rot_cw.block<1, 3>(0, 0)));
            const Mat33_t rot_cw = s_rot_cw / s_cw;
            const Vec3_t trans_cw = Sim3_cw.block<3, 1>(0, 3) / s_cw;
            const Vec3_t cam_center = -rot_cw.transpose() * trans_cw;

            std::set<data::landmark *> already_matched(matched_lms_in_keyfrm.begin(), matched_lms_in_keyfrm.end());
            already_matched.erase(static_cast<data::landmark *>(nullptr));

            for (auto lm : landmarks)
            {
                if (lm->will_be_erased())
                {
                    continue;
                }
                if (already_matched.count(lm))
                {
                    continue;
                }

                // Global standard 3D point coordinates
                const Vec3_t pos_w = lm->get_pos_in_world();

                // Reproject for visibility
                Vec2_t reproj;
                float x_right;
                const bool in_image = keyfrm->camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);

                // Through if reprojected outside the image
                if (!in_image)
                {
                    continue;
                }

                // Make sure it is within the ORB scale
                const Vec3_t cam_to_lm_vec = pos_w - cam_center;
                const auto cam_to_lm_dist = cam_to_lm_vec.norm();
                const auto max_cam_to_lm_dist = lm->get_max_valid_distance();
                const auto min_cam_to_lm_dist = lm->get_min_valid_distance();

                if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist)
                {
                    continue;
                }

                // Calculate the angle of the 3D point with the average observation vector, and discard if it is larger than the threshold (60deg).
                const Vec3_t obs_mean_normal = lm->get_obs_mean_normal();

                if (cam_to_lm_vec.dot(obs_mean_normal) < 0.5 * cam_to_lm_dist)
                {
                    continue;
                }

                // Get the feature point of the cell where the point that reprojected the 3D point exists
                const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, keyfrm);

                const auto indices = keyfrm->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm->scale_factors_.at(pred_scale_level));

                if (indices.empty())
                {
                    continue;
                }

                // Find the feature point with the closest descriptor
                const auto lm_desc = lm->get_descriptor();

                unsigned int best_dist = MAX_HAMMING_DIST;
                int best_idx = -1;

                for (const auto idx : indices)
                {
                    if (matched_lms_in_keyfrm.at(idx))
                    {
                        continue;
                    }

                    const auto scale_level = static_cast<unsigned int>(keyfrm->keypts_.at(idx).octave);

                    // TODO: Use keyfrm-> get_keypts_in_cell () to determine the scale
                    if (scale_level < pred_scale_level - 1 || pred_scale_level < scale_level)
                    {
                        continue;
                    }

                    const auto &desc = keyfrm->descriptors_.row(idx);

                    const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

                    if (hamm_dist < best_dist)
                    {
                        best_dist = hamm_dist;
                        best_idx = idx;
                    }
                }

                if (HAMMING_DIST_THR_LOW < best_dist)
                {
                    continue;
                }

                matched_lms_in_keyfrm.at(best_idx) = lm;
                ++num_matches;
            }

            return num_matches;
        }

        unsigned int projection::match_keyframes_mutually(data::keyframe *keyfrm_1, data::keyframe *keyfrm_2, std::vector<data::landmark *> &matched_lms_in_keyfrm_1,
                                                          const float &s_12, const Mat33_t &rot_12, const Vec3_t &trans_12, const float margin) const
        {
            // Pose of keyframe1
            const Mat33_t rot_1w = keyfrm_1->get_rotation();
            const Vec3_t trans_1w = keyfrm_1->get_translation();

            // Pose of keyframe2
            const Mat33_t rot_2w = keyfrm_2->get_rotation();
            const Vec3_t trans_2w = keyfrm_2->get_translation();

            // Similarity transformation of coordinate system between given cameras
            const Mat33_t s_rot_12 = s_12 * rot_12;
            const Mat33_t s_rot_21 = (1.0 / s_12) * rot_12.transpose();
            const Vec3_t trans_21 = -s_rot_21 * trans_12;

            const auto landmarks_1 = keyfrm_1->get_landmarks();
            const auto landmarks_2 = keyfrm_2->get_landmarks();

            // Mark keyframes 1 and 2 if there are already matching feature points
            std::vector<bool> is_already_matched_in_keyfrm_1(landmarks_1.size(), false);
            std::vector<bool> is_already_matched_in_keyfrm_2(landmarks_2.size(), false);

            for (unsigned int idx_1 = 0; idx_1 < landmarks_1.size(); ++idx_1)
            {
                auto *lm = matched_lms_in_keyfrm_1.at(idx_1);
                if (!lm)
                {
                    continue;
                }
                const auto idx_2 = lm->get_index_in_keyframe(keyfrm_2);
                if (0 <= idx_2 && idx_2 < static_cast<int>(landmarks_2.size()))
                {
                    is_already_matched_in_keyfrm_1.at(idx_1) = true;
                    is_already_matched_in_keyfrm_2.at(idx_2) = true;
                }
            }

            std::vector<int> matched_indices_2_in_keyfrm_1(landmarks_1.size(), -1);
            std::vector<int> matched_indices_1_in_keyfrm_2(landmarks_2.size(), -1);

            // The 3D point observed by keyframe1 is transformed into the coordinate system of keyframe2 and then reprojected.
            // Find the corresponding feature points
            // (world -- SE3 -> keyframe1 -- Sim3 --> keyframe2)
            // s_rot_21 * (rot_1w * pos_w + trans_1w) + trans_21
            // = s_rot_21 * rot_1w * pos_w + s_rot_21 * trans_1w + trans_21
            {
                const Mat33_t s_rot_21w = s_rot_21 * rot_1w;
                const Vec3_t trans_21w = s_rot_21 * trans_1w + trans_21;
                for (unsigned int idx_1 = 0; idx_1 < landmarks_1.size(); ++idx_1)
                {
                    auto *lm = landmarks_1.at(idx_1);
                    if (!lm)
                    {
                        continue;
                    }
                    if (lm->will_be_erased())
                    {
                        continue;
                    }

                    if (is_already_matched_in_keyfrm_1.at(idx_1))
                    {
                        continue;
                    }

                    // Global standard 3D point coordinates
                    const Vec3_t pos_w = lm->get_pos_in_world();
                    const Vec3_t pos_2 = s_rot_21w * pos_w + trans_21w;

                    // Reproject for visibility
                    Vec2_t reproj;
                    float x_right;
                    const bool in_image = keyfrm_2->camera_->reproject_to_image(s_rot_21w, trans_21w, pos_w, reproj, x_right);

                    // Through if reprojected outside the image
                    if (!in_image)
                    {
                        continue;
                    }

                    // Make sure it is within the ORB scale
                    const auto cam_to_lm_dist = pos_2.norm();
                    const auto max_cam_to_lm_dist = lm->get_max_valid_distance();
                    const auto min_cam_to_lm_dist = lm->get_min_valid_distance();

                    if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist)
                    {
                        continue;
                    }

                    // Get the feature point of the cell where the point that reprojected the 3D point exists
                    const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, keyfrm_2);

                    const auto indices = keyfrm_2->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm_2->scale_factors_.at(pred_scale_level));

                    if (indices.empty())
                    {
                        continue;
                    }

                    // Find the feature point with the closest descriptor
                    const auto lm_desc = lm->get_descriptor();

                    unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                    int best_idx_2 = -1;

                    for (const auto idx_2 : indices)
                    {
                        const auto scale_level = static_cast<unsigned int>(keyfrm_2->keypts_.at(idx_2).octave);

                        // TODO: Use keyfrm-> get_keypts_in_cell () to determine the scale
                        if (scale_level < pred_scale_level - 1 || pred_scale_level < scale_level)
                        {
                            continue;
                        }

                        const auto &desc = keyfrm_2->descriptors_.row(idx_2);

                        const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

                        if (hamm_dist < best_hamm_dist)
                        {
                            best_hamm_dist = hamm_dist;
                            best_idx_2 = idx_2;
                        }
                    }

                    if (best_hamm_dist <= HAMMING_DIST_THR_HIGH)
                    {
                        matched_indices_2_in_keyfrm_1.at(idx_1) = best_idx_2;
                    }
                }
            }

            // The 3D point observed by keyframe1 is transformed into the coordinate system of keyframe2 and then reprojected.
            // Find the corresponding feature points
            // (world -- SE3 -> keyframe2 -- Sim3 --> keyframe1)
            // s_rot_12 * (rot_2w * pos_w + trans_2w) + trans_12
            // = s_rot_12 * rot_2w * pos_w + s_rot_12 * trans_2w + trans_12
            {
                const Mat33_t s_rot_12w = s_rot_12 * rot_2w;
                const Vec3_t trans_12w = s_rot_12 * trans_2w + trans_12;
                for (unsigned int idx_2 = 0; idx_2 < landmarks_2.size(); ++idx_2)
                {
                    auto *lm = landmarks_2.at(idx_2);
                    if (!lm)
                    {
                        continue;
                    }
                    if (lm->will_be_erased())
                    {
                        continue;
                    }

                    if (is_already_matched_in_keyfrm_2.at(idx_2))
                    {
                        continue;
                    }

                    // Global standard 3D point coordinates
                    const Vec3_t pos_w = lm->get_pos_in_world();
                    const Vec3_t pos_1 = s_rot_12w * pos_w + trans_12w;

                    // Reproject for visibility
                    Vec2_t reproj;
                    float x_right;
                    const bool in_image = keyfrm_2->camera_->reproject_to_image(s_rot_12w, trans_12w, pos_w, reproj, x_right);

                    // Through if reprojected outside the image
                    if (!in_image)
                    {
                        continue;
                    }

                    // Make sure it is within the ORB scale
                    const auto cam_to_lm_dist = pos_1.norm();
                    const auto max_cam_to_lm_dist = lm->get_max_valid_distance();
                    const auto min_cam_to_lm_dist = lm->get_min_valid_distance();

                    if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist)
                    {
                        continue;
                    }

                    // Get the feature point of the cell where the point that reprojected the 3D point exists
                    const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, keyfrm_1);

                    const auto indices = keyfrm_1->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm_1->scale_factors_.at(pred_scale_level));

                    if (indices.empty())
                    {
                        continue;
                    }

                    // Find the feature point with the closest descriptor
                    const auto lm_desc = lm->get_descriptor();

                    unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                    int best_idx_1 = -1;

                    for (const auto idx_1 : indices)
                    {
                        const auto scale_level = static_cast<unsigned int>(keyfrm_1->keypts_.at(idx_1).octave);

                        // TODO: Use keyfrm-> get_keypts_in_cell () to determine the scale
                        if (scale_level < pred_scale_level - 1 || pred_scale_level < scale_level)
                        {
                            continue;
                        }

                        const auto &desc = keyfrm_1->descriptors_.row(idx_1);

                        const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

                        if (hamm_dist < best_hamm_dist)
                        {
                            best_hamm_dist = hamm_dist;
                            best_idx_1 = idx_1;
                        }
                    }

                    if (best_hamm_dist <= HAMMING_DIST_THR_HIGH)
                    {
                        matched_indices_1_in_keyfrm_2.at(idx_2) = best_idx_1;
                    }
                }
            }

            // record only cross-match
            unsigned int num_matches = 0;
            for (unsigned int i = 0; i < landmarks_1.size(); ++i)
            {
                const auto idx_2 = matched_indices_2_in_keyfrm_1.at(i);
                if (idx_2 < 0)
                {
                    continue;
                }

                const auto idx_1 = matched_indices_1_in_keyfrm_2.at(idx_2);
                if (idx_1 == static_cast<int>(i))
                {
                    matched_lms_in_keyfrm_1.at(idx_1) = landmarks_2.at(idx_2);
                    ++num_matches;
                }
            }

            return num_matches;
        }

    } // namespace match
} // namespace PLPSLAM
