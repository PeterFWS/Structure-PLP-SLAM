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

#include "PLPSLAM/match/fuse.h"
#include "PLPSLAM/camera/base.h"
#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/data/landmark_line.h"

#include <vector>
#include <unordered_set>

#include <spdlog/spdlog.h>

namespace PLPSLAM
{
    namespace match
    {

        unsigned int fuse::detect_duplication(data::keyframe *keyfrm, const Mat44_t &Sim3_cw, const std::vector<data::landmark *> &landmarks_to_check,
                                              const float margin, std::vector<data::landmark *> &duplicated_lms_in_keyfrm)
        {
            unsigned int num_fused = 0;

            // Disassemble Sim3 into SE3
            const Mat33_t s_rot_cw = Sim3_cw.block<3, 3>(0, 0);
            const auto s_cw = std::sqrt(s_rot_cw.block<1, 3>(0, 0).dot(s_rot_cw.block<1, 3>(0, 0)));
            const Mat33_t rot_cw = s_rot_cw / s_cw;
            const Vec3_t trans_cw = Sim3_cw.block<3, 1>(0, 3) / s_cw;
            const Vec3_t cam_center = -rot_cw.transpose() * trans_cw;

            duplicated_lms_in_keyfrm = std::vector<data::landmark *>(landmarks_to_check.size(), nullptr);

            const auto valid_lms_in_keyfrm = keyfrm->get_valid_landmarks();

            for (unsigned int i = 0; i < landmarks_to_check.size(); ++i)
            {
                auto *lm = landmarks_to_check.at(i);
                if (lm->will_be_erased())
                {
                    continue;
                }
                // If this 3D point and the feature point of the keyframe already correspond, there is no need to reproject and integrate, so through
                if (valid_lms_in_keyfrm.count(lm))
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

                // Calculate the angle of the 3D point with the average observation vector,
                // and discard if it is larger than the threshold value (60deg).
                const Vec3_t obs_mean_normal = lm->get_obs_mean_normal();

                if (cam_to_lm_vec.dot(obs_mean_normal) < 0.5 * cam_to_lm_dist)
                {
                    continue;
                }

                // Acquire the feature point of the cell where the point that reprojected the 3D point exists
                const int pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, keyfrm);

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
                    const auto scale_level = keyfrm->keypts_.at(idx).octave;

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

                auto *lm_in_keyfrm = keyfrm->get_landmark(best_idx);
                if (lm_in_keyfrm)
                {
                    // There is a 3D point corresponding to the best_idx of the keyframe-> if it overlaps
                    if (!lm_in_keyfrm->will_be_erased())
                    {
                        duplicated_lms_in_keyfrm.at(i) = lm_in_keyfrm;
                    }
                }
                else
                {
                    // There is no 3D point corresponding to best_idx in keyframe
                    // Add observation information
                    lm->add_observation(keyfrm, best_idx);
                    keyfrm->add_landmark(lm, best_idx);
                }

                ++num_fused;
            }

            return num_fused;
        }

        template <typename T>
        unsigned int fuse::replace_duplication(data::keyframe *keyfrm, const T &landmarks_to_check, const float margin)
        {
            unsigned int num_fused = 0;

            const Mat33_t rot_cw = keyfrm->get_rotation();
            const Vec3_t trans_cw = keyfrm->get_translation();
            const Vec3_t cam_center = keyfrm->get_cam_center();

            for (const auto lm : landmarks_to_check)
            {
                if (!lm)
                {
                    continue;
                }
                if (lm->will_be_erased())
                {
                    continue;
                }
                if (lm->is_observed_in_keyframe(keyfrm))
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

                // Calculate the angle of the 3D point with the average observation vector,
                // and discard if it is larger than the threshold (60deg).
                const Vec3_t obs_mean_normal = lm->get_obs_mean_normal();

                if (cam_to_lm_vec.dot(obs_mean_normal) < 0.5 * cam_to_lm_dist)
                {
                    continue;
                }

                // Acquire the feature point of the cell where the point that reprojected the 3D point exists
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
                    const auto &keypt = keyfrm->undist_keypts_.at(idx);

                    const auto scale_level = static_cast<unsigned int>(keypt.octave);

                    // TODO: Use keyfrm-> get_keypts_in_cell () to determine the scale
                    if (scale_level < pred_scale_level - 1 || pred_scale_level < scale_level)
                    {
                        continue;
                    }

                    if (keyfrm->stereo_x_right_.at(idx) >= 0)
                    {
                        // Calculate the reprojection error with 3 degrees of freedom if a stereo match exists
                        const auto e_x = reproj(0) - keypt.pt.x;
                        const auto e_y = reproj(1) - keypt.pt.y;
                        const auto e_x_right = x_right - keyfrm->stereo_x_right_.at(idx);
                        const auto reproj_error_sq = e_x * e_x + e_y * e_y + e_x_right * e_x_right;

                        // Degrees of freedom n=3
                        constexpr float chi_sq_3D = 7.81473;
                        if (chi_sq_3D < reproj_error_sq * keyfrm->inv_level_sigma_sq_.at(scale_level))
                        {
                            continue;
                        }
                    }
                    else
                    {
                        // Calculate the reprojection error with 2 degrees of freedom if stereo match does not exist
                        const auto e_x = reproj(0) - keypt.pt.x;
                        const auto e_y = reproj(1) - keypt.pt.y;
                        const auto reproj_error_sq = e_x * e_x + e_y * e_y;

                        // degrees of freedom n=2
                        constexpr float chi_sq_2D = 5.99146;
                        if (chi_sq_2D < reproj_error_sq * keyfrm->inv_level_sigma_sq_.at(scale_level))
                        {
                            continue;
                        }
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

                auto *lm_in_keyfrm = keyfrm->get_landmark(best_idx);
                if (lm_in_keyfrm)
                {
                    // There is a 3D point corresponding to the best_idx of keyframe-> When it overlaps
                    if (!lm_in_keyfrm->will_be_erased())
                    {
                        // Replace with more reliable (= more observed) 3D points
                        if (lm->num_observations() < lm_in_keyfrm->num_observations())
                        {
                            // Replace with lm_in_keyfrm
                            lm->replace(lm_in_keyfrm);
                        }
                        else
                        {
                            // Replace with lm
                            lm_in_keyfrm->replace(lm);
                        }
                    }
                }
                else
                {
                    // There is no 3D point corresponding to best_idx in keyframe
                    // Add observation information
                    lm->add_observation(keyfrm, best_idx);
                    keyfrm->add_landmark(lm, best_idx);
                }

                ++num_fused;
            }

            return num_fused;
        }

        // FW:
        template <typename T>
        unsigned int fuse::replace_duplication_line(data::keyframe *keyfrm, const T &landmarks_to_check, const float margin)
        {
            unsigned int num_fused = 0;

            const Mat33_t rot_cw = keyfrm->get_rotation();
            const Vec3_t trans_cw = keyfrm->get_translation();
            const Vec3_t cam_center = keyfrm->get_cam_center();

            for (const auto lm : landmarks_to_check)
            {
                if (!lm)
                {
                    continue;
                }
                if (lm->will_be_erased())
                {
                    continue;
                }
                if (lm->is_observed_in_keyframe(keyfrm))
                {
                    continue;
                }

                // Coordinates of starting point (sp) and ending point (ep)
                const Vec6_t pos_w = lm->get_pos_in_world();
                Vec3_t pos_w_sp = pos_w.head(3);
                Vec3_t pos_w_ep = pos_w.tail(3);

                // Reproject for visibility
                Vec2_t reproj_sp, reproj_ep;
                float x_right_sp, x_right_ep;
                const bool in_image_sp = keyfrm->camera_->reproject_to_image(rot_cw, trans_cw, pos_w_sp, reproj_sp, x_right_sp);
                const bool in_image_ep = keyfrm->camera_->reproject_to_image(rot_cw, trans_cw, pos_w_ep, reproj_ep, x_right_ep);

                // Through if reprojected outside the image
                if (!in_image_sp && !in_image_ep)
                {
                    continue;
                }

                // check if the line is partial occluded
                bool partial_occlusion = false;
                if (!in_image_sp || !in_image_ep)
                {
                    // check if the middle point are within the image frustum
                    Vec3_t pos_w_mp = 0.5 * (pos_w_sp + pos_w_ep);
                    Vec2_t reproj_mp;
                    float x_right_mp;
                    const bool in_image_mp = keyfrm->camera_->reproject_to_image(rot_cw, trans_cw, pos_w_mp, reproj_mp, x_right_mp);

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

                // Make sure it is within the ORB scale
                const Vec3_t cam_to_lm_vec_sp = pos_w_sp - cam_center;
                const auto cam_to_lm_dist_sp = cam_to_lm_vec_sp.norm();

                const Vec3_t cam_to_lm_vec_ep = pos_w_ep - cam_center;
                const auto cam_to_lm_dist_ep = cam_to_lm_vec_ep.norm();

                const auto max_cam_to_lm_dist = lm->get_max_valid_distance();
                const auto min_cam_to_lm_dist = lm->get_min_valid_distance();

                if (cam_to_lm_dist_sp < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist_sp ||
                    cam_to_lm_dist_ep < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist_ep)
                {
                    continue;
                }

                // Acquire the feature line of the cell where the line that reprojected the 3D line exists
                const Vec3_t cam_to_lm_vec_mp = 0.5 * (pos_w_sp + pos_w_ep) - cam_center;
                const auto cam_to_lm_dist_mp = cam_to_lm_vec_mp.norm();
                const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist_mp, keyfrm->_log_scale_factor_lsd, keyfrm->_num_scale_levels_lsd);

                const auto indices = keyfrm->get_keylines_in_cell(reproj_sp(0), reproj_sp(1),
                                                                  reproj_ep(0), reproj_ep(1),
                                                                  margin * keyfrm->_scale_factors_lsd.at(pred_scale_level));

                if (indices.empty())
                {
                    continue;
                }

                // Find the feature line with the closest descriptor
                const auto lm_desc = lm->get_descriptor();

                unsigned int best_dist = MAX_HAMMING_DIST;
                int best_idx = -1;

                for (const auto idx : indices)
                {
                    const auto &keyline = keyfrm->_keylsd.at(idx);

                    const auto scale_level = static_cast<unsigned int>(keyline.octave);

                    // Calculate the reprojection error with 3 degrees of freedom
                    Vec3_t proj_sp{reproj_sp(0), reproj_sp(1), 1.0};
                    Vec3_t proj_ep{reproj_ep(0), reproj_ep(1), 1.0};
                    Vec3_t proj_line = proj_sp.cross(proj_ep);

                    auto reproj_error_sp = (keyline.getStartPoint().x * proj_line(0) + keyline.getStartPoint().y * proj_line(1) + proj_line(2)) /
                                           sqrt(proj_line(0) * proj_line(0) + proj_line(1) * proj_line(1));

                    auto reproj_error_ep = (keyline.getEndPoint().x * proj_line(0) + keyline.getEndPoint().y * proj_line(1) + proj_line(2)) /
                                           sqrt(proj_line(0) * proj_line(0) + proj_line(1) * proj_line(1));

                    constexpr float chi_sq_2D = 5.99146;
                    if (chi_sq_2D < (reproj_error_sp * reproj_error_sp + reproj_error_ep * reproj_error_ep) * keyfrm->_inv_level_sigma_sq_lsd.at(scale_level))
                    {
                        continue;
                    }

                    const auto &desc = keyfrm->_lbd_descr.row(idx);

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

                auto *lm_in_keyfrm = keyfrm->get_landmark_line(best_idx);
                if (lm_in_keyfrm)
                {
                    // There is a 3D line corresponding to the best_idx of keyframe-> When it overlaps
                    if (!lm_in_keyfrm->will_be_erased())
                    {
                        // Replace with more reliable (= more observed) 3D lines
                        if (lm->num_observations() < lm_in_keyfrm->num_observations())
                        {
                            // Replace with lm_in_keyfrm
                            lm->replace(lm_in_keyfrm);
                        }
                        else
                        {
                            // Replace with lm
                            lm_in_keyfrm->replace(lm);
                        }
                    }
                }
                else
                {
                    // There is no 3D line corresponding to best_idx in keyframe
                    // Add observation information
                    lm->add_observation(keyfrm, best_idx);
                    keyfrm->add_landmark_line(lm, best_idx);
                }

                ++num_fused;
            }

            // spdlog::info("num_fused line: {}", num_fused);

            return num_fused;
        }

        // Explicitly materialize
        template unsigned int fuse::replace_duplication(data::keyframe *, const std::vector<data::landmark *> &, const float);
        template unsigned int fuse::replace_duplication(data::keyframe *, const std::unordered_set<data::landmark *> &, const float);

        // FW:
        template unsigned int fuse::replace_duplication_line(data::keyframe *, const std::vector<data::Line *> &, const float);
        template unsigned int fuse::replace_duplication_line(data::keyframe *, const std::unordered_set<data::Line *> &, const float);

    } // namespace match
} // namespace PLPSLAM
