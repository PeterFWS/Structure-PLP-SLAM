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
#include "PLPSLAM/match/robust.h"
#include "PLPSLAM/match/angle_checker.h"
#include "PLPSLAM/solve/essential_solver.h"

#ifdef USE_DBOW2
#include <DBoW2/FeatureVector.h>
#else
#include <fbow/fbow.h>
#endif

namespace PLPSLAM
{
    namespace match
    {

        unsigned int robust::match_for_triangulation(data::keyframe *keyfrm_1, data::keyframe *keyfrm_2, const Mat33_t &E_12,
                                                     std::vector<std::pair<unsigned int, unsigned int>> &matched_idx_pairs)
        {
            unsigned int num_matches = 0;

            angle_checker<int> angle_checker;

            // Find the coordinates of the epipole of keyframe2 = Project the camera center of keyframe1 onto keyframe2
            const Vec3_t cam_center_1 = keyfrm_1->get_cam_center();
            const Mat33_t rot_2w = keyfrm_2->get_rotation();
            const Vec3_t trans_2w = keyfrm_2->get_translation();
            Vec3_t epiplane_in_keyfrm_2;
            keyfrm_2->camera_->reproject_to_bearing(rot_2w, trans_2w, cam_center_1, epiplane_in_keyfrm_2);

            // Get 3D point information of keyframe
            const auto assoc_lms_in_keyfrm_1 = keyfrm_1->get_landmarks();
            const auto assoc_lms_in_keyfrm_2 = keyfrm_2->get_landmarks();

            // Store matching information
            // In order to request the correspondence for each feature point of keyframe1, exclude the ones that already correspond to keyframe1 in keyframe2.
            std::vector<bool> is_already_matched_in_keyfrm_2(keyfrm_2->num_keypts_, false);
            // Store the idx of keyframe2 that corresponds to the idx of keyframe1
            std::vector<int> matched_indices_2_in_keyfrm_1(keyfrm_1->num_keypts_, -1);

#ifdef USE_DBOW2
            DBoW2::FeatureVector::const_iterator itr_1 = keyfrm_1->bow_feat_vec_.begin();
            DBoW2::FeatureVector::const_iterator itr_2 = keyfrm_2->bow_feat_vec_.begin();
            const DBoW2::FeatureVector::const_iterator itr_1_end = keyfrm_1->bow_feat_vec_.end();
            const DBoW2::FeatureVector::const_iterator itr_2_end = keyfrm_2->bow_feat_vec_.end();
#else
            fbow::BoWFeatVector::const_iterator itr_1 = keyfrm_1->bow_feat_vec_.begin();
            fbow::BoWFeatVector::const_iterator itr_2 = keyfrm_2->bow_feat_vec_.begin();
            const fbow::BoWFeatVector::const_iterator itr_1_end = keyfrm_1->bow_feat_vec_.end();
            const fbow::BoWFeatVector::const_iterator itr_2_end = keyfrm_2->bow_feat_vec_.end();
#endif

            while (itr_1 != itr_1_end && itr_2 != itr_2_end)
            {
                // Check if the node number (first) of BoW tree matches
                if (itr_1->first == itr_2->first)
                {
                    // If the node number (first) of BoW tree matches,
                    // Bring the feature point index (second) and check if it corresponds
                    const auto &keyfrm_1_indices = itr_1->second;
                    const auto &keyfrm_2_indices = itr_2->second;

                    for (const auto idx_1 : keyfrm_1_indices)
                    {
                        auto lm_1 = assoc_lms_in_keyfrm_1.at(idx_1);
                        // Through if a 3D point "exists" (because it is a matching before triangulation)
                        if (lm_1)
                        {
                            continue;
                        }

                        // Check if it is a stereo key point
                        const bool is_stereo_keypt_1 = 0 <= keyfrm_1->stereo_x_right_.at(idx_1);

                        // Get feature points / features
                        const auto &keypt_1 = keyfrm_1->undist_keypts_.at(idx_1);
                        const Vec3_t &bearing_1 = keyfrm_1->bearings_.at(idx_1);
                        const auto &desc_1 = keyfrm_1->descriptors_.row(idx_1);

                        // Find the keyframe2 feature that has the closest Hamming distance
                        unsigned int best_hamm_dist = HAMMING_DIST_THR_LOW;
                        int best_idx_2 = -1;

                        for (const auto idx_2 : keyfrm_2_indices)
                        {
                            auto lm_2 = assoc_lms_in_keyfrm_2.at(idx_2);
                            // Through if a 3D point "exists" (because it is a matching before triangulation)
                            if (lm_2)
                            {
                                continue;
                            }

                            // Through if support has already been obtained
                            if (is_already_matched_in_keyfrm_2.at(idx_2))
                            {
                                continue;
                            }

                            /// Check if it is a stereo key point
                            const bool is_stereo_keypt_2 = 0 <= keyfrm_2->stereo_x_right_.at(idx_2);

                            // Get feature points / features
                            const Vec3_t &bearing_2 = keyfrm_2->bearings_.at(idx_2);
                            const auto &desc_2 = keyfrm_2->descriptors_.row(idx_2);

                            // Distance calculation
                            const auto hamm_dist = compute_descriptor_distance_32(desc_1, desc_2);

                            if (HAMMING_DIST_THR_LOW < hamm_dist || best_hamm_dist < hamm_dist)
                            {
                                continue;
                            }

                            if (!is_stereo_keypt_1 && !is_stereo_keypt_2)
                            {
                                // If both are not stereo keypoints, do not use feature points near Epipole
                                const auto cos_dist = epiplane_in_keyfrm_2.dot(bearing_2);
                                // Threshold for the angle between epipole and bearing (= 3.0deg)
                                constexpr double cos_dist_thr = 0.99862953475;

                                // Do not match if the sandwich angle is smaller than the threshold
                                if (cos_dist_thr < cos_dist)
                                {
                                    continue;
                                }
                            }

                            // Consistency check by E matrix
                            const bool is_inlier = check_epipolar_constraint(bearing_1, bearing_2, E_12,
                                                                             keyfrm_1->scale_factors_.at(keypt_1.octave));
                            if (is_inlier)
                            {
                                best_idx_2 = idx_2;
                                best_hamm_dist = hamm_dist;
                            }
                        }

                        if (best_idx_2 < 0)
                        {
                            continue;
                        }

                        is_already_matched_in_keyfrm_2.at(best_idx_2) = true;
                        matched_indices_2_in_keyfrm_1.at(idx_1) = best_idx_2;
                        ++num_matches;

                        if (check_orientation_)
                        {
                            const auto delta_angle = keypt_1.angle - keyfrm_2->undist_keypts_.at(best_idx_2).angle;
                            angle_checker.append_delta_angle(delta_angle, idx_1);
                        }
                    }

                    ++itr_1;
                    ++itr_2;
                }
                else if (itr_1->first < itr_2->first)
                {
                    itr_1 = keyfrm_1->bow_feat_vec_.lower_bound(itr_2->first);
                }
                else
                {
                    itr_2 = keyfrm_2->bow_feat_vec_.lower_bound(itr_1->first);
                }
            }

            if (check_orientation_)
            {
                const auto invalid_matches = angle_checker.get_invalid_matches();
                for (const auto invalid_idx : invalid_matches)
                {
                    matched_indices_2_in_keyfrm_1.at(invalid_idx) = -1;
                    --num_matches;
                }
            }

            matched_idx_pairs.clear();
            matched_idx_pairs.reserve(num_matches);

            for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_keyfrm_1.size(); ++idx_1)
            {
                if (matched_indices_2_in_keyfrm_1.at(idx_1) < 0)
                {
                    continue;
                }
                matched_idx_pairs.emplace_back(std::make_pair(idx_1, matched_indices_2_in_keyfrm_1.at(idx_1)));
            }

            return num_matches;
        }

        unsigned int robust::match_frame_and_keyframe(data::frame &frm, data::keyframe *keyfrm,
                                                      std::vector<data::landmark *> &matched_lms_in_frm)
        {
            // Initialize
            const auto num_frm_keypts = frm.num_keypts_; // int
            const auto keyfrm_lms = keyfrm->get_landmarks();
            unsigned int num_inlier_matches = 0;
            matched_lms_in_frm = std::vector<data::landmark *>(num_frm_keypts, nullptr);

            // Calculate brute-force match
            std::vector<std::pair<int, int>> matches;
            brute_force_match(frm, keyfrm, matches);

            // Extract only inliers with eight-point RANSAC
            solve::essential_solver solver(frm.bearings_, keyfrm->bearings_, matches);
            solver.find_via_ransac(50, false);
            if (!solver.solution_is_valid())
            {
                return 0;
            }
            const auto is_inlier_matches = solver.get_inlier_matches();

            // Store information
            for (unsigned int i = 0; i < matches.size(); ++i)
            {
                if (!is_inlier_matches.at(i))
                {
                    continue;
                }
                const auto frm_idx = matches.at(i).first;
                const auto keyfrm_idx = matches.at(i).second;

                matched_lms_in_frm.at(frm_idx) = keyfrm_lms.at(keyfrm_idx);
                ++num_inlier_matches;
            }

            return num_inlier_matches;
        }

        unsigned int robust::brute_force_match(data::frame &frm, data::keyframe *keyfrm, std::vector<std::pair<int, int>> &matches)
        {
            unsigned int num_matches = 0;

            angle_checker<int> angle_checker;

            // 1. Get frame and keyframe information

            const auto num_keypts_1 = frm.num_keypts_;
            const auto num_keypts_2 = keyfrm->num_keypts_;

            const auto keypts_1 = frm.keypts_;
            const auto keypts_2 = keyfrm->keypts_;

            const auto lms_2 = keyfrm->get_landmarks();

            const auto &descs_1 = frm.descriptors_;
            const auto &descs_2 = keyfrm->descriptors_;

            // 2. For each descriptor of keyframe, find the descriptor of the frame closest to the first and second
            // Keyframe descriptors are only for those associated with 3D points

            // idx_2 corresponding to each idx_1
            auto matched_indices_2_in_1 = std::vector<int>(num_keypts_1, -1);
            // avoid duplication
            std::unordered_set<int> already_matched_indices_1;

            // (1) loop through the keypoints in the keyframe
            for (unsigned int idx_2 = 0; idx_2 < num_keypts_2; ++idx_2)
            {
                // Target only those with valid 3D points
                auto lm_2 = lms_2.at(idx_2);
                if (!lm_2)
                {
                    continue;
                }
                if (lm_2->will_be_erased())
                {
                    continue;
                }

                // Get the keyframe descriptor
                const auto &desc_2 = descs_2.row(idx_2); // cv::Mat

                // Find the descriptor of the frame closest to the 1st and 2nd
                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                int best_idx_1 = -1;
                unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;

                // (2) loop through the keypoints in the current frame
                for (unsigned int idx_1 = 0; idx_1 < num_keypts_1; ++idx_1)
                {
                    // avoid duplication
                    if (static_cast<bool>(already_matched_indices_1.count(idx_1)))
                    {
                        continue;
                    }

                    const auto &desc_1 = descs_1.row(idx_1); // cv::Mat

                    const auto hamm_dist = compute_descriptor_distance_32(desc_2, desc_1);

                    if (hamm_dist < best_hamm_dist)
                    {
                        second_best_hamm_dist = best_hamm_dist;
                        best_hamm_dist = hamm_dist;
                        best_idx_1 = idx_1;
                    }
                    else if (hamm_dist < second_best_hamm_dist)
                    {
                        second_best_hamm_dist = hamm_dist;
                    }
                }

                if (HAMMING_DIST_THR_LOW < best_hamm_dist)
                {
                    continue;
                }

                if (best_idx_1 < 0)
                {
                    continue;
                }

                // ratio test
                if (lowe_ratio_ * second_best_hamm_dist < static_cast<float>(best_hamm_dist))
                {
                    continue;
                }

                matched_indices_2_in_1.at(best_idx_1) = idx_2;
                // avoid duplication
                already_matched_indices_1.insert(best_idx_1);

                // why we don't check the orientation in robust tracking?
                if (check_orientation_)
                {
                    const auto delta_angle = keypts_1.at(best_idx_1).angle - keypts_2.at(idx_2).angle;
                    angle_checker.append_delta_angle(delta_angle, best_idx_1);
                }

                ++num_matches;
            }

            if (check_orientation_)
            {
                const auto invalid_matches = angle_checker.get_invalid_matches();
                for (const auto invalid_idx_1 : invalid_matches)
                {
                    matched_indices_2_in_1.at(invalid_idx_1) = -1;
                    --num_matches;
                }
            }

            // (3) aggregate inlier matches
            matches.clear();
            matches.reserve(num_matches);
            for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_1.size(); ++idx_1)
            {
                const auto idx_2 = matched_indices_2_in_1.at(idx_1);
                if (idx_2 < 0)
                {
                    continue;
                }
                matches.emplace_back(std::make_pair(idx_1, idx_2)); // inx_1 from current frame, idx_2 from keyframe
            }

            return num_matches;
        }

        bool robust::check_epipolar_constraint(const Vec3_t &bearing_1, const Vec3_t &bearing_2,
                                               const Mat33_t &E_12, const float bearing_1_scale_factor)
        {
            // normal vector of t epipolar plane on keyframe1
            const Vec3_t epiplane_in_1 = E_12 * bearing_2;

            // Find the angle between the normal vector and bearing
            const auto cos_residual = epiplane_in_1.dot(bearing_1) / epiplane_in_1.norm();
            const auto residual_rad = M_PI / 2.0 - std::abs(std::acos(cos_residual));

            // inlier threshold (= 0.2deg)
            // (e.g. FOV = 90deg, 0.2deg is equivalent to 2pix in the horizontal direction in a camera with 900pix in the horizontal direction)
            // TODO: Threshold parameterization
            constexpr double residual_deg_thr = 0.2;
            constexpr double residual_rad_thr = residual_deg_thr * M_PI / 180.0;

            // The larger the feature point scale, the looser the threshold
            // TODO: Considering threshold weighting
            return residual_rad < residual_rad_thr * bearing_1_scale_factor;
        }

    } // namespace match
} // namespace PLPSLAM
