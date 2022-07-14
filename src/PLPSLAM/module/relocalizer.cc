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

#include "PLPSLAM/data/frame.h"
#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/data/bow_database.h"
#include "PLPSLAM/module/relocalizer.h"
#include "PLPSLAM/util/fancy_index.h"

#include <spdlog/spdlog.h>

namespace PLPSLAM
{
    namespace module
    {

        relocalizer::relocalizer(data::bow_database *bow_db,
                                 const double bow_match_lowe_ratio, const double proj_match_lowe_ratio,
                                 const unsigned int min_num_bow_matches, const unsigned int min_num_valid_obs)
            : bow_db_(bow_db),
              min_num_bow_matches_(min_num_bow_matches), min_num_valid_obs_(min_num_valid_obs),
              bow_matcher_(bow_match_lowe_ratio, true), proj_matcher_(proj_match_lowe_ratio, true),
              pose_optimizer_(),
              _pose_optimizer_extended_line()
        {
            spdlog::debug("CONSTRUCT: module::relocalizer");
        }

        relocalizer::~relocalizer()
        {
            spdlog::debug("DESTRUCT: module::relocalizer");
        }

        bool relocalizer::relocalize(data::frame &curr_frm)
        {
            curr_frm.compute_bow();

            // acquire relocalization candidates
            const auto reloc_candidates = bow_db_->acquire_relocalization_candidates(&curr_frm); // FW: std::vector<keyframe *>
            if (reloc_candidates.empty())
            {
                return false;
            }
            const auto num_candidates = reloc_candidates.size();

            std::vector<std::vector<data::landmark *>> matched_landmarks(num_candidates);

            // For each candidate, find the corresponding points with the Bow tree matcher
            for (unsigned int i = 0; i < num_candidates; ++i)
            {
                auto keyfrm = reloc_candidates.at(i);
                if (keyfrm->will_be_erased())
                {
                    continue;
                }

                // FW: acquire 3D-2D matches via checking 2D-2D matches using BoW
                const auto num_matches = bow_matcher_.match_frame_and_keyframe(keyfrm, curr_frm, matched_landmarks.at(i));

                // discard the candidate if the number of 2D-3D matches is less than the threshold
                if (num_matches < min_num_bow_matches_)
                {
                    continue;
                }

                // setup PnP solver with the current 2D-3D matches
                const auto valid_indices = extract_valid_indices(matched_landmarks.at(i));
                auto pnp_solver = setup_pnp_solver(valid_indices, curr_frm.bearings_, curr_frm.keypts_,
                                                   matched_landmarks.at(i), curr_frm.scale_factors_);

                // [1] Estimate the initial camera pose with EPnP(+RANSAC)
                pnp_solver->find_via_ransac(30);
                if (!pnp_solver->solution_is_valid())
                {
                    continue;
                }

                curr_frm.cam_pose_cw_ = pnp_solver->get_best_cam_pose();
                curr_frm.update_pose_params();

                // [2] Apply pose optimizer
                // get the inlier indices after EPnP+RANSAC
                const auto inlier_indices = util::resample_by_indices(valid_indices, pnp_solver->get_inlier_flags());

                // set 2D-3D matches for the pose optimization
                curr_frm.landmarks_ = std::vector<data::landmark *>(curr_frm.num_keypts_, nullptr);
                std::set<data::landmark *> already_found_landmarks;
                for (const auto idx : inlier_indices)
                {
                    // Set only valid 3D points in current frame
                    curr_frm.landmarks_.at(idx) = matched_landmarks.at(i).at(idx);
                    // Already record the 3D points corresponding to the feature points
                    already_found_landmarks.insert(matched_landmarks.at(i).at(idx));
                }

                // pose optimization
                auto num_valid_obs = pose_optimizer_.optimize(curr_frm);
                // discard the candidate if the number of the inliers is less than the threshold
                if (num_valid_obs < min_num_bow_matches_ / 2)
                {
                    continue;
                }

                // reject outliers
                for (unsigned int idx = 0; idx < curr_frm.num_keypts_; idx++)
                {
                    if (!curr_frm.outlier_flags_.at(idx))
                    {
                        continue;
                    }
                    curr_frm.landmarks_.at(idx) = nullptr;
                }

                // [3] Apply projection match to increase 2D-3D matches
                // projection match based on the pre-optimized camera pose
                auto num_found = proj_matcher_.match_frame_and_keyframe(curr_frm, reloc_candidates.at(i), already_found_landmarks, 10, 100);
                // discard the candidate if the number of the inliers is less than the threshold
                if (num_valid_obs + num_found < min_num_valid_obs_)
                {
                    continue;
                }

                // FW: find also 3D lines
                std::set<data::Line *> already_found_landmarks_line;
                auto num_line3d_found = proj_matcher_.match_frame_and_keyframe_line(curr_frm, reloc_candidates.at(i), already_found_landmarks_line, 10, 100);
                spdlog::info("3D lines found during relocalization: {}", num_line3d_found);
                if (num_line3d_found > 0)
                {
                    for (unsigned int idx = 0; idx < curr_frm._num_keylines; ++idx)
                    {
                        if (!curr_frm._landmarks_line.at(idx))
                        {
                            continue;
                        }
                        already_found_landmarks_line.insert(curr_frm._landmarks_line.at(idx));
                    }
                }

                // [4] Re-apply the pose optimizer
                if (num_line3d_found > 0)
                {
                    // FW:
                    num_valid_obs = _pose_optimizer_extended_line.optimize(curr_frm);
                }
                else
                {
                    num_valid_obs = pose_optimizer_.optimize(curr_frm);
                }

                // If it falls below the threshold, perform a projection match again.
                if (num_valid_obs < min_num_valid_obs_)
                {
                    // Excludes those that are already supported
                    already_found_landmarks.clear();
                    for (unsigned int idx = 0; idx < curr_frm.num_keypts_; ++idx)
                    {
                        if (!curr_frm.landmarks_.at(idx))
                        {
                            continue;
                        }
                        already_found_landmarks.insert(curr_frm.landmarks_.at(idx));
                    }
                    // Perform projection match again-> Set 2D-3D support
                    auto num_additional = proj_matcher_.match_frame_and_keyframe(curr_frm, reloc_candidates.at(i), already_found_landmarks, 3, 64);

                    // Discard if less than threshold
                    if (num_valid_obs + num_additional < min_num_valid_obs_)
                    {
                        continue;
                    }

                    // FW:
                    already_found_landmarks_line.clear();
                    for (unsigned int idx = 0; idx < curr_frm._num_keylines; ++idx)
                    {
                        if (!curr_frm._landmarks_line.at(idx))
                        {
                            continue;
                        }
                        already_found_landmarks_line.insert(curr_frm._landmarks_line.at(idx));
                    }
                    auto num_additional_line = proj_matcher_.match_frame_and_keyframe_line(curr_frm, reloc_candidates.at(i), already_found_landmarks_line, 3, 64);
                    spdlog::info("3D lines found after relocalization: {}", num_additional_line);

                    // Optimize again
                    if (num_additional_line > 0)
                    {
                        num_valid_obs = _pose_optimizer_extended_line.optimize(curr_frm);
                    }
                    else
                    {
                        num_valid_obs = pose_optimizer_.optimize(curr_frm);
                    }

                    // Discard if less than threshold
                    if (num_valid_obs < min_num_valid_obs_)
                    {
                        continue;
                    }
                }

                // relocalize successfully
                spdlog::info("relocalization succeeded");
                // TODO: Set the reference key frame for the current frame

                // reject outliers
                for (unsigned int idx = 0; idx < curr_frm.num_keypts_; ++idx)
                {
                    if (!curr_frm.outlier_flags_.at(idx))
                    {
                        continue;
                    }
                    curr_frm.landmarks_.at(idx) = nullptr;
                }

                // FW:
                for (unsigned int idx = 0; idx < curr_frm._num_keylines; ++idx)
                {
                    if (!curr_frm._outlier_flags_line.at(idx))
                    {
                        continue;
                    }
                    curr_frm._landmarks_line.at(idx) = nullptr;
                }

                return true;
            }

            curr_frm.cam_pose_cw_is_valid_ = false;
            return false;
        }

        std::vector<unsigned int> relocalizer::extract_valid_indices(const std::vector<data::landmark *> &landmarks) const
        {
            std::vector<unsigned int> valid_indices;
            valid_indices.reserve(landmarks.size());
            for (unsigned int idx = 0; idx < landmarks.size(); ++idx)
            {
                auto lm = landmarks.at(idx);
                if (!lm)
                {
                    continue;
                }
                if (lm->will_be_erased())
                {
                    continue;
                }
                valid_indices.push_back(idx);
            }
            return valid_indices;
        }

        std::unique_ptr<solve::pnp_solver> relocalizer::setup_pnp_solver(const std::vector<unsigned int> &valid_indices,
                                                                         const eigen_alloc_vector<Vec3_t> &bearings,
                                                                         const std::vector<cv::KeyPoint> &keypts,
                                                                         const std::vector<data::landmark *> &matched_landmarks,
                                                                         const std::vector<float> &scale_factors) const
        {
            // resample valid elements
            const auto valid_bearings = util::resample_by_indices(bearings, valid_indices);
            const auto valid_keypts = util::resample_by_indices(keypts, valid_indices);
            const auto valid_assoc_lms = util::resample_by_indices(matched_landmarks, valid_indices);
            eigen_alloc_vector<Vec3_t> valid_landmarks(valid_indices.size());
            for (unsigned int i = 0; i < valid_indices.size(); ++i)
            {
                valid_landmarks.at(i) = valid_assoc_lms.at(i)->get_pos_in_world();
            }
            // setup PnP solver
            return std::unique_ptr<solve::pnp_solver>(new solve::pnp_solver(valid_bearings, valid_keypts, valid_landmarks, scale_factors));
        }

    } // namespace module
} // namespace PLPSLAM
