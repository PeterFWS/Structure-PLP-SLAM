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
#include "PLPSLAM/match/area.h"
#include "PLPSLAM/match/angle_checker.h"

namespace PLPSLAM
{
    namespace match
    {

        unsigned int area::match_in_consistent_area(data::frame &frm_1, data::frame &frm_2, std::vector<cv::Point2f> &prev_matched_pts,
                                                    std::vector<int> &matched_indices_2_in_frm_1, int margin)
        {
            unsigned int num_matches = 0;

            angle_checker<int> angle_checker;

            matched_indices_2_in_frm_1 = std::vector<int>(frm_1.undist_keypts_.size(), -1);

            std::vector<unsigned int> matched_dists_in_frm_2(frm_2.undist_keypts_.size(), MAX_HAMMING_DIST);
            std::vector<int> matched_indices_1_in_frm_2(frm_2.undist_keypts_.size(), -1);

            for (unsigned int idx_1 = 0; idx_1 < frm_1.undist_keypts_.size(); ++idx_1)
            {
                const auto &undist_keypt_1 = frm_1.undist_keypts_.at(idx_1); // cv::KeyPoint
                const auto scale_level_1 = undist_keypt_1.octave;            // const int, octave (pyramid layer) from which the keypoint has been extracted

                // Use only 0th scale feature points
                if (0 < scale_level_1)
                {
                    continue;
                }

                // Bring the feature points of the cells around the feature point that matched the previous one
                const auto indices = frm_2.get_keypoints_in_cell(prev_matched_pts.at(idx_1).x, prev_matched_pts.at(idx_1).y,
                                                                 margin, scale_level_1, scale_level_1);
                if (indices.empty())
                {
                    continue;
                }

                const auto &desc_1 = frm_1.descriptors_.row(idx_1);

                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;
                int best_idx_2 = -1;

                for (const auto idx_2 : indices)
                {
                    const auto &desc_2 = frm_2.descriptors_.row(idx_2);

                    const auto hamm_dist = compute_descriptor_distance_32(desc_1, desc_2);

                    // Through if the already matched points are closer
                    if (matched_dists_in_frm_2.at(idx_2) <= hamm_dist)
                    {
                        continue;
                    }

                    if (hamm_dist < best_hamm_dist)
                    {
                        second_best_hamm_dist = best_hamm_dist;
                        best_hamm_dist = hamm_dist;
                        best_idx_2 = idx_2;
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

                // ratio test
                if (second_best_hamm_dist * lowe_ratio_ < static_cast<float>(best_hamm_dist))
                {
                    continue;
                }

                // idx_1 - best_idx_2 is presumed to be the best response

                // If the point corresponding to best_idx_2 already exists (= prev_idx_1),
                // It is necessary to delete the corresponding information of the corresponding matched_indices_2_in_frm_1.at (prev_idx_1) in order to overwrite it.
                // (matched_indices_1_in_frm_2.at (best_idx_2) is overwritten and does not need to be deleted)
                const auto prev_idx_1 = matched_indices_1_in_frm_2.at(best_idx_2);
                if (0 <= prev_idx_1)
                {
                    matched_indices_2_in_frm_1.at(prev_idx_1) = -1;
                    --num_matches;
                }

                // Record each other's correspondence information
                matched_indices_2_in_frm_1.at(idx_1) = best_idx_2;
                matched_indices_1_in_frm_2.at(best_idx_2) = idx_1;
                matched_dists_in_frm_2.at(best_idx_2) = best_hamm_dist;
                ++num_matches;

                if (check_orientation_)
                {
                    const auto delta_angle = frm_1.undist_keypts_.at(idx_1).angle - frm_2.undist_keypts_.at(best_idx_2).angle;
                    angle_checker.append_delta_angle(delta_angle, idx_1);
                }
            }

            if (check_orientation_)
            {
                const auto invalid_matches = angle_checker.get_invalid_matches();
                for (const auto invalid_idx_1 : invalid_matches)
                {
                    if (0 <= matched_indices_2_in_frm_1.at(invalid_idx_1))
                    {
                        matched_indices_2_in_frm_1.at(invalid_idx_1) = -1;
                        --num_matches;
                    }
                }
            }

            // Update previous matches
            for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_frm_1.size(); ++idx_1)
            {
                if (0 <= matched_indices_2_in_frm_1.at(idx_1))
                {
                    prev_matched_pts.at(idx_1) = frm_2.undist_keypts_.at(matched_indices_2_in_frm_1.at(idx_1)).pt;
                }
            }

            return num_matches;
        }

    } // namespace match
} // namespace PLPSLAM
