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

#ifndef PLPSLAM_MODULE_RELOCALIZER_H
#define PLPSLAM_MODULE_RELOCALIZER_H

#include "PLPSLAM/match/bow_tree.h"
#include "PLPSLAM/match/projection.h"
#include "PLPSLAM/optimize/pose_optimizer.h"
#include "PLPSLAM/solve/pnp_solver.h"

#include "PLPSLAM/optimize/pose_optimizer_extended_line.h"

namespace PLPSLAM
{

    namespace data
    {
        class frame;
        class bow_database;
    } // namespace data

    namespace module
    {

        class relocalizer
        {
        public:
            //! Constructor
            explicit relocalizer(data::bow_database *bow_db,
                                 const double bow_match_lowe_ratio = 0.75, const double proj_match_lowe_ratio = 0.9,
                                 const unsigned int min_num_bow_matches = 20, const unsigned int min_num_valid_obs = 50);

            //! Destructor
            virtual ~relocalizer();

            //! Relocalize the specified frame
            bool relocalize(data::frame &curr_frm);

        private:
            //! Extract valid (non-deleted) landmarks from landmark vector
            std::vector<unsigned int> extract_valid_indices(const std::vector<data::landmark *> &landmarks) const;

            //! Setup PnP solver with the specified 2D-3D matches
            std::unique_ptr<solve::pnp_solver> setup_pnp_solver(const std::vector<unsigned int> &valid_indices,
                                                                const eigen_alloc_vector<Vec3_t> &bearings,
                                                                const std::vector<cv::KeyPoint> &keypts,
                                                                const std::vector<data::landmark *> &matched_landmarks,
                                                                const std::vector<float> &scale_factors) const;

            //! BoW database
            data::bow_database *bow_db_;

            //! minimum threshold of the number of BoW matches
            const unsigned int min_num_bow_matches_;
            //! minimum threshold of the number of valid (= inlier after pose optimization) matches
            const unsigned int min_num_valid_obs_;

            //! BoW matcher
            const match::bow_tree bow_matcher_;
            //! projection matcher
            const match::projection proj_matcher_;
            //! pose optimizer
            const optimize::pose_optimizer pose_optimizer_;

            // FW:
            const optimize::pose_optimizer_extended_line _pose_optimizer_extended_line;
        };

    } // namespace module
} // namespace PLPSLAM

#endif // PLPSLAM_MODULE_RELOCALIZER_H
