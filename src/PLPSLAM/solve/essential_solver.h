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

#ifndef PLPSLAM_SOLVE_ESSENTIAL_SOLVER_H
#define PLPSLAM_SOLVE_ESSENTIAL_SOLVER_H

#include "PLPSLAM/type.h"

#include <vector>

#include <opencv2/core.hpp>

namespace PLPSLAM
{
    namespace solve
    {

        class essential_solver
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            //! Constructor
            essential_solver(const eigen_alloc_vector<Vec3_t> &bearings_1, const eigen_alloc_vector<Vec3_t> &bearings_2,
                             const std::vector<std::pair<int, int>> &matches_12);

            //! Destructor
            virtual ~essential_solver() = default;

            //! Find the most reliable essential matrix via RANSAC
            void find_via_ransac(const unsigned int max_num_iter, const bool recompute = true);

            //! Check if the solution is valid or not
            bool solution_is_valid() const
            {
                return solution_is_valid_;
            }

            //! Get the best score
            double get_best_score() const
            {
                return best_score_;
            }

            //! Get the most reliable essential matrix
            Mat33_t get_best_E_21() const
            {
                return best_E_21_;
            }

            //! Get the inlier matches
            std::vector<bool> get_inlier_matches() const
            {
                return is_inlier_match_;
            }

            //! Compute an essential matrix with 8-point algorithm
            static Mat33_t compute_E_21(const eigen_alloc_vector<Vec3_t> &bearings_1, const eigen_alloc_vector<Vec3_t> &bearings_2);

            //! Decompose an essential matrix to four pairs of rotation and translation
            static bool decompose(const Mat33_t &E_21, eigen_alloc_vector<Mat33_t> &init_rots, eigen_alloc_vector<Vec3_t> &init_transes);

            //! Create an essential matrix from camera poses
            static Mat33_t create_E_21(const Mat33_t &rot_1w, const Vec3_t &trans_1w, const Mat33_t &rot_2w, const Vec3_t &trans_2w);

        private:
            //! Check inliers of the epipolar constraint
            //! (Note: inlier flags are set to `inlier_match` and a score is returned)
            float check_inliers(const Mat33_t &E_21, std::vector<bool> &is_inlier_match);

            //! bearing vectors of shot 1
            const eigen_alloc_vector<Vec3_t> &bearings_1_;
            //! bearing vectors of shot 2
            const eigen_alloc_vector<Vec3_t> &bearings_2_;
            //! matched indices between shots 1 and 2
            const std::vector<std::pair<int, int>> &matches_12_;

            //! solution is valid or not
            bool solution_is_valid_ = false;
            //! best score of RANSAC
            double best_score_ = 0.0;
            //! most reliable essential matrix
            Mat33_t best_E_21_;
            //! inlier matches computed via RANSAC
            std::vector<bool> is_inlier_match_;
        };

    } // namespace solve
} // namespace PLPSLAM

#endif // PLPSLAM_SOLVE_ESSENTIAL_SOLVER_H
