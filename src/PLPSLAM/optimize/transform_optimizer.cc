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

#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/optimize/transform_optimizer.h"
#include "PLPSLAM/optimize/g2o/sim3/transform_vertex.h"
#include "PLPSLAM/optimize/g2o/sim3/mutual_reproj_edge_wrapper.h"

#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

namespace PLPSLAM
{
    namespace optimize
    {

        transform_optimizer::transform_optimizer(const bool fix_scale, const unsigned int num_iter)
            : fix_scale_(fix_scale), num_iter_(num_iter)
        {
        }

        unsigned int transform_optimizer::optimize(data::keyframe *keyfrm_1, data::keyframe *keyfrm_2,
                                                   std::vector<data::landmark *> &matched_lms_in_keyfrm_2,
                                                   ::g2o::Sim3 &g2o_Sim3_12, const float chi_sq) const
        {
            const float sqrt_chi_sq = std::sqrt(chi_sq);

            // 1. build optimizer

            auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverEigen<::g2o::BlockSolverX::PoseMatrixType>>();
            auto block_solver = ::g2o::make_unique<::g2o::BlockSolverX>(std::move(linear_solver));
            auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

            ::g2o::SparseOptimizer optimizer;
            optimizer.setAlgorithm(algorithm);

            // 1. Create a vertex for Sim3 transform

            auto Sim3_12_vtx = new g2o::sim3::transform_vertex();
            Sim3_12_vtx->setId(0);
            Sim3_12_vtx->setEstimate(g2o_Sim3_12);
            Sim3_12_vtx->setFixed(false);
            Sim3_12_vtx->fix_scale_ = fix_scale_;
            Sim3_12_vtx->rot_1w_ = keyfrm_1->get_rotation();
            Sim3_12_vtx->trans_1w_ = keyfrm_1->get_translation();
            Sim3_12_vtx->rot_2w_ = keyfrm_2->get_rotation();
            Sim3_12_vtx->trans_2w_ = keyfrm_2->get_translation();
            optimizer.addVertex(Sim3_12_vtx);

            // 2. Add landmark constraint

            // Wrapper that contains the following two edges
            // - A constraint edge that re-projects the 3D points observed by keyfrm_2 onto keyfrm_1 (the camera model follows that of keyfrm_1)
            // - A constraint edge that reprojects the 3D point observed by keyfrm_1 onto keyfrm_2 (the camera model follows that of keyfrm_2)
            using reproj_edge_wrapper = g2o::sim3::mutual_reproj_edge_wapper<data::keyframe>;
            std::vector<reproj_edge_wrapper> mutual_edges;
            // Corresponding number
            const unsigned int num_matches = matched_lms_in_keyfrm_2.size();
            mutual_edges.reserve(num_matches);

            // All 3D points observed by keyfrm_1
            const auto lms_in_keyfrm_1 = keyfrm_1->get_landmarks();

            // Number of valid responses
            unsigned int num_valid_matches = 0;

            for (unsigned int idx1 = 0; idx1 < num_matches; ++idx1)
            {
                // target only those with matching information
                if (!matched_lms_in_keyfrm_2.at(idx1))
                {
                    continue;
                }

                auto lm_1 = lms_in_keyfrm_1.at(idx1);
                auto lm_2 = matched_lms_in_keyfrm_2.at(idx1);

                // Target only those that have both 3D points
                if (!lm_1 || !lm_2)
                {
                    continue;
                }
                if (lm_1->will_be_erased() || lm_2->will_be_erased())
                {
                    continue;
                }

                const auto idx2 = lm_2->get_index_in_keyframe(keyfrm_2);

                if (idx2 < 0)
                {
                    continue;
                }

                // Create forward/backward edges and set in optimizer
                reproj_edge_wrapper mutual_edge(keyfrm_1, idx1, lm_1, keyfrm_2, idx2, lm_2, Sim3_12_vtx, sqrt_chi_sq);
                optimizer.addEdge(mutual_edge.edge_12_);
                optimizer.addEdge(mutual_edge.edge_21_);

                ++num_valid_matches;
                mutual_edges.push_back(mutual_edge);
            }

            // 3. Perform optimization

            optimizer.initializeOptimization();
            optimizer.optimize(5);

            // 4. Processing to remove outlier

            unsigned int num_outliers = 0;
            for (unsigned int i = 0; i < num_valid_matches; ++i)
            {
                auto edge_12 = mutual_edges.at(i).edge_12_;
                auto edge_21 = mutual_edges.at(i).edge_21_;

                // inlier check
                if (edge_12->chi2() < chi_sq && edge_21->chi2() < chi_sq)
                {
                    continue;
                }

                // Process to make outlier
                const auto idx1 = mutual_edges.at(i).idx1_;
                matched_lms_in_keyfrm_2.at(idx1) = nullptr;

                mutual_edges.at(i).set_as_outlier();
                ++num_outliers;
            }

            if (num_valid_matches - num_outliers < 10)
            {
                return 0;
            }

            // 5. Optimize again

            optimizer.initializeOptimization();
            optimizer.optimize(num_iter_);

            // 6. count inlier

            unsigned int num_inliers = 0;
            for (unsigned int i = 0; i < num_valid_matches; ++i)
            {
                auto edge_12 = mutual_edges.at(i).edge_12_;
                auto edge_21 = mutual_edges.at(i).edge_21_;

                // outlier check
                if (mutual_edges.at(i).is_outlier())
                {
                    continue;
                }

                // outlier check
                if (chi_sq < edge_12->chi2() || chi_sq < edge_21->chi2())
                {
                    // Process to make outlier
                    unsigned int idx1 = mutual_edges.at(i).idx1_;
                    matched_lms_in_keyfrm_2.at(idx1) = nullptr;
                    continue;
                }

                ++num_inliers;
            }

            // 7. Retrieve results

            g2o_Sim3_12 = Sim3_12_vtx->estimate();

            return num_inliers;
        }

    } // namespace optimize
} // namespace PLPSLAM
