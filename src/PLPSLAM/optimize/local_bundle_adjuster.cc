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
#include "PLPSLAM/data/map_database.h"
#include "PLPSLAM/optimize/local_bundle_adjuster.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex_container.h"
#include "PLPSLAM/optimize/g2o/se3/shot_vertex_container.h"
#include "PLPSLAM/optimize/g2o/se3/reproj_edge_wrapper.h"
#include "PLPSLAM/util/converter.h"

#include <unordered_map>

#include <Eigen/StdVector>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <spdlog/spdlog.h>

namespace PLPSLAM
{
    namespace optimize
    {

        local_bundle_adjuster::local_bundle_adjuster(data::map_database *map_db,
                                                     const unsigned int num_first_iter,
                                                     const unsigned int num_second_iter)
            : _map_db(map_db),
              num_first_iter_(num_first_iter),
              num_second_iter_(num_second_iter)
        {
        }

        void local_bundle_adjuster::optimize(data::keyframe *curr_keyfrm, bool *const force_stop_flag) const
        {
            // FW: the local BA optimizer only happens when a keyframe is detected and insterted into the map
            //  both 3D point and keyframe will be optimized (called in mapping_module::mapping_with_new_keyframe())

            // --------------------------------------------------------------------------------------------------------------------------------------
            // ------------------------------------------[1] Aggregate local/fixed keyframes and local landmarks-------------------------------------
            // --------------------------------------------------------------------------------------------------------------------------------------

            // find local keyframes of the current keyframe
            std::unordered_map<unsigned int, data::keyframe *> local_keyfrms; // id <-> keyframe*
            local_keyfrms[curr_keyfrm->id_] = curr_keyfrm;
            const auto curr_covisibilities = curr_keyfrm->graph_node_->get_covisibilities(); // std::vector<keyframe *>

            // Aggregate local keyframes
            // loop though all the keyframes which have co-visibility -> observe common 3D points
            for (auto local_keyfrm : curr_covisibilities)
            {
                if (!local_keyfrm)
                {

                    continue;
                }
                if (local_keyfrm->will_be_erased())
                {
                    continue;
                }

                local_keyfrms[local_keyfrm->id_] = local_keyfrm;
            }

            // optimize local landmarks seen in local keyframes
            std::unordered_map<unsigned int, data::landmark *> local_lms; // id <-> 3D point*

            // Aggregate local landmarks
            // loop through all the local keyframes to find local landmarks
            for (auto local_keyfrm : local_keyfrms)
            {
                const auto landmarks = local_keyfrm.second->get_landmarks();
                for (auto local_lm : landmarks)
                {
                    if (!local_lm)
                    {
                        continue;
                    }
                    if (local_lm->will_be_erased())
                    {
                        continue;
                    }

                    // Avoid duplication
                    if (local_lms.count(local_lm->id_))
                    {
                        // if count = 1
                        continue;
                    }

                    local_lms[local_lm->id_] = local_lm;
                }
            }

            // "fixed" keyframes: keyframes which observe local landmarks but which are NOT in local keyframes
            // those keyframe will be included in optimization, but the pose will not be updated
            std::unordered_map<unsigned int, data::keyframe *> fixed_keyfrms;

            // Aggregate fixed keyframes
            // loop through all the local landmarks we found before to find more (fixed) keyframe
            for (auto local_lm : local_lms)
            {
                const auto observations = local_lm.second->get_observations(); // std::map<keyframe *, unsigned int>
                for (auto &obs : observations)
                {
                    auto fixed_keyfrm = obs.first;
                    if (!fixed_keyfrm)
                    {
                        continue;
                    }
                    if (fixed_keyfrm->will_be_erased())
                    {
                        continue;
                    }

                    // Do not add if it belongs to local keyframes
                    if (local_keyfrms.count(fixed_keyfrm->id_))
                    {
                        continue;
                    }

                    // Avoid duplication
                    if (fixed_keyfrms.count(fixed_keyfrm->id_))
                    {
                        continue;
                    }

                    fixed_keyfrms[fixed_keyfrm->id_] = fixed_keyfrm;
                }
            }

            // --------------------------------------------------------------------------------------------------------------------------------------
            // -------------------------------------------------------[2] Build optimizer------------------------------------------------------------
            // --------------------------------------------------------------------------------------------------------------------------------------

            // The solver type defines the method to solve the matrix inverse and the structure of the sparse matrix.
            // define solver type, such as linear solver, here use Csparse library as the backend solver
            auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverCSparse<::g2o::BlockSolver_6_3::PoseMatrixType>>();
            // create a solver
            auto block_solver = ::g2o::make_unique<::g2o::BlockSolver_6_3>(std::move(linear_solver));
            // create the optimization algorithm, such as Gauss-Newton, Gradient-Descent , Levenberg-Marquardt
            auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

            // declaring an optimizer
            ::g2o::SparseOptimizer optimizer;
            // setup the optimizer
            optimizer.setAlgorithm(algorithm);
            optimizer.setVerbose(false);

            if (force_stop_flag)
            {
                optimizer.setForceStopFlag(force_stop_flag);
            }

            // --------------------------------------------------------------------------------------------------------------------------------------
            // ---------------------------------------[3] Convert keyframe to g2o vertex and set to optimizer----------------------------------------
            // --------------------------------------------------------------------------------------------------------------------------------------

            // shot vertex container, just a vector saves all the vertex
            g2o::se3::shot_vertex_container keyfrm_vtx_container(0, local_keyfrms.size() + fixed_keyfrms.size());
            // Save the converted keyframes to vertex
            std::unordered_map<unsigned int, data::keyframe *> all_keyfrms;

            // Set local keyframes to optimizer
            for (auto &id_local_keyfrm_pair : local_keyfrms)
            {
                auto local_keyfrm = id_local_keyfrm_pair.second;

                all_keyfrms.emplace(id_local_keyfrm_pair);
                auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(local_keyfrm, local_keyfrm->id_ == 0); // local_keyfrm->id_ == 0 return false, this vertex will not remain constant
                optimizer.addVertex(keyfrm_vtx);
            }

            // Set fixed keyframes to optimizer
            for (auto &id_fixed_keyfrm_pair : fixed_keyfrms)
            {
                auto fixed_keyfrm = id_fixed_keyfrm_pair.second;

                all_keyfrms.emplace(id_fixed_keyfrm_pair);
                auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(fixed_keyfrm, true); // true: this vertex will remain constant
                optimizer.addVertex(keyfrm_vtx);
            }

            // --------------------------------------------------------------------------------------------------------------------------------------
            // ------------------------------------------------[4] Connect keyframe and landmark vertex with reprojection edge-----------------------
            // --------------------------------------------------------------------------------------------------------------------------------------

            // landmark vertex container
            g2o::landmark_vertex_container lm_vtx_container(keyfrm_vtx_container.get_max_vertex_id() + 1, local_lms.size());

            // container of reprojection edge
            using reproj_edge_wrapper = g2o::se3::reproj_edge_wrapper<data::keyframe>;
            std::vector<reproj_edge_wrapper> reproj_edge_wraps;
            reproj_edge_wraps.reserve(all_keyfrms.size() * local_lms.size());

            // Chi-square value with significance level of 5%
            // Degrees of freedom n=2
            constexpr float chi_sq_2D = 5.99146;
            const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
            // Degrees of freedom n=3
            constexpr float chi_sq_3D = 7.81473;
            const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

            for (auto &id_local_lm_pair : local_lms)
            {
                auto local_lm = id_local_lm_pair.second;

                // Convert landmark to g2o vertex and set to optimizer
                auto lm_vtx = lm_vtx_container.create_vertex(local_lm, false); // false -> vertex not constant, will be optimized
                optimizer.addVertex(lm_vtx);

                // add point-keyframe edge
                const auto observations = local_lm->get_observations();
                for (const auto &obs : observations)
                {
                    auto keyfrm = obs.first;
                    auto idx = obs.second; // keypoint's id
                    if (!keyfrm)
                    {
                        continue;
                    }
                    if (keyfrm->will_be_erased())
                    {
                        continue;
                    }

                    const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm); // keyframe vertex
                    const auto &undist_keypt = keyfrm->undist_keypts_.at(idx);       // undistorted keypoint
                    const float x_right = keyfrm->stereo_x_right_.at(idx);           // if monocular, x_right <0
                    const float inv_sigma_sq = keyfrm->inv_level_sigma_sq_.at(undist_keypt.octave);
                    const auto sqrt_chi_sq = (keyfrm->camera_->setup_type_ == camera::setup_type_t::Monocular)
                                                 ? sqrt_chi_sq_2D
                                                 : sqrt_chi_sq_3D;

                    // create reprojection_error edge between keyframe vertex and landmark vertex
                    auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, local_lm, lm_vtx,
                                                                idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                                inv_sigma_sq, sqrt_chi_sq);

                    // aggregate edges
                    reproj_edge_wraps.push_back(reproj_edge_wrap);
                    optimizer.addEdge(reproj_edge_wrap.edge_);
                }
            }

            // --------------------------------------------------------------------------------------------------------------------------------------
            // ---------------------------------------------------- [5] Run the first optimization---------------------------------------------------
            // --------------------------------------------------------------------------------------------------------------------------------------

            if (force_stop_flag)
            {
                if (*force_stop_flag)
                {
                    return;
                }
            }

            optimizer.initializeOptimization();  // initialize the optimizer we built before in steps [1]-[4]
            optimizer.optimize(num_first_iter_); // run the optimizer, num_first_iter_ = 5

            // --------------------------------------------------------------------------------------------------------------------------------------
            // ---------------------------------------------------[6] Remove outliers and run a second optimization----------------------------------
            // --------------------------------------------------------------------------------------------------------------------------------------

            bool run_robust_BA = true;

            if (force_stop_flag)
            {
                if (*force_stop_flag)
                {
                    run_robust_BA = false;
                }
            }

            if (run_robust_BA)
            {
                for (auto &reproj_edge_wrap : reproj_edge_wraps)
                {
                    auto edge = reproj_edge_wrap.edge_;

                    auto local_lm = reproj_edge_wrap.lm_;
                    if (local_lm->will_be_erased())
                    {
                        continue;
                    }

                    if (reproj_edge_wrap.is_monocular_)
                    {
                        if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive())
                        {
                            reproj_edge_wrap.set_as_outlier();
                        }
                    }
                    else
                    {
                        if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive())
                        {
                            reproj_edge_wrap.set_as_outlier();
                        }
                    }

                    edge->setRobustKernel(nullptr);
                }

                // run the second BA, after removing outliers from the graph
                optimizer.initializeOptimization();
                optimizer.optimize(num_second_iter_); // num_second_iter_ = 10
            }

            // --------------------------------------------------------------------------------------------------------------------------------------
            // ---------------------------------------------[7] Aggregate outliers-------------------------------------------------------------------
            // --------------------------------------------------------------------------------------------------------------------------------------

            std::vector<std::pair<data::keyframe *, data::landmark *>> outlier_observations;
            outlier_observations.reserve(reproj_edge_wraps.size());

            for (auto &reproj_edge_wrap : reproj_edge_wraps)
            {
                auto edge = reproj_edge_wrap.edge_;

                auto local_lm = reproj_edge_wrap.lm_;
                if (local_lm->will_be_erased())
                {
                    continue;
                }

                if (reproj_edge_wrap.is_monocular_)
                {
                    if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive())
                    {
                        outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
                    }
                }
                else
                {
                    if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive())
                    {
                        outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
                    }
                }
            }

            // --------------------------------------------------------------------------------------------------------------------------------------
            // --------------------------------------------------[8] Update information--------------------------------------------------------------
            // --------------------------------------------------------------------------------------------------------------------------------------

            {
                // lock the map database
                std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

                // remove outlier landmark from keyframe, and remove the link between landmark to keyframe
                if (!outlier_observations.empty())
                {
                    for (auto &outlier_obs : outlier_observations) // keyframe* <-> 3D point*
                    {
                        auto keyfrm = outlier_obs.first;
                        auto lm = outlier_obs.second;
                        keyfrm->erase_landmark(lm);
                        lm->erase_observation(keyfrm);
                    }
                }

                // Update pose of local keyframe
                for (auto id_local_keyfrm_pair : local_keyfrms)
                {
                    auto local_keyfrm = id_local_keyfrm_pair.second;

                    auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(local_keyfrm);
                    local_keyfrm->set_cam_pose(keyfrm_vtx->estimate());
                }

                // Update 3D position of landmark
                for (auto id_local_lm_pair : local_lms)
                {
                    auto local_lm = id_local_lm_pair.second;

                    auto lm_vtx = lm_vtx_container.get_vertex(local_lm);
                    local_lm->set_pos_in_world(lm_vtx->estimate());
                    local_lm->update_normal_and_depth();
                }
            }
        }

    } // namespace optimize
} // namespace PLPSLAM
