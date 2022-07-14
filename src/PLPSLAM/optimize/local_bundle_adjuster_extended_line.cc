/**
 * This file is part of Structure PLP-SLAM.
 *
 * Copyright 2022 DFKI (German Research Center for Artificial Intelligence)
 * Developed by Fangwen Shu <Fangwen.Shu@dfki.de>
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

#include "PLPSLAM/optimize/local_bundle_adjuster_extended_line.h"

#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/data/map_database.h"
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

// FW: 3D Line using Plücker coordinates, and orthonormal representation (4DOF minimal representation)
#include "PLPSLAM/data/landmark_line.h"
#include "PLPSLAM/optimize/g2o/line3d.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex_line3d.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex_container_line3d.h"

namespace PLPSLAM
{
    namespace optimize
    {

        local_bundle_adjuster_extended_line::local_bundle_adjuster_extended_line(data::map_database *map_db,
                                                                                 const unsigned int num_first_iter,
                                                                                 const unsigned int num_second_iter)
            : _map_db(map_db),
              num_first_iter_(num_first_iter),
              num_second_iter_(num_second_iter)
        {
        }

        void local_bundle_adjuster_extended_line::optimize(data::keyframe *curr_keyfrm, bool *const force_stop_flag) const
        {
            // FW: the local BA optimizer only happens when a keyframe is detected and inserted into the map
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

            // FW:
            std::unordered_map<unsigned int, data::Line *> local_lms_line; // id <-> 3D line*

            // Aggregate local landmarks (both points and lines)
            // loop through all the local keyframes to find local landmarks
            for (auto local_keyfrm : local_keyfrms)
            {
                // Aggregate 3D points
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

                // FW: Aggregate 3D lines
                const auto landmarks_line = local_keyfrm.second->get_landmarks_line();
                for (auto local_lm_line : landmarks_line)
                {
                    if (!local_lm_line)
                    {
                        continue;
                    }
                    if (local_lm_line->will_be_erased())
                    {
                        continue;
                    }

                    // Avoid duplication
                    if (local_lms_line.count(local_lm_line->_id))
                    {
                        // if count = 1
                        continue;
                    }
                    local_lms_line[local_lm_line->_id] = local_lm_line;
                }
            }

            // "fixed" keyframes: keyframes which observe local landmarks but which are NOT in local keyframes
            // those keyframe will be included in optimization, but the pose will not be updated
            std::unordered_map<unsigned int, data::keyframe *> fixed_keyfrms;

            // FW: Aggregate fixed keyframes
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

            // FW: Aggregate fixed keyframes using local lines
            for (auto local_lm_line : local_lms_line)
            {
                const auto observations = local_lm_line.second->get_observations(); // std::map<keyframe *, unsigned int>

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

            if (_setVerbose)
            {
                spdlog::info("-- LocalBA with line -- [1] Aggregate local/fixed keyframes and local landmarks");
            }

            // --------------------------------------------------------------------------------------------------------------------------------------
            // -------------------------------------------------------[2] Build optimizer------------------------------------------------------------
            // --------------------------------------------------------------------------------------------------------------------------------------

            // The solver type defines the method to solve the matrix inverse and the structure of the sparse matrix.
            // define solver type, such as linear solver, here use Csparse library as the backend solver
            auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverCSparse<::g2o::BlockSolverX::PoseMatrixType>>();
            // create a solver
            auto block_solver = ::g2o::make_unique<::g2o::BlockSolverX>(std::move(linear_solver));
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

            if (_setVerbose)
            {
                spdlog::info("-- LocalBA with line -- [2] Build optimizer");
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

            if (_setVerbose)
            {
                spdlog::info("-- LocalBA with line -- [3] Convert keyframe to g2o vertex and set to optimizer");
            }

            // --------------------------------------------------------------------------------------------------------------------------------------
            // --------------------------------------[4] Connect keyframe and landmark vertex with reprojection edge-----------------------
            // --------------------------------------------------------------------------------------------------------------------------------------

            // Chi-square value with significance level of 5%
            // Degrees of freedom n=2
            constexpr float chi_sq_2D = 5.99146;
            const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
            // Degrees of freedom n=3
            constexpr float chi_sq_3D = 7.81473;
            const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

            // [4.1] standard point landmark
            // landmark vertex container
            g2o::landmark_vertex_container lm_vtx_container(keyfrm_vtx_container.get_max_vertex_id() + 1, local_lms.size());
            // container of reprojection edge
            using reproj_edge_wrapper = g2o::se3::reproj_edge_wrapper<data::keyframe>;
            std::vector<reproj_edge_wrapper> reproj_edge_wraps;
            reproj_edge_wraps.reserve(all_keyfrms.size() * local_lms.size());
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

            if (_setVerbose)
            {
                spdlog::info("-- LocalBA with line -- [4.1] reprojection edge (standard)");
            }

            // FW:
            // [4.2] add 3D line as vertex, and edge with keyframe
            g2o::landmark_vertex_container_line3d line3d_vtx_container(lm_vtx_container.get_max_vertex_id() + 1, local_lms_line.size());
            std::vector<reproj_edge_wrapper> reproj_edge_wraps_for_line3d;
            reproj_edge_wraps_for_line3d.reserve(all_keyfrms.size() * local_lms_line.size());

            if (!local_lms_line.empty())
            {
                for (auto &id_local_lm_line_pair : local_lms_line)
                {
                    auto local_lm_line = id_local_lm_line_pair.second;

                    if (!local_lm_line || local_lm_line->will_be_erased())
                    {
                        continue;
                    }

                    // convert landmark to g2o vertex and set to optimizer
                    auto line3d_vtx = line3d_vtx_container.create_vertex(local_lm_line->_id, local_lm_line->get_PlueckerCoord(), false);
                    optimizer.addVertex(line3d_vtx);

                    // add edge between 3D line and keyframe
                    const auto observations = local_lm_line->get_observations();
                    if (observations.empty())
                    {
                        continue;
                    }

                    for (const auto &obs : observations)
                    {
                        auto keyfrm = obs.first;
                        auto idx = obs.second; // keyline's id
                        if (!keyfrm)
                        {
                            continue;
                        }
                        if (keyfrm->will_be_erased())
                        {
                            continue;
                        }

                        const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm); // keyframe vertex
                        const auto &keyline = keyfrm->_keylsd.at(idx);                   // undistorted keyline
                        const float inv_sigma_sq = keyfrm->_inv_level_sigma_sq_lsd.at(keyline.octave);
                        const auto sqrt_chi_sq = sqrt_chi_sq_2D;

                        // create reprojection_error edge between keyframe vertex and 3D line vertex
                        auto line_reproj_edge = reproj_edge_wrapper(keyfrm, keyfrm_vtx, line3d_vtx, idx,
                                                                    keyline.getStartPoint(), keyline.getEndPoint(),
                                                                    inv_sigma_sq, sqrt_chi_sq);

                        // aggregate edges
                        reproj_edge_wraps_for_line3d.push_back(line_reproj_edge);
                        optimizer.addEdge(line_reproj_edge.edge_);
                    }
                }
            }

            if (_setVerbose)
            {
                spdlog::info("-- LocalBA with line -- [4.2] reprojection edge (keyframe-line)");
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

            if (_setVerbose)
            {
                spdlog::info("-- LocalBA with line -- [5] Run the first optimization");
            }

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

                // FW: remove 3D line outliers
                if (!reproj_edge_wraps_for_line3d.empty())
                {
                    for (auto &line_reproj_edge : reproj_edge_wraps_for_line3d)
                    {
                        auto edge = line_reproj_edge.edge_;
                        auto local_lm_line = line_reproj_edge._lm_line;
                        if (local_lm_line->will_be_erased())
                        {
                            continue;
                        }

                        if (chi_sq_2D < edge->chi2() || !line_reproj_edge.depth_is_positive_via_endpoints_trimming())
                        {
                            line_reproj_edge.set_as_outlier();
                        }

                        edge->setRobustKernel(nullptr);
                    }
                }

                // run the second BA, after removing outliers from the graph
                optimizer.initializeOptimization();
                optimizer.optimize(num_second_iter_); // num_second_iter_ = 10
            }

            if (_setVerbose)
            {
                spdlog::info("-- LocalBA with line -- [6] Run the second optimization");
            }

            // --------------------------------------------------------------------------------------------------------------------------------------
            // ---------------------------------------------[7] Aggregate outliers-------------------------------------------------------------------
            // --------------------------------------------------------------------------------------------------------------------------------------

            std::vector<std::pair<data::keyframe *, data::landmark *>> outlier_observations;
            outlier_observations.reserve(reproj_edge_wraps.size());

            // FW:
            std::vector<std::pair<data::keyframe *, data::Line *>> outlier_observations_line;
            outlier_observations_line.reserve(reproj_edge_wraps_for_line3d.size());

            // point
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

            // FW: aggregate 3D line outliers
            if (!reproj_edge_wraps_for_line3d.empty())
            {
                for (auto &line_reproj_edge : reproj_edge_wraps_for_line3d)
                {
                    auto edge = line_reproj_edge.edge_;

                    auto local_lm_line = line_reproj_edge._lm_line;
                    if (local_lm_line->will_be_erased())
                    {
                        continue;
                    }

                    if (chi_sq_2D < edge->chi2() || !line_reproj_edge.depth_is_positive_via_endpoints_trimming())
                    {
                        outlier_observations_line.emplace_back(std::make_pair(line_reproj_edge.shot_, line_reproj_edge._lm_line));
                    }
                }
            }

            if (_setVerbose)
            {
                spdlog::info("-- LocalBA with line -- [7] Aggregate outliers");
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

                if (_setVerbose)
                {
                    spdlog::info("-- LocalBA with line -- [8.1] remove 3D point observation");
                }

                // FW: remove 3D line observation
                if (!outlier_observations_line.empty())
                {
                    for (auto &outlier_obs_line : outlier_observations_line)
                    {
                        auto keyfrm = outlier_obs_line.first;
                        auto lm_line = outlier_obs_line.second;
                        keyfrm->erase_landmark_line(lm_line);
                        lm_line->erase_observation(keyfrm);
                    }
                }

                if (_setVerbose)
                {
                    spdlog::info("-- LocalBA with line -- [8.2] remove 3D line observation");
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

                // FW: Update the position of 3D line
                if (!local_lms_line.empty())
                {
                    for (auto id_local_lm_line_pair : local_lms_line)
                    {
                        auto local_lm_line = id_local_lm_line_pair.second;

                        auto line3d_vtx = line3d_vtx_container.get_vertex(local_lm_line->_id);
                        auto pos_w_pluecker = line3d_vtx->estimate(); // Vec6_t: Plücker coordinates

                        // update Plücker coordinates, for optimization
                        local_lm_line->set_PlueckerCoord_without_update_endpoints(pos_w_pluecker);

                        // update endpoints, only for map visualization: endpoints trimming
                        Vec6_t updated_pose_w;
                        if (endpoint_trimming(local_lm_line, pos_w_pluecker, updated_pose_w))
                        {
                            local_lm_line->set_pos_in_world_without_update_pluecker(updated_pose_w);
                            local_lm_line->update_information();
                        }
                        else
                        {
                            local_lm_line->prepare_for_erasing(); //  outlier found by trimming
                        }
                    }
                }
            }

            if (_setVerbose)
            {
                spdlog::info("-- LocalBA with line -- [8] Update information");
                spdlog::info("");
            }
        }

        bool local_bundle_adjuster_extended_line::endpoint_trimming(data::Line *local_lm_line,
                                                                    const Vec6_t &plucker_coord,
                                                                    Vec6_t &updated_pose_w) const
        {
            // FW:
            // references:
            // "Elaborate Monocular Point and Line SLAM with Robust Initialization", ICCV'19
            // "Building a 3-D line-based map using stereo SLAM", IEEE Transactions on Robotics'15

            auto ref_kf = local_lm_line->get_ref_keyframe();
            int idx = local_lm_line->get_index_in_keyframe(ref_kf);

            if (idx == -1)
            {
                return false;
            }

            // [1] get endpoints from the detected line segment
            auto keyline = ref_kf->_keylsd.at(idx);
            auto sp = keyline.getStartPoint();
            auto ep = keyline.getEndPoint();

            // [2] get the line function of re-projected 3D line
            auto camera = static_cast<camera::perspective *>(ref_kf->camera_);
            Mat44_t cam_pose_wc = ref_kf->get_cam_pose();

            const Mat33_t rot_cw = cam_pose_wc.block<3, 3>(0, 0);
            const Vec3_t trans_cw = cam_pose_wc.block<3, 1>(0, 3);

            Mat66_t transformation_line_cw = Eigen::Matrix<double, 6, 6>::Zero();
            transformation_line_cw.block<3, 3>(0, 0) = rot_cw;
            transformation_line_cw.block<3, 3>(3, 3) = rot_cw;
            transformation_line_cw.block<3, 3>(0, 3) = skew(trans_cw) * rot_cw;

            Mat33_t _K;
            _K << camera->fy_, 0.0, 0.0,
                0.0, camera->fx_, 0.0,
                -camera->fy_ * camera->cx_, -camera->fx_ * camera->cy_, camera->fx_ * camera->fy_;

            Vec3_t reproj_line_function;
            reproj_line_function = _K * (transformation_line_cw * plucker_coord).block<3, 1>(0, 0);

            double l1 = reproj_line_function(0);
            double l2 = reproj_line_function(1);
            double l3 = reproj_line_function(2);

            // [3] calculate closet point on the re-projected line
            double x_sp_closet = -(sp.y - (l2 / l1) * sp.x + (l3 / l2)) * ((l1 * l2) / (l1 * l1 + l2 * l2));
            double y_sp_closet = -(l1 / l2) * x_sp_closet - (l3 / l2);

            double x_ep_closet = -(ep.y - (l2 / l1) * ep.x + (l3 / l2)) * ((l1 * l2) / (l1 * l1 + l2 * l2));
            double y_ep_closet = -(l1 / l2) * x_ep_closet - (l3 / l2);

            // [4] calculate another point
            double x_0sp = 0;
            double y_0sp = sp.y - (l2 / l1) * sp.x;

            double x_0ep = 0;
            double y_0ep = ep.y - (l2 / l1) * ep.x;

            // [5] calculate 3D plane
            Mat34_t P;
            Mat34_t rotation_translation_combined = Eigen::Matrix<double, 3, 4>::Zero();
            rotation_translation_combined.block<3, 3>(0, 0) = rot_cw;
            rotation_translation_combined.block<3, 1>(0, 3) = trans_cw;
            P = camera->eigen_cam_matrix_ * rotation_translation_combined;

            Vec3_t point2d_sp_closet{x_sp_closet, y_sp_closet, 1.0};
            Vec3_t point2d_0sp{x_0sp, y_0sp, 1.0};
            Vec3_t line_temp_sp = point2d_sp_closet.cross(point2d_0sp);
            Vec4_t plane3d_temp_sp = P.transpose() * line_temp_sp;

            Vec3_t point2d_ep_closet{x_ep_closet, y_ep_closet, 1.0};
            Vec3_t point2d_0ep{x_0ep, y_0ep, 1.0};
            Vec3_t line_temp_ep = point2d_ep_closet.cross(point2d_0ep);
            Vec4_t plane3d_temp_ep = P.transpose() * line_temp_ep;

            // [6] calculate intersection of the 3D plane and 3d line
            Mat44_t line3d_pluecker_matrix = Eigen::Matrix<double, 4, 4>::Zero();
            Vec3_t m = plucker_coord.head<3>();
            Vec3_t d = plucker_coord.tail<3>();
            line3d_pluecker_matrix.block<3, 3>(0, 0) = skew(m);
            line3d_pluecker_matrix.block<3, 1>(0, 3) = d;
            line3d_pluecker_matrix.block<1, 3>(3, 0) = -d.transpose();

            Vec4_t intersect_endpoint_sp, intersect_endpoint_ep;
            intersect_endpoint_sp = line3d_pluecker_matrix * plane3d_temp_sp;
            intersect_endpoint_ep = line3d_pluecker_matrix * plane3d_temp_ep;

            updated_pose_w << intersect_endpoint_sp(0) / intersect_endpoint_sp(3),
                intersect_endpoint_sp(1) / intersect_endpoint_sp(3),
                intersect_endpoint_sp(2) / intersect_endpoint_sp(3),
                intersect_endpoint_ep(0) / intersect_endpoint_ep(3),
                intersect_endpoint_ep(1) / intersect_endpoint_ep(3),
                intersect_endpoint_ep(2) / intersect_endpoint_ep(3);

            // Debug:
            double distance_of_change_sp = (updated_pose_w.head<3>() - local_lm_line->get_pos_in_world().head<3>()).norm() / ref_kf->compute_median_depth(true);
            double distance_of_change_ep = (updated_pose_w.tail<3>() - local_lm_line->get_pos_in_world().tail<3>()).norm() / ref_kf->compute_median_depth(true);

            // FW: an elegant way of outlier filtering :)
            if (distance_of_change_sp > 0.1 || distance_of_change_ep > 0.1)
            {
                // spdlog::info("distance of change, sp: {}", distance_of_change_sp);
                // spdlog::info("distance of change, ep: {}", distance_of_change_ep);
                // std::cout << std::endl;

                return false; // change is too big, considered as outlier
            }

            return true;
        }

    } // namespace optimize
} // namespace PLPSLAM
