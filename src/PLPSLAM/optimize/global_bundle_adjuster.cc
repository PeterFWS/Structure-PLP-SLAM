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
#include "PLPSLAM/optimize/global_bundle_adjuster.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex_container.h"
#include "PLPSLAM/optimize/g2o/se3/shot_vertex_container.h"
#include "PLPSLAM/optimize/g2o/se3/reproj_edge_wrapper.h"
#include "PLPSLAM/util/converter.h"

#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <spdlog/spdlog.h>

// FW:
#include "PLPSLAM/data/landmark_line.h"
#include "PLPSLAM/optimize/g2o/line3d.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex_line3d.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex_container_line3d.h"

namespace PLPSLAM
{
    namespace optimize
    {

        global_bundle_adjuster::global_bundle_adjuster(data::map_database *map_db,
                                                       const unsigned int num_iter,
                                                       const bool use_huber_kernel)
            : map_db_(map_db),
              num_iter_(num_iter),
              use_huber_kernel_(use_huber_kernel)
        {
        }

        void global_bundle_adjuster::optimize(const unsigned int lead_keyfrm_id_in_global_BA, bool *const force_stop_flag) const
        {
            // [1] collect data

            const auto keyfrms = map_db_->get_all_keyframes();
            auto lms = map_db_->get_all_landmarks();

            std::vector<bool> is_optimized_lm(lms.size(), true);

            // FW:
            auto lms_line = map_db_->get_all_landmarks_line();
            std::vector<bool> is_optimized_lm_line(lms_line.size(), true);

            // [2] build optimizer

            auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverCSparse<::g2o::BlockSolverX::PoseMatrixType>>();
            auto block_solver = ::g2o::make_unique<::g2o::BlockSolverX>(std::move(linear_solver));
            auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

            ::g2o::SparseOptimizer optimizer;
            optimizer.setAlgorithm(algorithm);

            if (force_stop_flag)
            {
                optimizer.setForceStopFlag(force_stop_flag);
            }

            // [3] convert keyframe to g2o vertex and set to optimizer

            // shot vertex of container
            g2o::se3::shot_vertex_container keyfrm_vtx_container(0, keyfrms.size());

            // set keyframes to optimizer
            for (const auto keyfrm : keyfrms)
            {
                if (!keyfrm)
                {
                    continue;
                }
                if (keyfrm->will_be_erased())
                {
                    continue;
                }

                auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(keyfrm, keyfrm->id_ == 0);
                optimizer.addVertex(keyfrm_vtx);
            }

            // [4] Connect keyframe and landmark vertex with reprojection edge

            // landmark vertex of container
            g2o::landmark_vertex_container lm_vtx_container(keyfrm_vtx_container.get_max_vertex_id() + 1, lms.size());

            // reprojection edge of container
            using reproj_edge_wrapper = g2o::se3::reproj_edge_wrapper<data::keyframe>;
            std::vector<reproj_edge_wrapper> reproj_edge_wraps;
            reproj_edge_wraps.reserve(10 * lms.size());

            // Chi-square value with significance level of 5%
            // Degree of freedom n=2
            constexpr float chi_sq_2D = 5.99146;
            const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
            // Degree of freedom n=3
            constexpr float chi_sq_3D = 7.81473;
            const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

            for (unsigned int i = 0; i < lms.size(); ++i)
            {
                auto lm = lms.at(i);
                if (!lm)
                {
                    continue;
                }
                if (lm->will_be_erased())
                {
                    continue;
                }

                // Convert landmark to g2o vertex and set to optimizer
                auto lm_vtx = lm_vtx_container.create_vertex(lm, false);
                optimizer.addVertex(lm_vtx);

                unsigned int num_edges = 0;
                const auto observations = lm->get_observations();
                for (const auto &obs : observations)
                {
                    auto keyfrm = obs.first;
                    auto idx = obs.second;
                    if (!keyfrm)
                    {
                        continue;
                    }
                    if (keyfrm->will_be_erased())
                    {
                        continue;
                    }

                    if (!keyfrm_vtx_container.contain(keyfrm))
                    {
                        continue;
                    }

                    const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
                    const auto &undist_keypt = keyfrm->undist_keypts_.at(idx);
                    const float x_right = keyfrm->stereo_x_right_.at(idx);
                    const float inv_sigma_sq = keyfrm->inv_level_sigma_sq_.at(undist_keypt.octave);
                    const auto sqrt_chi_sq = (keyfrm->camera_->setup_type_ == camera::setup_type_t::Monocular) ? sqrt_chi_sq_2D : sqrt_chi_sq_3D;
                    auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, lm, lm_vtx,
                                                                idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                                inv_sigma_sq, sqrt_chi_sq, use_huber_kernel_);
                    reproj_edge_wraps.push_back(reproj_edge_wrap);
                    optimizer.addEdge(reproj_edge_wrap.edge_);
                    ++num_edges;
                }

                if (num_edges == 0)
                {
                    optimizer.removeVertex(lm_vtx);
                    is_optimized_lm.at(i) = false;
                }
            }

            // FW: add 3D line as vertex, and edge with keyframe
            g2o::landmark_vertex_container_line3d line3d_vtx_contrainer(lm_vtx_container.get_max_vertex_id() + 1, lms_line.size());
            std::vector<reproj_edge_wrapper> reproj_edge_wraps_for_line3d;
            reproj_edge_wraps_for_line3d.reserve(10 * lms_line.size());
            if (map_db_->_b_use_line_tracking && !lms_line.empty())
            {
                for (unsigned int i = 0; i < lms_line.size(); ++i)
                {
                    auto lm_line = lms_line.at(i);
                    if (!lm_line)
                    {
                        continue;
                    }
                    if (lm_line->will_be_erased())
                    {
                        continue;
                    }

                    // Convert landmark to g2o vertex and set to optimizer
                    auto line3d_vtx = line3d_vtx_contrainer.create_vertex(lm_line, false);
                    optimizer.addVertex(line3d_vtx);

                    unsigned int num_edges = 0;
                    const auto observations = lm_line->get_observations();
                    for (const auto &obs : observations)
                    {
                        auto keyfrm = obs.first;
                        auto idx = obs.second;
                        if (!keyfrm)
                        {
                            continue;
                        }
                        if (keyfrm->will_be_erased())
                        {
                            continue;
                        }

                        if (!keyfrm_vtx_container.contain(keyfrm))
                        {
                            continue;
                        }

                        const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
                        const auto &keyline = keyfrm->_keylsd.at(idx);
                        const float inv_sigma_sq = keyfrm->_inv_level_sigma_sq_lsd.at(keyline.octave);
                        const auto sqrt_chi_sq = sqrt_chi_sq_2D;

                        auto line_reproj_edge = reproj_edge_wrapper(keyfrm, keyfrm_vtx, line3d_vtx, idx,
                                                                    keyline.getStartPoint(), keyline.getEndPoint(),
                                                                    inv_sigma_sq, sqrt_chi_sq);

                        reproj_edge_wraps_for_line3d.push_back(line_reproj_edge);
                        optimizer.addEdge(line_reproj_edge.edge_);
                        ++num_edges;
                    }

                    if (num_edges == 0)
                    {
                        optimizer.removeVertex(line3d_vtx);
                        is_optimized_lm_line.at(i) = false;
                    }
                }
            }

            // [5] Perform optimization

            optimizer.initializeOptimization();
            optimizer.optimize(num_iter_);

            if (force_stop_flag && *force_stop_flag)
            {
                return;
            }

            // [7] Retrieve results

            for (auto keyfrm : keyfrms)
            {
                if (keyfrm->will_be_erased())
                {
                    continue;
                }
                auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
                const auto cam_pose_cw = util::converter::to_eigen_mat(keyfrm_vtx->estimate());
                if (lead_keyfrm_id_in_global_BA == 0)
                {
                    keyfrm->set_cam_pose(cam_pose_cw);
                }
                else
                {
                    keyfrm->cam_pose_cw_after_loop_BA_ = cam_pose_cw;
                    keyfrm->loop_BA_identifier_ = lead_keyfrm_id_in_global_BA;
                }
            }

            for (unsigned int i = 0; i < lms.size(); ++i)
            {
                if (!is_optimized_lm.at(i))
                {
                    continue;
                }

                auto lm = lms.at(i);
                if (!lm)
                {
                    continue;
                }
                if (lm->will_be_erased())
                {
                    continue;
                }

                auto lm_vtx = lm_vtx_container.get_vertex(lm);
                const Vec3_t pos_w = lm_vtx->estimate();

                if (lead_keyfrm_id_in_global_BA == 0)
                {
                    lm->set_pos_in_world(pos_w);
                    lm->update_normal_and_depth();
                }
                else
                {
                    lm->pos_w_after_global_BA_ = pos_w;
                    lm->loop_BA_identifier_ = lead_keyfrm_id_in_global_BA;
                }
            }

            // FW:
            if (map_db_->_b_use_line_tracking && !lms_line.empty())
            {
                for (unsigned int i = 0; i < lms_line.size(); ++i)
                {
                    if (!is_optimized_lm_line.at(i))
                    {
                        continue;
                    }

                    auto lm_line = lms_line.at(i);
                    if (!lm_line)
                    {
                        continue;
                    }
                    if (lm_line->will_be_erased())
                    {
                        continue;
                    }

                    auto lm_line_vtx = line3d_vtx_contrainer.get_vertex(lm_line);
                    Vec6_t pos_w_pluecker = lm_line_vtx->estimate();

                    if (lead_keyfrm_id_in_global_BA == 0)
                    {
                        lm_line->set_PlueckerCoord_without_update_endpoints(pos_w_pluecker);

                        // endpoint trimming via its reference keyframe
                        Vec6_t updated_pose_w;
                        if (endpoint_trimming(lm_line, pos_w_pluecker, updated_pose_w))
                        {
                            lm_line->set_pos_in_world_without_update_pluecker(updated_pose_w);
                            lm_line->update_information();
                        }
                        else
                        {
                            lm_line->prepare_for_erasing(); //  outlier found by trimming
                        }
                    }
                    else
                    {
                        lm_line->_pos_w_after_global_BA = pos_w_pluecker;
                        lm_line->_loop_BA_identifier = lead_keyfrm_id_in_global_BA;
                    }
                }
            }
        }

        // FW:
        bool global_bundle_adjuster::endpoint_trimming(data::Line *local_lm_line,
                                                       const Vec6_t &plucker_coord,
                                                       Vec6_t &updated_pose_w) const
        {
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

            // check positive depth
            Vec4_t pos_w_sp;
            pos_w_sp << intersect_endpoint_sp(0) / intersect_endpoint_sp(3),
                intersect_endpoint_sp(1) / intersect_endpoint_sp(3),
                intersect_endpoint_sp(2) / intersect_endpoint_sp(3),
                1.0;

            Vec4_t pos_w_ep;
            pos_w_ep << intersect_endpoint_ep(0) / intersect_endpoint_ep(3),
                intersect_endpoint_ep(1) / intersect_endpoint_ep(3),
                intersect_endpoint_ep(2) / intersect_endpoint_ep(3),
                1.0;

            Vec3_t pos_c_sp = rotation_translation_combined * pos_w_sp;
            Vec3_t pos_c_ep = rotation_translation_combined * pos_w_ep;

            return 0 < pos_c_sp(2) && 0 < pos_c_ep(2);
        }

    } // namespace optimize
} // namespace PLPSLAM
