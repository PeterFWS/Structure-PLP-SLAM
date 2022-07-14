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
#include "PLPSLAM/optimize/graph_optimizer.h"
#include "PLPSLAM/optimize/g2o/sim3/shot_vertex.h"
#include "PLPSLAM/optimize/g2o/sim3/graph_opt_edge.h"
#include "PLPSLAM/util/converter.h"

#include <Eigen/StdVector>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include "PLPSLAM/data/landmark_line.h"

namespace PLPSLAM
{
    namespace optimize
    {

        graph_optimizer::graph_optimizer(data::map_database *map_db, const bool fix_scale)
            : map_db_(map_db),
              fix_scale_(fix_scale)
        {
        }

        void graph_optimizer::optimize(data::keyframe *loop_keyfrm, data::keyframe *curr_keyfrm,
                                       const module::keyframe_Sim3_pairs_t &non_corrected_Sim3s,
                                       const module::keyframe_Sim3_pairs_t &pre_corrected_Sim3s,
                                       const std::map<data::keyframe *, std::set<data::keyframe *>> &loop_connections) const
        {
            // [1] build optimizer

            auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverCSparse<::g2o::BlockSolver_7_3::PoseMatrixType>>();
            auto block_solver = ::g2o::make_unique<::g2o::BlockSolver_7_3>(std::move(linear_solver));
            auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

            ::g2o::SparseOptimizer optimizer;
            optimizer.setAlgorithm(algorithm);

            // [2] add vertex

            const auto all_keyfrms = map_db_->get_all_keyframes();
            const auto all_lms = map_db_->get_all_landmarks();

            // FW:
            const auto all_lms_line = map_db_->get_all_landmarks_line();
            std::unordered_map<unsigned int, data::keyframe *> keyframes;

            const unsigned int max_keyfrm_id = map_db_->get_max_keyframe_id();

            // Convert all keyframe poses (before modification) to Sim3 and save
            std::vector<::g2o::Sim3, Eigen::aligned_allocator<::g2o::Sim3>> Sim3s_cw(max_keyfrm_id + 1);
            // Save the added vertex
            std::vector<g2o::sim3::shot_vertex *> vertices(max_keyfrm_id + 1);

            constexpr int min_weight = 100;

            for (auto keyfrm : all_keyfrms)
            {
                if (keyfrm->will_be_erased())
                {
                    continue;
                }
                auto keyfrm_vtx = new g2o::sim3::shot_vertex();

                const auto id = keyfrm->id_;

                // Check if pose is corrected before optimization
                const auto iter = pre_corrected_Sim3s.find(keyfrm);
                if (iter != pre_corrected_Sim3s.end())
                {
                    // If the pose has been corrected before optimization, take out that pose and set it in vertex
                    Sim3s_cw.at(id) = iter->second;
                    keyfrm_vtx->setEstimate(iter->second);
                }
                else
                {
                    // If the pose is not corrected, convert the pose of keyframe to Sim3 and set it
                    const Mat33_t rot_cw = keyfrm->get_rotation();
                    const Vec3_t trans_cw = keyfrm->get_translation();
                    const ::g2o::Sim3 Sim3_cw(rot_cw, trans_cw, 1.0);

                    Sim3s_cw.at(id) = Sim3_cw;
                    keyfrm_vtx->setEstimate(Sim3_cw);
                }

                // Fix the starting point of the loop
                if (*keyfrm == *loop_keyfrm)
                {
                    keyfrm_vtx->setFixed(true);
                }

                // set vertex to optimizer
                keyfrm_vtx->setId(id);
                keyfrm_vtx->fix_scale_ = fix_scale_;

                optimizer.addVertex(keyfrm_vtx);
                vertices.at(id) = keyfrm_vtx;

                // FW: used for debugging, correction of 3D lines
                keyframes[id] = keyfrm;
            }

            // [3] add edge

            // Save between which keyframe the edge was added
            std::set<std::pair<unsigned int, unsigned int>> inserted_edge_pairs;

            // Function to add constraint edge
            const auto insert_edge =
                [&optimizer, &vertices, &inserted_edge_pairs](unsigned int id1, unsigned int id2, const ::g2o::Sim3 &Sim3_21)
            {
                auto edge = new g2o::sim3::graph_opt_edge();
                edge->setVertex(0, vertices.at(id1));
                edge->setVertex(1, vertices.at(id2));
                edge->setMeasurement(Sim3_21);

                edge->information() = MatRC_t<7, 7>::Identity();

                optimizer.addEdge(edge);
                inserted_edge_pairs.insert(std::make_pair(std::min(id1, id2), std::max(id1, id2)));
            };

            // [3.1] Add a loop edge with a threshold weight or more
            // FW: this is the first type of edge
            // these are the "new" edges due to map point match/correction from loop closure
            for (const auto &loop_connection : loop_connections)
            {
                auto keyfrm = loop_connection.first;
                const auto &connected_keyfrms = loop_connection.second;

                const auto id1 = keyfrm->id_;
                const ::g2o::Sim3 &Sim3_1w = Sim3s_cw.at(id1);
                const ::g2o::Sim3 Sim3_w1 = Sim3_1w.inverse();

                for (auto connected_keyfrm : connected_keyfrms)
                {
                    const auto id2 = connected_keyfrm->id_;

                    // For edges other than current vs loop, only those that exceed the weight threshold
                    // Add
                    if ((id1 != curr_keyfrm->id_ || id2 != loop_keyfrm->id_) && keyfrm->graph_node_->get_weight(connected_keyfrm) < min_weight)
                    {
                        continue;
                    }

                    // Calculate relative pose
                    const ::g2o::Sim3 &Sim3_2w = Sim3s_cw.at(id2);
                    const ::g2o::Sim3 Sim3_21 = Sim3_2w * Sim3_w1;

                    // Add constraint edge
                    insert_edge(id1, id2, Sim3_21);
                }
            }

            // [3.2] Add edges other than loop connection
            for (auto keyfrm : all_keyfrms)
            {
                // Take out the pose of one keyframe
                const auto id1 = keyfrm->id_;

                // Check whether it is included in covisibilities and always use the pose before correction
                // (Because both sides need to be before correction in order to calculate the relative pose correctly)
                const auto iter1 = non_corrected_Sim3s.find(keyfrm);
                const ::g2o::Sim3 Sim3_w1 = ((iter1 != non_corrected_Sim3s.end()) ? iter1->second : Sim3s_cw.at(id1)).inverse();

                // FW: this is the second type of edge
                // from spanning tree, parent node, which is the keyframe has the highest covisibility
                auto parent_node = keyfrm->graph_node_->get_spanning_parent();
                if (parent_node)
                {
                    const auto id2 = parent_node->id_;

                    // Prevent duplication
                    if (id1 <= id2)
                    {
                        continue;
                    }

                    // Check whether it is included in covisibilities and always use the pose before correction
                    // (Because both sides need to be before correction in order to calculate the relative attitude correctly)
                    const auto iter2 = non_corrected_Sim3s.find(parent_node);
                    const ::g2o::Sim3 &Sim3_2w = (iter2 != non_corrected_Sim3s.end()) ? iter2->second : Sim3s_cw.at(id2);

                    // Calculate relative pose
                    const ::g2o::Sim3 Sim3_21 = Sim3_2w * Sim3_w1;

                    // Add constraint edge
                    insert_edge(id1, id2, Sim3_21);
                }

                // add loop edge regardless of weight
                // FW: this is the third type of edge
                // the edge between the current keyframe and the closed-loop matching keyframe
                const auto loop_edges = keyfrm->graph_node_->get_loop_edges();
                for (auto connected_keyfrm : loop_edges)
                {
                    const auto id2 = connected_keyfrm->id_;

                    // Prevent duplication
                    if (id1 <= id2)
                    {
                        continue;
                    }

                    // Check whether it is included in covisibilities and always use the pose before correction
                    // (Because both sides need to be before correction in order to calculate the relative attitude correctly)
                    const auto iter2 = non_corrected_Sim3s.find(connected_keyfrm);
                    const ::g2o::Sim3 &Sim3_2w = (iter2 != non_corrected_Sim3s.end()) ? iter2->second : Sim3s_cw.at(id2);

                    // Calculate relative pose
                    const ::g2o::Sim3 Sim3_21 = Sim3_2w * Sim3_w1;

                    // Add constraint edge
                    insert_edge(id1, id2, Sim3_21);
                }

                // Add covisibilities with more than threshold weight
                // FW: fourth type of edge
                // keyframes with a common points more than 100
                const auto connected_keyfrms = keyfrm->graph_node_->get_covisibilities_over_weight(min_weight);
                for (auto connected_keyfrm : connected_keyfrms)
                {
                    // null check
                    if (!connected_keyfrm || !parent_node)
                    {
                        continue;
                    }

                    // Exclude parent-child edge because it has already been added
                    if (*connected_keyfrm == *parent_node || keyfrm->graph_node_->has_spanning_child(connected_keyfrm))
                    {
                        continue;
                    }

                    // If it supports loop, it has already been added, so it is excluded
                    if (static_cast<bool>(loop_edges.count(connected_keyfrm)))
                    {
                        continue;
                    }

                    if (connected_keyfrm->will_be_erased())
                    {
                        continue;
                    }

                    const auto id2 = connected_keyfrm->id_;

                    // Prevent duplication
                    if (id1 <= id2)
                    {
                        continue;
                    }

                    if (static_cast<bool>(inserted_edge_pairs.count(std::make_pair(std::min(id1, id2), std::max(id1, id2)))))
                    {
                        continue;
                    }

                    // Check whether it is included in covisibilities and always use the pose before correction
                    // (Because both sides need to be before correction in order to calculate the relative attitude correctly)
                    const auto iter2 = non_corrected_Sim3s.find(connected_keyfrm);
                    const ::g2o::Sim3 &Sim3_2w = (iter2 != non_corrected_Sim3s.end()) ? iter2->second : Sim3s_cw.at(id2);

                    // Calculate relative pose
                    const ::g2o::Sim3 Sim3_21 = Sim3_2w * Sim3_w1;

                    // Add constraint edge
                    insert_edge(id1, id2, Sim3_21);
                }
            }

            // [4] run pose graph optimization

            optimizer.initializeOptimization();
            optimizer.optimize(50);

            // [5] pose updated

            {
                std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

                // Save the poses of all keyframes (after correction) to correct the position of the point cloud
                std::vector<::g2o::Sim3, Eigen::aligned_allocator<::g2o::Sim3>> corrected_Sim3s_wc(max_keyfrm_id + 1);

                for (auto keyfrm : all_keyfrms)
                {
                    const auto id = keyfrm->id_;

                    auto keyfrm_vtx = static_cast<g2o::sim3::shot_vertex *>(optimizer.vertex(id));

                    const ::g2o::Sim3 &corrected_Sim3_cw = keyfrm_vtx->estimate();
                    const float s = corrected_Sim3_cw.scale();
                    const Mat33_t rot_cw = corrected_Sim3_cw.rotation().toRotationMatrix();
                    const Vec3_t trans_cw = corrected_Sim3_cw.translation() / s;

                    const Mat44_t cam_pose_cw = util::converter::to_eigen_cam_pose(rot_cw, trans_cw);
                    keyfrm->set_cam_pose(cam_pose_cw);

                    corrected_Sim3s_wc.at(id) = corrected_Sim3_cw.inverse();
                }

                // Correct position of point cloud
                for (auto lm : all_lms)
                {
                    if (lm->will_be_erased())
                    {
                        continue;
                    }

                    const auto id = (lm->loop_fusion_identifier_ == curr_keyfrm->id_)
                                        ? lm->ref_keyfrm_id_in_loop_fusion_
                                        : lm->get_ref_keyframe()->id_;

                    const ::g2o::Sim3 &Sim3_cw = Sim3s_cw.at(id);
                    const ::g2o::Sim3 &corrected_Sim3_wc = corrected_Sim3s_wc.at(id);

                    const Vec3_t pos_w = lm->get_pos_in_world();
                    const Vec3_t corrected_pos_w = corrected_Sim3_wc.map(Sim3_cw.map(pos_w));

                    lm->set_pos_in_world(corrected_pos_w);
                    lm->update_normal_and_depth();
                }

                // FW: correct position of line cloud
                if (map_db_->_b_use_line_tracking && !all_lms_line.empty())
                {
                    for (auto lm_line : all_lms_line)
                    {
                        if (lm_line->will_be_erased())
                        {
                            continue;
                        }

                        if (!lm_line->get_ref_keyframe())
                        {
                            lm_line->prepare_for_erasing();
                            continue;
                        }

                        if (!keyframes.count(lm_line->get_ref_keyframe()->id_))
                        {
                            // spdlog::warn("to_json: 3D line's ref_keyframe not found");
                            lm_line->prepare_for_erasing();
                            continue;
                        }

                        const auto id = (lm_line->_loop_fusion_identifier == curr_keyfrm->id_)
                                            ? lm_line->_ref_keyfrm_id_in_loop_fusion
                                            : lm_line->get_ref_keyframe()->id_;

                        const ::g2o::Sim3 &Sim3_cw = Sim3s_cw.at(id);
                        const ::g2o::Sim3 &corrected_Sim3_wc = corrected_Sim3s_wc.at(id);

                        // construct "Sim3_cw" for the pluecker coordinates
                        double scale_cw = Sim3_cw.scale();
                        Mat33_t rot_cw = Sim3_cw.rotation().toRotationMatrix();
                        Vec3_t trans_cw = Sim3_cw.translation();

                        Mat66_t Sim3_cw_pluecker = Eigen::Matrix<double, 6, 6>::Zero();
                        Sim3_cw_pluecker.block<3, 3>(0, 0) = scale_cw * rot_cw;
                        Sim3_cw_pluecker.block<3, 3>(3, 3) = rot_cw;
                        Sim3_cw_pluecker.block<3, 3>(0, 3) = skew(trans_cw) * rot_cw;

                        // construct "corrected_Sim3_wc" for the pluecker coordinates
                        double scale_wc = corrected_Sim3_wc.scale();
                        Mat33_t rot_wc = corrected_Sim3_wc.rotation().toRotationMatrix();
                        Vec3_t trans_wc = corrected_Sim3_wc.translation();

                        Mat66_t corrected_Sim3_wc_pluecker = Eigen::Matrix<double, 6, 6>::Zero();
                        corrected_Sim3_wc_pluecker.block<3, 3>(0, 0) = scale_wc * rot_wc;
                        corrected_Sim3_wc_pluecker.block<3, 3>(3, 3) = rot_wc;
                        corrected_Sim3_wc_pluecker.block<3, 3>(0, 3) = skew(trans_wc) * rot_wc;

                        const Vec6_t pos_w = lm_line->get_PlueckerCoord();
                        Vec6_t corrected_pos_w = corrected_Sim3_wc_pluecker * Sim3_cw_pluecker * pos_w;

                        lm_line->set_PlueckerCoord_without_update_endpoints(corrected_pos_w);

                        // endpoints trimming via its reference keyframe
                        Vec6_t updated_pose_w_endpoints;
                        if (endpoint_trimming(lm_line, corrected_pos_w, updated_pose_w_endpoints))
                        {
                            lm_line->set_pos_in_world_without_update_pluecker(updated_pose_w_endpoints);
                            lm_line->update_information();
                        }
                        else
                        {
                            lm_line->prepare_for_erasing();
                        }
                    }
                }
            }
        }

        // FW:
        bool graph_optimizer::endpoint_trimming(data::Line *local_lm_line,
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
