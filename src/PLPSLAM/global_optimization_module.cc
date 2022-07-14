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

#include "PLPSLAM/mapping_module.h"
#include "PLPSLAM/global_optimization_module.h"
#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/data/map_database.h"
#include "PLPSLAM/match/fuse.h"
#include "PLPSLAM/util/converter.h"

#include <spdlog/spdlog.h>

// FW:
#include "PLPSLAM/data/landmark_line.h"

namespace PLPSLAM
{

    global_optimization_module::global_optimization_module(data::map_database *map_db, data::bow_database *bow_db,
                                                           data::bow_vocabulary *bow_vocab, const bool fix_scale)
        : loop_detector_(new module::loop_detector(bow_db, bow_vocab, fix_scale)),
          loop_bundle_adjuster_(new module::loop_bundle_adjuster(map_db)),
          graph_optimizer_(new optimize::graph_optimizer(map_db, fix_scale))
    {
        spdlog::debug("CONSTRUCT: global_optimization_module");

        // FW:
        _b_use_line_tracking = map_db->_b_use_line_tracking;
    }

    global_optimization_module::~global_optimization_module()
    {
        abort_loop_BA();
        if (thread_for_loop_BA_)
        {
            thread_for_loop_BA_->join();
        }
        spdlog::debug("DESTRUCT: global_optimization_module");
    }

    void global_optimization_module::set_tracking_module(tracking_module *tracker)
    {
        tracker_ = tracker;
    }

    void global_optimization_module::set_mapping_module(mapping_module *mapper)
    {
        mapper_ = mapper;
        loop_bundle_adjuster_->set_mapping_module(mapper);
    }

    void global_optimization_module::enable_loop_detector()
    {
        spdlog::info("enable loop detector");
        loop_detector_->enable_loop_detector();
    }

    void global_optimization_module::disable_loop_detector()
    {
        spdlog::info("disable loop detector");
        loop_detector_->disable_loop_detector();
    }

    bool global_optimization_module::loop_detector_is_enabled() const
    {
        return loop_detector_->is_enabled();
    }

    void global_optimization_module::run()
    {
        spdlog::info("start global optimization module");

        is_terminated_ = false;

        while (true)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));

            // check if termination is requested
            if (terminate_is_requested())
            {
                // terminate and break
                terminate();
                break;
            }

            // check if pause is requested
            if (pause_is_requested())
            {
                // pause and wait
                pause();
                // check if termination or reset is requested during pause
                while (is_paused() && !terminate_is_requested() && !reset_is_requested())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                }
            }

            // check if reset is requested
            if (reset_is_requested())
            {
                // reset and continue
                reset();
                continue;
            }

            // if the queue is empty, the following process is not needed
            if (!keyframe_is_queued())
            {
                continue;
            }

            // dequeue the keyframe from the queue -> cur_keyfrm_
            {
                std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
                cur_keyfrm_ = keyfrms_queue_.front();
                keyfrms_queue_.pop_front();
            }

            // not to be removed during loop detection and correction
            cur_keyfrm_->set_not_to_be_erased();

            // pass the current keyframe to the loop detector
            loop_detector_->set_current_keyframe(cur_keyfrm_);

            // detect some loop candidate with BoW
            // FW: [1] detect loop candidates -> get candidate keyframes
            if (!loop_detector_->detect_loop_candidates())
            {
                // could not find
                // allow the removal of the current keyframe
                cur_keyfrm_->set_to_be_erased();
                continue;
            }

            // validate candidates and select ONE candidate from them
            // FW: [2] LoopClosing::ComputeSim3() in the ORB-SLAM2, find the ONE keyframe for closing the loop, calculate the Sim3, and get 3D-2D matches
            if (!loop_detector_->validate_candidates())
            {
                // could not find
                // allow the removal of the current keyframe
                cur_keyfrm_->set_to_be_erased();
                continue;
            }

            // FW: [3] Correct map and camera poses using the Sim3 calculated from step [2],
            // + Essential graph optimization (correct map and camera poses again),
            // + global BA (correct map and camera poses again).
            correct_loop();
        }

        spdlog::info("terminate global optimization module");
    }

    void global_optimization_module::queue_keyframe(data::keyframe *keyfrm)
    {
        std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
        if (keyfrm->id_ != 0)
        {
            keyfrms_queue_.push_back(keyfrm);
        }
    }

    bool global_optimization_module::keyframe_is_queued() const
    {
        std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
        return (!keyfrms_queue_.empty());
    }

    void global_optimization_module::correct_loop()
    {
        auto final_candidate_keyfrm = loop_detector_->get_selected_candidate_keyframe();

        spdlog::info("detect loop: keyframe {} - keyframe {}", final_candidate_keyfrm->id_, cur_keyfrm_->id_);
        loop_bundle_adjuster_->count_loop_BA_execution();

        // [1] pre-processing

        // stop the mapping module and the previous loop bundle adjuster
        // pause the mapping module
        mapper_->request_pause();

        // abort the previous loop bundle adjuster
        if (thread_for_loop_BA_ || loop_bundle_adjuster_->is_running())
        {
            abort_loop_BA();
        }

        // wait till the mapping module pauses
        while (!mapper_->is_paused())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }

        // update the graph, because we got new matched landmarks by loop detection -> loop_detector::curr_match_lms_observed_in_cand_
        cur_keyfrm_->graph_node_->update_connections();

        // [2] compute the Sim3 of the covisibilities of the current keyframe whose Sim3 is already estimated by the loop detector
        //    then, the covisibilities are moved to the corrected positions
        //    finally, landmarks observed in them are also moved to the correct position using the camera poses before and after camera pose correction

        // acquire the covisibilities of the current keyframe
        std::vector<data::keyframe *> curr_neighbors = cur_keyfrm_->graph_node_->get_covisibilities();
        curr_neighbors.push_back(cur_keyfrm_);

        // Sim3 camera poses BEFORE loop correction
        module::keyframe_Sim3_pairs_t Sim3s_nw_before_correction;
        // Sim3 camera poses AFTER loop correction
        module::keyframe_Sim3_pairs_t Sim3s_nw_after_correction;

        const auto g2o_Sim3_cw_after_correction = loop_detector_->get_Sim3_world_to_current();
        {
            std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

            // camera pose of the current keyframe BEFORE loop correction
            const Mat44_t cam_pose_wc_before_correction = cur_keyfrm_->get_cam_pose_inv();

            // compute Sim3s BEFORE loop correction
            Sim3s_nw_before_correction = get_Sim3s_before_loop_correction(curr_neighbors);
            // compute Sim3s AFTER loop correction
            Sim3s_nw_after_correction = get_Sim3s_after_loop_correction(cam_pose_wc_before_correction, g2o_Sim3_cw_after_correction, curr_neighbors);

            // correct covisibility landmark positions
            correct_covisibility_landmarks(Sim3s_nw_before_correction, Sim3s_nw_after_correction);
            // correct covisibility keyframe camera poses
            correct_covisibility_keyframes(Sim3s_nw_after_correction);

            // FW:
            // also correct covisibility 3D lines' position, after correction of keyframes
            if (_b_use_line_tracking)
            {
                correct_covisibility_landmarks_line(Sim3s_nw_before_correction, Sim3s_nw_after_correction);
            }
        }

        // [3] resolve duplications of landmarks caused by loop fusion

        const auto curr_match_lms_observed_in_cand = loop_detector_->current_matched_landmarks_observed_in_candidate();
        replace_duplicated_landmarks(curr_match_lms_observed_in_cand, Sim3s_nw_after_correction);

        // [4] extract the new connections created after loop fusion

        const auto new_connections = extract_new_connections(curr_neighbors);

        // [5] pose graph optimization - > Optimizer::OptimizeEssentialGraph() in ORB-SLAM2

        // FW:
        // 3D line cloud will be updated after graph optimization
        graph_optimizer_->optimize(final_candidate_keyfrm, cur_keyfrm_, Sim3s_nw_before_correction, Sim3s_nw_after_correction, new_connections);

        // add a loop edge
        final_candidate_keyfrm->graph_node_->add_loop_edge(cur_keyfrm_);
        cur_keyfrm_->graph_node_->add_loop_edge(final_candidate_keyfrm);

        // [6] launch loop BA

        while (loop_bundle_adjuster_->is_running())
        {
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
        if (thread_for_loop_BA_)
        {
            thread_for_loop_BA_->join();
            thread_for_loop_BA_.reset(nullptr);
        }
        thread_for_loop_BA_ = std::unique_ptr<std::thread>(new std::thread(&module::loop_bundle_adjuster::optimize, loop_bundle_adjuster_.get(), cur_keyfrm_->id_));

        // [7] post-processing

        // resume the mapping module
        mapper_->resume();

        // set the loop fusion information to the loop detector
        loop_detector_->set_loop_correct_keyframe_id(cur_keyfrm_->id_);
    }

    module::keyframe_Sim3_pairs_t global_optimization_module::get_Sim3s_before_loop_correction(const std::vector<data::keyframe *> &neighbors) const
    {
        module::keyframe_Sim3_pairs_t Sim3s_nw_before_loop_correction;

        for (const auto neighbor : neighbors)
        {
            // camera pose of `neighbor` BEFORE loop correction
            const Mat44_t cam_pose_nw = neighbor->get_cam_pose();
            // create Sim3 from SE3
            const Mat33_t &rot_nw = cam_pose_nw.block<3, 3>(0, 0);
            const Vec3_t &trans_nw = cam_pose_nw.block<3, 1>(0, 3);
            const g2o::Sim3 Sim3_nw_before_correction(rot_nw, trans_nw, 1.0);
            Sim3s_nw_before_loop_correction[neighbor] = Sim3_nw_before_correction;
        }

        return Sim3s_nw_before_loop_correction;
    }

    module::keyframe_Sim3_pairs_t global_optimization_module::get_Sim3s_after_loop_correction(const Mat44_t &cam_pose_wc_before_correction,
                                                                                              const g2o::Sim3 &g2o_Sim3_cw_after_correction,
                                                                                              const std::vector<data::keyframe *> &neighbors) const
    {
        module::keyframe_Sim3_pairs_t Sim3s_nw_after_loop_correction;

        for (auto neighbor : neighbors)
        {
            // camera pose of `neighbor` BEFORE loop correction
            const Mat44_t cam_pose_nw_before_correction = neighbor->get_cam_pose();
            // create the relative Sim3 from the current to `neighbor`
            const Mat44_t cam_pose_nc = cam_pose_nw_before_correction * cam_pose_wc_before_correction;
            const Mat33_t &rot_nc = cam_pose_nc.block<3, 3>(0, 0);
            const Vec3_t &trans_nc = cam_pose_nc.block<3, 1>(0, 3);
            const g2o::Sim3 Sim3_nc(rot_nc, trans_nc, 1.0);
            // compute the camera poses AFTER loop correction of the neighbors
            const g2o::Sim3 Sim3_nw_after_correction = Sim3_nc * g2o_Sim3_cw_after_correction;
            Sim3s_nw_after_loop_correction[neighbor] = Sim3_nw_after_correction;
        }

        return Sim3s_nw_after_loop_correction;
    }

    void global_optimization_module::correct_covisibility_landmarks(const module::keyframe_Sim3_pairs_t &Sim3s_nw_before_correction,
                                                                    const module::keyframe_Sim3_pairs_t &Sim3s_nw_after_correction) const
    {
        for (const auto &keyframe_sim3_pair : Sim3s_nw_after_correction)
        {
            auto neighbor = keyframe_sim3_pair.first;
            // neighbor->world AFTER loop correction
            const auto Sim3_wn_after_correction = keyframe_sim3_pair.second.inverse();
            // world->neighbor BEFORE loop correction
            const auto &Sim3_nw_before_correction = Sim3s_nw_before_correction.at(neighbor);

            const auto ngh_landmarks = neighbor->get_landmarks();
            for (auto lm : ngh_landmarks)
            {
                if (!lm)
                {
                    continue;
                }
                if (lm->will_be_erased())
                {
                    continue;
                }

                // avoid duplication
                if (lm->loop_fusion_identifier_ == cur_keyfrm_->id_)
                {
                    continue;
                }
                lm->loop_fusion_identifier_ = cur_keyfrm_->id_;

                // correct position of `lm`
                const Vec3_t pos_w_before_correction = lm->get_pos_in_world();
                const Vec3_t pos_w_after_correction = Sim3_wn_after_correction.map(Sim3_nw_before_correction.map(pos_w_before_correction));
                lm->set_pos_in_world(pos_w_after_correction);
                // update geometry
                lm->update_normal_and_depth();

                // record the reference keyframe used in loop fusion of landmarks
                lm->ref_keyfrm_id_in_loop_fusion_ = neighbor->id_;
            }
        }
    }

    // FW:
    void global_optimization_module::correct_covisibility_landmarks_line(const module::keyframe_Sim3_pairs_t &Sim3s_nw_before_correction,
                                                                         const module::keyframe_Sim3_pairs_t &Sim3s_nw_after_correction) const
    {
        for (const auto &keyframe_sim3_pair : Sim3s_nw_after_correction)
        {
            auto neighbor = keyframe_sim3_pair.first;
            // neighbor->world AFTER loop correction
            const auto Sim3_wn_after_correction = keyframe_sim3_pair.second.inverse();
            // world->neighbor BEFORE loop correction
            const auto &Sim3_nw_before_correction = Sim3s_nw_before_correction.at(neighbor);

            const auto ngh_landmarks_line = neighbor->get_landmarks_line();
            for (auto lm_line : ngh_landmarks_line)
            {
                if (!lm_line)
                {
                    continue;
                }
                if (lm_line->will_be_erased())
                {
                    continue;
                }

                // avoid duplication
                if (lm_line->_loop_fusion_identifier == cur_keyfrm_->id_)
                {
                    continue;
                }
                lm_line->_loop_fusion_identifier = cur_keyfrm_->id_;

                // correct position of `lm_line`
                const Vec6_t pos_w_before_correction = lm_line->get_PlueckerCoord();

                // construct "Sim3_nw_before_correction" for line's pluecker coordinates
                double scale_nw_before = Sim3_nw_before_correction.scale();
                Mat33_t rot_nw_before = Sim3_nw_before_correction.rotation().toRotationMatrix();
                Vec3_t trans_nw_before = Sim3_nw_before_correction.translation();

                Mat66_t Sim3_nw_before_correction_pluecker = Eigen::Matrix<double, 6, 6>::Zero();
                Sim3_nw_before_correction_pluecker.block<3, 3>(0, 0) = scale_nw_before * rot_nw_before;
                Sim3_nw_before_correction_pluecker.block<3, 3>(3, 3) = rot_nw_before;
                Sim3_nw_before_correction_pluecker.block<3, 3>(0, 3) = skew(trans_nw_before) * rot_nw_before;

                // construct "Sim3_wn_after_correction" for line's pluecker coordinates
                double scale_wn_after = Sim3_wn_after_correction.scale();
                Mat33_t rot_wn_after = Sim3_wn_after_correction.rotation().toRotationMatrix();
                Vec3_t trans_wn_after = Sim3_wn_after_correction.translation();

                Mat66_t Sim3_wn_after_correction_pluecker = Eigen::Matrix<double, 6, 6>::Zero();
                Sim3_wn_after_correction_pluecker.block<3, 3>(0, 0) = scale_wn_after * rot_wn_after;
                Sim3_wn_after_correction_pluecker.block<3, 3>(3, 3) = rot_wn_after;
                Sim3_wn_after_correction_pluecker.block<3, 3>(0, 3) = skew(trans_wn_after) * rot_wn_after;

                // calculate the pluecker coordinates after Sim3 correction
                Vec6_t pos_w_after_correction = Sim3_wn_after_correction_pluecker * Sim3_nw_before_correction_pluecker * pos_w_before_correction;
                lm_line->set_PlueckerCoord_without_update_endpoints(pos_w_after_correction);

                // endpoints trimming via its reference keyframe
                Vec6_t updated_pose_w_endpoints;
                if (endpoint_trimming(lm_line, pos_w_after_correction, updated_pose_w_endpoints))
                {
                    lm_line->set_pos_in_world_without_update_pluecker(updated_pose_w_endpoints);
                    // update geometry
                    lm_line->update_information();

                    // record the reference keyframe used in loop fusion of landmarks
                    lm_line->_ref_keyfrm_id_in_loop_fusion = neighbor->id_;
                }
                else
                {
                    lm_line->prepare_for_erasing();
                }
            }
        }
    }

    // FW:
    bool global_optimization_module::endpoint_trimming(data::Line *local_lm_line,
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

    void global_optimization_module::correct_covisibility_keyframes(const module::keyframe_Sim3_pairs_t &Sim3s_nw_after_correction) const
    {
        for (const auto &keyframe_sim3_pair : Sim3s_nw_after_correction)
        {
            auto neighbor = keyframe_sim3_pair.first;
            const auto Sim3_nw_after_correction = keyframe_sim3_pair.second;

            const auto s_nw = Sim3_nw_after_correction.scale();
            const Mat33_t rot_nw = Sim3_nw_after_correction.rotation().toRotationMatrix();
            const Vec3_t trans_nw = Sim3_nw_after_correction.translation() / s_nw;
            const Mat44_t cam_pose_nw = util::converter::to_eigen_cam_pose(rot_nw, trans_nw);
            neighbor->set_cam_pose(cam_pose_nw);

            // update graph
            neighbor->graph_node_->update_connections();
        }
    }

    void global_optimization_module::replace_duplicated_landmarks(const std::vector<data::landmark *> &curr_match_lms_observed_in_cand,
                                                                  const module::keyframe_Sim3_pairs_t &Sim3s_nw_after_correction) const
    {
        // resolve duplications of landmarks between the current keyframe and the loop candidate
        {
            std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

            for (unsigned int idx = 0; idx < cur_keyfrm_->num_keypts_; ++idx)
            {
                auto curr_match_lm_in_cand = curr_match_lms_observed_in_cand.at(idx);
                if (!curr_match_lm_in_cand)
                {
                    continue;
                }

                auto lm_in_curr = cur_keyfrm_->get_landmark(idx);
                if (lm_in_curr)
                {
                    // if the landmark corresponding `idx` exists,
                    // replace it with `curr_match_lm_in_cand` (observed in the candidate)
                    lm_in_curr->replace(curr_match_lm_in_cand);
                }
                else
                {
                    // if landmark corresponding `idx` does not exists,
                    // add association between the current keyframe and `curr_match_lm_in_cand`
                    cur_keyfrm_->add_landmark(curr_match_lm_in_cand, idx);
                    curr_match_lm_in_cand->add_observation(cur_keyfrm_, idx);
                    curr_match_lm_in_cand->compute_descriptor();
                }
            }
        }

        // resolve duplications of landmarks between the current keyframe and the candidates of the loop candidate
        const auto curr_match_lms_observed_in_cand_covis = loop_detector_->current_matched_landmarks_observed_in_candidate_covisibilities();
        match::fuse fuser(0.8);
        for (const auto &t : Sim3s_nw_after_correction)
        {
            auto neighbor = t.first;
            const Mat44_t Sim3_nw_after_correction = util::converter::to_eigen_mat(t.second);

            // reproject the landmarks observed in the current keyframe to the neighbor,
            // then search duplication of the landmarks
            std::vector<data::landmark *> lms_to_replace(curr_match_lms_observed_in_cand_covis.size(), nullptr);
            fuser.detect_duplication(neighbor, Sim3_nw_after_correction, curr_match_lms_observed_in_cand_covis, 4, lms_to_replace);

            std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);
            // if any landmark duplication is found, replace it
            for (unsigned int i = 0; i < curr_match_lms_observed_in_cand_covis.size(); ++i)
            {
                auto lm_to_replace = lms_to_replace.at(i);
                if (lm_to_replace)
                {
                    lm_to_replace->replace(curr_match_lms_observed_in_cand_covis.at(i));
                }
            }
        }
    }

    auto global_optimization_module::extract_new_connections(const std::vector<data::keyframe *> &covisibilities) const
        -> std::map<data::keyframe *, std::set<data::keyframe *>>
    {
        std::map<data::keyframe *, std::set<data::keyframe *>> new_connections;

        for (auto covisibility : covisibilities)
        {
            // acquire neighbors BEFORE loop fusion (because update_connections() is not called yet)
            const auto neighbors_before_update = covisibility->graph_node_->get_covisibilities();

            // call update_connections()
            covisibility->graph_node_->update_connections();
            // acquire neighbors AFTER loop fusion
            new_connections[covisibility] = covisibility->graph_node_->get_connected_keyframes();

            // remove covisibilities
            for (const auto keyfrm_to_erase : covisibilities)
            {
                new_connections.at(covisibility).erase(keyfrm_to_erase);
            }
            // remove nighbors before loop fusion
            for (const auto keyfrm_to_erase : neighbors_before_update)
            {
                new_connections.at(covisibility).erase(keyfrm_to_erase);
            }
        }

        return new_connections;
    }

    void global_optimization_module::request_reset()
    {
        {
            std::lock_guard<std::mutex> lock(mtx_reset_);
            reset_is_requested_ = true;
        }

        // BLOCK until reset
        while (true)
        {
            {
                std::lock_guard<std::mutex> lock(mtx_reset_);
                if (!reset_is_requested_)
                {
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(3000));
        }
    }

    bool global_optimization_module::reset_is_requested() const
    {
        std::lock_guard<std::mutex> lock(mtx_reset_);
        return reset_is_requested_;
    }

    void global_optimization_module::reset()
    {
        std::lock_guard<std::mutex> lock(mtx_reset_);
        spdlog::info("reset global optimization module");
        keyfrms_queue_.clear();
        loop_detector_->set_loop_correct_keyframe_id(0);
        reset_is_requested_ = false;
    }

    void global_optimization_module::request_pause()
    {
        std::lock_guard<std::mutex> lock1(mtx_pause_);
        pause_is_requested_ = true;
    }

    bool global_optimization_module::pause_is_requested() const
    {
        std::lock_guard<std::mutex> lock(mtx_pause_);
        return pause_is_requested_;
    }

    bool global_optimization_module::is_paused() const
    {
        std::lock_guard<std::mutex> lock(mtx_pause_);
        return is_paused_;
    }

    void global_optimization_module::pause()
    {
        std::lock_guard<std::mutex> lock(mtx_pause_);
        spdlog::info("pause global optimization module");
        is_paused_ = true;
    }

    void global_optimization_module::resume()
    {
        std::lock_guard<std::mutex> lock1(mtx_pause_);
        std::lock_guard<std::mutex> lock2(mtx_terminate_);

        // if it has been already terminated, cannot resume
        if (is_terminated_)
        {
            return;
        }

        is_paused_ = false;
        pause_is_requested_ = false;

        spdlog::info("resume global optimization module");
    }

    void global_optimization_module::request_terminate()
    {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        terminate_is_requested_ = true;
    }

    bool global_optimization_module::is_terminated() const
    {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        return is_terminated_;
    }

    bool global_optimization_module::terminate_is_requested() const
    {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        return terminate_is_requested_;
    }

    void global_optimization_module::terminate()
    {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        is_terminated_ = true;
    }

    bool global_optimization_module::loop_BA_is_running() const
    {
        return loop_bundle_adjuster_->is_running();
    }

    void global_optimization_module::abort_loop_BA()
    {
        loop_bundle_adjuster_->abort();
    }

} // namespace PLPSLAM
