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

#include "PLPSLAM/type.h"
#include "PLPSLAM/mapping_module.h"
#include "PLPSLAM/global_optimization_module.h"
#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/data/map_database.h"
#include "PLPSLAM/match/fuse.h"
#include "PLPSLAM/match/robust.h"
#include "PLPSLAM/module/two_view_triangulator.h"
#include "PLPSLAM/module/two_view_triangulator_line.h"
#include "PLPSLAM/solve/essential_solver.h"

#include "PLPSLAM/camera/perspective.h"
#include <Eigen/SVD>

#include <unordered_set>
#include <thread>

#include <spdlog/spdlog.h>

// FW:
#include "PLPSLAM/planar_mapping_module.h"

namespace PLPSLAM
{

    mapping_module::mapping_module(data::map_database *map_db, const bool is_monocular)
        : local_map_cleaner_(new module::local_map_cleaner(is_monocular)),
          map_db_(map_db),
          local_bundle_adjuster_(new optimize::local_bundle_adjuster(map_db_)),
          _local_bundle_adjuster_extended_plane(new optimize::local_bundle_adjuster_extended_plane(map_db_)), // FW: TODO: not initialize if no planar mapping
          _local_bundle_adjuster_extended_line(new optimize::local_bundle_adjuster_extended_line(map_db_)),   // FW: TODO: not initialize if no line tracking
          is_monocular_(is_monocular)
    {
        spdlog::debug("CONSTRUCT: mapping_module");

        // FW: pass the boolean variable to map_cleaner
        if (map_db_->_b_use_line_tracking)
        {
            local_map_cleaner_->_b_use_line_tracking = true;
        }
    }

    mapping_module::~mapping_module()
    {
        spdlog::debug("DESTRUCT: mapping_module");
    }

    void mapping_module::set_tracking_module(tracking_module *tracker)
    {
        tracker_ = tracker;
    }

    void mapping_module::set_global_optimization_module(global_optimization_module *global_optimizer)
    {
        global_optimizer_ = global_optimizer;
    }

    void mapping_module::set_planar_mapping_module(Planar_Mapping_module *planar_mapper)
    {
        _planar_mapper = planar_mapper;
    }

    void mapping_module::run()
    {
        spdlog::info("start mapping module");

        is_terminated_ = false;

        while (true)
        {
            // waiting time for the other threads
            std::this_thread::sleep_for(std::chrono::milliseconds(5));

            // LOCK
            set_keyframe_acceptability(false);

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
                // if any keyframe is queued, all of them must be processed before the pause
                while (keyframe_is_queued())
                {
                    // create and extend the map with the new keyframe
                    mapping_with_new_keyframe();
                    // send the new keyframe to the global optimization module
                    global_optimizer_->queue_keyframe(cur_keyfrm_);
                }
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
                // reset, UNLOCK and continue
                reset();
                set_keyframe_acceptability(true);
                continue;
            }

            // if the queue is empty, the following process is not needed
            if (!keyframe_is_queued())
            {
                // UNLOCK and continue
                set_keyframe_acceptability(true);
                continue;
            }

            // create and extend the map with the new keyframe
            mapping_with_new_keyframe();

            // send the new keyframe to the global optimization module
            global_optimizer_->queue_keyframe(cur_keyfrm_);

            // LOCK end
            set_keyframe_acceptability(true);
        }

        spdlog::info("terminate mapping module");
    }

    void mapping_module::queue_keyframe(data::keyframe *keyfrm)
    {
        std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
        keyfrms_queue_.push_back(keyfrm);
        abort_local_BA_ = true;
    }

    unsigned int mapping_module::get_num_queued_keyframes() const
    {
        std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
        return keyfrms_queue_.size();
    }

    bool mapping_module::keyframe_is_queued() const
    {
        std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
        return !keyfrms_queue_.empty();
    }

    bool mapping_module::get_keyframe_acceptability() const
    {
        return keyfrm_acceptability_;
    }

    void mapping_module::set_keyframe_acceptability(const bool acceptability)
    {
        keyfrm_acceptability_ = acceptability;
    }

    void mapping_module::abort_local_BA()
    {
        abort_local_BA_ = true;
    }

    void mapping_module::mapping_with_new_keyframe()
    {
        // dequeue
        {
            std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
            // dequeue -> cur_keyfrm_
            cur_keyfrm_ = keyfrms_queue_.front();
            keyfrms_queue_.pop_front();
        }

        // set the origin keyframe
        local_map_cleaner_->set_origin_keyframe_id(map_db_->origin_keyfrm_->id_);

        // store the new keyframe to the database
        // FW: -> compute BOW of cur_keyfrm
        //     -> update graph (check if associate landmark is linked to this keyframe)
        //     -> for the lm which is observed in the keyframe but not linked, add observation and update normal&depth and calculate descriptor
        //     -> add this keyframe into the map_db
        // FW: same procedure for 3D line landmark
        store_new_keyframe();

        // remove redundant landmarks (MapPoint Culling)
        local_map_cleaner_->remove_redundant_landmarks(cur_keyfrm_->id_);

        // FW: MapLines Culling
        if (map_db_->_b_use_line_tracking)
        {
            local_map_cleaner_->remove_redundant_landmarks_line(cur_keyfrm_->id_);
        }

        // triangulate new landmarks between the current keyframe and each of the covisibilities
        // FW: similarly, try to match 2D line segments cross keyframes and triangulate 3D Line
        create_new_landmarks();

        if (keyframe_is_queued())
        {
            return;
        }

        // detect and resolve the duplication of the landmarks observed in the current frame
        // FW: similar procedure for 3D lines
        update_new_keyframe();

        if (keyframe_is_queued() || pause_is_requested())
        {
            return;
        }

        // FW: detect new plane
        if (map_db_->_b_seg_or_not)
        {
            if (_planar_mapper->process_new_kf(cur_keyfrm_))
            {
                _planar_mapper->refinement();
            }
        }

        // local BA after a new keyframe is added
        abort_local_BA_ = false;
        if (2 < map_db_->get_num_keyframes())
        {
            if (map_db_->_b_use_line_tracking)
            { // FW: local BA using lines
                _local_bundle_adjuster_extended_line->optimize(cur_keyfrm_, &abort_local_BA_);
            }
            else
            {
                // (default)
                local_bundle_adjuster_->optimize(cur_keyfrm_, &abort_local_BA_);
            }
        }

        // FW: TODO:
        // extended local BA with point and plane
        // if (map_db_->_b_seg_or_not)
        // {
        //     if (2 < map_db_->get_num_keyframes())
        //     {
        //         if (1 <= map_db_->get_num_landmark_planes())
        //         {
        //             _local_bundle_adjuster_extended_plane->optimize(cur_keyfrm_, &abort_local_BA_);
        //         }
        //         else
        //         {
        //             // (default)
        //             local_bundle_adjuster_->optimize(cur_keyfrm_, &abort_local_BA_);
        //         }
        //     }
        // }

        // remove redundant keyframes after local BA
        local_map_cleaner_->remove_redundant_keyframes(cur_keyfrm_);
    }

    void mapping_module::store_new_keyframe()
    {
        // compute BoW feature vector
        cur_keyfrm_->compute_bow();

        // update graph (check 3D points)
        const auto cur_lms = cur_keyfrm_->get_landmarks();
        for (unsigned int idx = 0; idx < cur_lms.size(); ++idx)
        {
            auto lm = cur_lms.at(idx);
            if (!lm)
            {
                continue;
            }
            if (lm->will_be_erased())
            {
                continue;
            }

            // if `lm` does not have the observation information from `cur_keyfrm_`,
            // add the association between the keyframe and the landmark
            if (lm->is_observed_in_keyframe(cur_keyfrm_))
            {
                // if `lm` is correctly observed, make it be checked by the local map cleaner
                local_map_cleaner_->add_fresh_landmark(lm);
                continue;
            }

            // update connection
            lm->add_observation(cur_keyfrm_, idx);
            // update geometry
            lm->update_normal_and_depth();
            lm->compute_descriptor();
        }

        // FW: update graph (check 3D lines)
        if (map_db_->_b_use_line_tracking)
        {
            const auto cur_lms_line = cur_keyfrm_->get_landmarks_line();
            for (unsigned int idx_l = 0; idx_l < cur_lms_line.size(); ++idx_l)
            {
                auto lm_line = cur_lms_line.at(idx_l);
                if (!lm_line)
                {
                    continue;
                }
                if (lm_line->will_be_erased())
                {
                    continue;
                }

                if (lm_line->is_observed_in_keyframe(cur_keyfrm_))
                {
                    // this 3D line is observed, add it to local map cleaner
                    local_map_cleaner_->add_fresh_landmark_line(lm_line);
                    continue;
                }

                // if this 3D line is somehow not linked to the curr_keyframe, add observation
                lm_line->add_observation(cur_keyfrm_, idx_l);
                lm_line->update_information();
                lm_line->compute_descriptor();
            }
        }

        // FW: TODO: connections updated according to 3D points, for now without 3D line
        cur_keyfrm_->graph_node_->update_connections();

        // store the new keyframe to the map database
        map_db_->add_keyframe(cur_keyfrm_);
    }

    void mapping_module::create_new_landmarks()
    {
        // get the covisibilities of `cur_keyfrm_`
        // in order to triangulate landmarks between `cur_keyfrm_` and each of the covisibilities
        constexpr unsigned int num_covisibilities = 10;
        const auto cur_covisibilities = cur_keyfrm_->graph_node_->get_top_n_covisibilities(num_covisibilities * (is_monocular_ ? 2 : 1)); // return is std::vector<keyframe *>

        // camera center of the current keyframe
        const Vec3_t cur_cam_center = cur_keyfrm_->get_cam_center();

        for (unsigned int i = 0; i < cur_covisibilities.size(); ++i)
        {
            // if any keyframe is queued, abort the triangulation
            if (1 < i && keyframe_is_queued())
            {
                return;
            }

            // get the neighbor keyframe
            auto ngh_keyfrm = cur_covisibilities.at(i);

            // camera center of the neighbor keyframe
            const Vec3_t ngh_cam_center = ngh_keyfrm->get_cam_center();

            // compute the baseline between the current and neighbor keyframes
            const Vec3_t baseline_vec = ngh_cam_center - cur_cam_center;
            const auto baseline_dist = baseline_vec.norm();
            if (is_monocular_)
            {
                // if the scene scale is much smaller than the baseline, abort the triangulation
                const float median_depth_in_ngh = ngh_keyfrm->compute_median_depth(true);
                if (baseline_dist < 0.02 * median_depth_in_ngh)
                {
                    continue;
                }
            }
            else
            {
                // for stereo setups, it needs longer baseline than the stereo baseline
                if (baseline_dist < ngh_keyfrm->camera_->true_baseline_)
                {
                    continue;
                }
            }

            // FW: matching and triangulation in parallel
            if (map_db_->_b_use_line_tracking)
            {
                std::thread threadPointTriangulation(&mapping_module::triangulate_with_two_keyframes, this, cur_keyfrm_, ngh_keyfrm);
                std::thread threadLineTriangulation(&mapping_module::triangulate_line_with_two_keyframes, this, cur_keyfrm_, ngh_keyfrm);
                threadPointTriangulation.join();
                threadLineTriangulation.join();
            }
            else
            {
                // (default)
                triangulate_with_two_keyframes(cur_keyfrm_, ngh_keyfrm);
            }
        }
    }

    void mapping_module::triangulate_with_two_keyframes(data::keyframe *cur_keyfrm, data::keyframe *ngh_keyfrm)
    {
        // lowe's_ratio will not be used
        match::robust robust_matcher(0.0, false);
        // estimate matches between the current and neighbor keyframes,
        // then reject outliers using Essential matrix computed from the two camera poses

        // (cur bearing) * E_ngh_to_cur * (ngh bearing) = 0
        // const Mat33_t E_ngh_to_cur = solve::essential_solver::create_E_21(ngh_keyfrm, cur_keyfrm_);
        const Mat33_t E_ngh_to_cur = solve::essential_solver::create_E_21(ngh_keyfrm->get_rotation(), ngh_keyfrm->get_translation(),
                                                                          cur_keyfrm->get_rotation(), cur_keyfrm->get_translation());

        // vector of matches (idx in the current, idx in the neighbor)
        std::vector<std::pair<unsigned int, unsigned int>> matches;
        robust_matcher.match_for_triangulation(cur_keyfrm, ngh_keyfrm, E_ngh_to_cur, matches);

        // FW: initialize the two view triangulator -> pass the keyframe pointer to that object, with poses information
        const module::two_view_triangulator triangulator(cur_keyfrm, ngh_keyfrm, 1.0);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (unsigned int i = 0; i < matches.size(); ++i)
        {
            const auto idx_1 = matches.at(i).first;
            const auto idx_2 = matches.at(i).second;

            // triangulate between idx_1 and idx_2
            Vec3_t pos_w;

            // FW: triangulation + check positive depth + check reprojection error + check scale consistency
            if (!triangulator.triangulate(idx_1, idx_2, pos_w))
            {
                // failed
                continue;
            }
            // succeeded

            // create a landmark object
            auto lm = new data::landmark(pos_w, cur_keyfrm, map_db_);

            lm->add_observation(cur_keyfrm, idx_1);
            lm->add_observation(ngh_keyfrm, idx_2);

            cur_keyfrm->add_landmark(lm, idx_1);
            ngh_keyfrm->add_landmark(lm, idx_2);

            lm->compute_descriptor();
            lm->update_normal_and_depth();

            map_db_->add_landmark(lm);
            // wait for redundancy check
#ifdef USE_OPENMP
#pragma omp critical
#endif
            {
                local_map_cleaner_->add_fresh_landmark(lm);
            }
        }
    }

    // FW:
    void mapping_module::triangulate_line_with_two_keyframes(data::keyframe *cur_keyfrm, data::keyframe *ngh_keyfrm)
    {
        if (cur_keyfrm->_keylsd.size() == 0 || ngh_keyfrm->_keylsd.size() == 0)
        {
            return;
        }

        // FW: TODO: maybe find a more robust way to match lines?
        // match line features
        std::vector<cv::DMatch> lsd_matches;
        cv::Ptr<cv::line_descriptor::BinaryDescriptorMatcher> binary_descriptor_matcher;
        binary_descriptor_matcher = cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

        // cur_keyfrm (query), ngh_keyfrm (train)
        binary_descriptor_matcher->match(cur_keyfrm->_lbd_descr, ngh_keyfrm->_lbd_descr, lsd_matches);

        // select best matches
        std::vector<cv::DMatch> good_matches;
        std::vector<cv::line_descriptor::KeyLine> good_Keylines;
        good_matches.clear();
        for (unsigned j = 0; j < lsd_matches.size(); j++)
        {
            // spdlog::info("lsd_matches.distance: {}", lsd_matches[j].distance);

            if (lsd_matches[j].distance < 50) // original: 30
            {
                cv::DMatch mt = lsd_matches[j];
                cv::line_descriptor::KeyLine line1 = cur_keyfrm->_keylsd[mt.queryIdx];
                cv::line_descriptor::KeyLine line2 = ngh_keyfrm->_keylsd[mt.trainIdx];

                // check the distance
                cv::Point2f serr = line1.getStartPoint() - line2.getStartPoint();
                cv::Point2f eerr = line1.getEndPoint() - line2.getEndPoint();
                const float distance_s = sqrt(serr.dot(serr));
                const float distance_e = sqrt(eerr.dot(eerr));

                // check the angle
                const float angle = abs((abs(line1.angle) - abs(line2.angle))) * 180 / 3.14;

                // spdlog::info("distance between sp: {}", distance_s);
                // spdlog::info("distance between ep: {}", distance_e);
                // spdlog::info("angle_1: {}", line1.angle * 180 / 3.14);
                // spdlog::info("angle_2: {}", line2.angle * 180 / 3.14);
                // spdlog::info("angle: {}", angle);
                // std::cout << std::endl;

                // select good matches, we give bigger thresholds as we match across keyframes
                if (distance_s < 400 && distance_e < 400 && angle < 20) // original: distance < 200, angle < 5 deg
                {
                    good_matches.push_back(lsd_matches[j]);
                }
            }
        }

        // // save / visualize best matches for debugging
        // if (good_matches.size() > 10)
        // {
        //     visualize_line_match(cur_keyfrm_->get_img_rgb(),
        //                          ngh_keyfrm->get_img_rgb(),
        //                          cur_keyfrm_->_keylsd,
        //                          ngh_keyfrm->_keylsd,
        //                          good_matches);

        //     std::cout << "good_matches = " << good_matches.size()
        //               << " cur_keyframe->keylsd = " << cur_keyfrm_->_keylsd.size()
        //               << " ngh_keyframe->keylsd = " << ngh_keyfrm->_keylsd.size()
        //               << std::endl;
        // }

        // initialize the two view triangulator for line
        const module::two_view_triangulator_line triangulator_line(cur_keyfrm, ngh_keyfrm, 1.0);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (unsigned int k = 0; k < good_matches.size(); k++)
        {
            // variables should be all in Eigen, not in cv::Mat
            // get matched line segments
            cv::DMatch mt = good_matches[k];

            // avoid duplicate triangulation
            if (cur_keyfrm->get_landmark_line(mt.queryIdx) || ngh_keyfrm->get_landmark_line(mt.trainIdx))
                continue;

            Vec6_t pos_w_line;
            if (!triangulator_line.triangulate(mt.queryIdx, mt.trainIdx, pos_w_line))
            {
                continue;
            }

            // construct 3D line landmark
            auto lm_line = new data::Line(pos_w_line, cur_keyfrm, map_db_);

            // link keyframe to landmark
            lm_line->add_observation(cur_keyfrm, mt.queryIdx);
            lm_line->add_observation(ngh_keyfrm, mt.trainIdx);

            // link landmark to keyframe
            cur_keyfrm->add_landmark_line(lm_line, mt.queryIdx);
            ngh_keyfrm->add_landmark_line(lm_line, mt.trainIdx);

            // calculate distinctive descriptors
            lm_line->compute_descriptor();

            // update some information
            lm_line->update_information();

            // add landmark to the map database
            map_db_->add_landmark_line(lm_line);

#ifdef USE_OPENMP
#pragma omp critical
#endif
            {
                // add it also to the local map cleaner
                local_map_cleaner_->add_fresh_landmark_line(lm_line);
            }
        }
    }

    void mapping_module::update_new_keyframe()
    {
        // get the targets to check landmark fusion
        const auto fuse_tgt_keyfrms = get_second_order_covisibilities(is_monocular_ ? 20 : 10, 5); // std::unordered_set<data::keyframe *>

        // resolve the duplication of landmarks between the current keyframe and the targets
        fuse_landmark_duplication(fuse_tgt_keyfrms);

        // FW:
        if (map_db_->_b_use_line_tracking)
        {
            fuse_landmark_duplication_line(fuse_tgt_keyfrms);
        }

        // update the geometries (3D points)
        const auto cur_landmarks = cur_keyfrm_->get_landmarks();
        for (const auto lm : cur_landmarks)
        {
            if (!lm)
            {
                continue;
            }
            if (lm->will_be_erased())
            {
                continue;
            }
            lm->compute_descriptor();
            lm->update_normal_and_depth();
        }

        // FW: update the geometries (3D lines)
        if (map_db_->_b_use_line_tracking)
        {
            const auto cur_landmarks_line = cur_keyfrm_->get_landmarks_line();
            for (const auto lm_line : cur_landmarks_line)
            {
                if (!lm_line)
                {
                    continue;
                }
                if (lm_line->will_be_erased())
                {
                    continue;
                }
                lm_line->compute_descriptor();
                lm_line->update_information();
            }
        }

        // update the graph
        // FW: TODO: now not with 3D line for updating graph
        cur_keyfrm_->graph_node_->update_connections();
    }

    std::unordered_set<data::keyframe *> mapping_module::get_second_order_covisibilities(const unsigned int first_order_thr,
                                                                                         const unsigned int second_order_thr)
    {
        // if monocular, first_order_thr = 20; second_order_thr = 5
        const auto cur_covisibilities = cur_keyfrm_->graph_node_->get_top_n_covisibilities(first_order_thr); // std::vector<keyframe *>

        std::unordered_set<data::keyframe *> fuse_tgt_keyfrms;
        fuse_tgt_keyfrms.reserve(cur_covisibilities.size() * 2);

        for (const auto first_order_covis : cur_covisibilities)
        {
            if (first_order_covis->will_be_erased())
            {
                continue;
            }

            // check if the keyframe is aleady inserted
            if (static_cast<bool>(fuse_tgt_keyfrms.count(first_order_covis)))
            {
                continue;
            }

            fuse_tgt_keyfrms.insert(first_order_covis);

            // get the covisibilities of the covisibility of the current keyframe
            const auto ngh_covisibilities = first_order_covis->graph_node_->get_top_n_covisibilities(second_order_thr);
            for (const auto second_order_covis : ngh_covisibilities)
            {
                if (second_order_covis->will_be_erased())
                {
                    continue;
                }
                // "the covisibilities of the covisibility" contains the current keyframe
                if (*second_order_covis == *cur_keyfrm_)
                {
                    continue;
                }

                fuse_tgt_keyfrms.insert(second_order_covis);
            }
        }

        return fuse_tgt_keyfrms;
    }

    void mapping_module::fuse_landmark_duplication(const std::unordered_set<data::keyframe *> &fuse_tgt_keyfrms)
    {
        match::fuse matcher;

        {
            // reproject the landmarks observed in the current keyframe to each of the targets, and acquire
            // - additional matches
            // - duplication of matches
            // then, add matches and solve duplication
            auto cur_landmarks = cur_keyfrm_->get_landmarks();
            for (const auto fuse_tgt_keyfrm : fuse_tgt_keyfrms)
            {
                matcher.replace_duplication(fuse_tgt_keyfrm, cur_landmarks);
            }
        }

        {
            // reproject the landmarks observed in each of the targets to each of the current frame, and acquire
            // - additional matches
            // - duplication of matches
            // then, add matches and solve duplication
            std::unordered_set<data::landmark *> candidate_landmarks_to_fuse;
            candidate_landmarks_to_fuse.reserve(fuse_tgt_keyfrms.size() * cur_keyfrm_->num_keypts_);

            for (const auto fuse_tgt_keyfrm : fuse_tgt_keyfrms)
            {
                const auto fuse_tgt_landmarks = fuse_tgt_keyfrm->get_landmarks();

                for (const auto lm : fuse_tgt_landmarks)
                {
                    if (!lm)
                    {
                        continue;
                    }
                    if (lm->will_be_erased())
                    {
                        continue;
                    }

                    if (static_cast<bool>(candidate_landmarks_to_fuse.count(lm)))
                    {
                        continue;
                    }
                    candidate_landmarks_to_fuse.insert(lm);
                }
            }

            matcher.replace_duplication(cur_keyfrm_, candidate_landmarks_to_fuse);
        }
    }

    // FW:
    void mapping_module::fuse_landmark_duplication_line(const std::unordered_set<data::keyframe *> &fuse_tgt_keyfrms)
    {
        match::fuse matcher;

        float margin = 10.0; // FW: debug this margin

        {
            // reproject the landmarks observed in the current keyframe to each of the targets, and acquire
            // - additional matches
            // - duplication of matches
            // then, add matches and solve duplication
            auto cur_landmarks_line = cur_keyfrm_->get_landmarks_line();
            for (const auto fuse_tgt_keyfrm : fuse_tgt_keyfrms)
            {
                matcher.replace_duplication_line(fuse_tgt_keyfrm, cur_landmarks_line, margin);
            }
        }

        {
            // reproject the landmarks observed in each of the targets to each of the current frame, and acquire
            // - additional matches
            // - duplication of matches
            // then, add matches and solve duplication
            std::unordered_set<data::Line *> candidate_landmarks_to_fuse;
            candidate_landmarks_to_fuse.reserve(fuse_tgt_keyfrms.size() * cur_keyfrm_->_num_keylines);

            for (const auto fuse_tgt_keyfrm : fuse_tgt_keyfrms)
            {
                const auto fuse_tgt_landmarks_line = fuse_tgt_keyfrm->get_landmarks_line();

                for (const auto lm_line : fuse_tgt_landmarks_line)
                {
                    if (!lm_line)
                    {
                        continue;
                    }
                    if (lm_line->will_be_erased())
                    {
                        continue;
                    }

                    if (static_cast<bool>(candidate_landmarks_to_fuse.count(lm_line)))
                    {
                        continue;
                    }
                    candidate_landmarks_to_fuse.insert(lm_line);
                }
            }

            matcher.replace_duplication_line(cur_keyfrm_, candidate_landmarks_to_fuse, margin);
        }
    }

    void mapping_module::request_reset()
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

    bool mapping_module::reset_is_requested() const
    {
        std::lock_guard<std::mutex> lock(mtx_reset_);
        return reset_is_requested_;
    }

    void mapping_module::reset()
    {
        std::lock_guard<std::mutex> lock(mtx_reset_);
        spdlog::info("reset mapping module");
        keyfrms_queue_.clear();
        local_map_cleaner_->reset();
        reset_is_requested_ = false;
    }

    void mapping_module::request_pause()
    {
        std::lock_guard<std::mutex> lock1(mtx_pause_);
        pause_is_requested_ = true;
        std::lock_guard<std::mutex> lock2(mtx_keyfrm_queue_);
        abort_local_BA_ = true;
    }

    bool mapping_module::is_paused() const
    {
        std::lock_guard<std::mutex> lock(mtx_pause_);
        return is_paused_;
    }

    bool mapping_module::pause_is_requested() const
    {
        std::lock_guard<std::mutex> lock(mtx_pause_);
        return pause_is_requested_ && !force_to_run_;
    }

    void mapping_module::pause()
    {
        std::lock_guard<std::mutex> lock(mtx_pause_);
        spdlog::info("pause mapping module");
        is_paused_ = true;
    }

    bool mapping_module::set_force_to_run(const bool force_to_run)
    {
        std::lock_guard<std::mutex> lock(mtx_pause_);

        if (force_to_run && is_paused_)
        {
            return false;
        }

        force_to_run_ = force_to_run;
        return true;
    }

    void mapping_module::resume()
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

        // clear the queue
        for (auto &new_keyframe : keyfrms_queue_)
        {
            delete new_keyframe;
        }
        keyfrms_queue_.clear();

        spdlog::info("resume mapping module");
    }

    void mapping_module::request_terminate()
    {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        terminate_is_requested_ = true;
    }

    bool mapping_module::is_terminated() const
    {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        return is_terminated_;
    }

    bool mapping_module::terminate_is_requested() const
    {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        return terminate_is_requested_;
    }

    void mapping_module::terminate()
    {
        std::lock_guard<std::mutex> lock1(mtx_pause_);
        std::lock_guard<std::mutex> lock2(mtx_terminate_);
        is_paused_ = true;
        is_terminated_ = true;
    }

    // FW: used for debug
    void mapping_module::visualize_line_match(cv::Mat imageMat1, cv::Mat imageMat2,
                                              std::vector<cv::line_descriptor::KeyLine> octave0_1,
                                              std::vector<cv::line_descriptor::KeyLine> octave0_2,
                                              std::vector<cv::DMatch> good_matches)
    {
        cv::Mat img1, img2;
        if (imageMat1.channels() != 3)
        {
            cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
        }
        else
        {
            img1 = imageMat1;
        }

        if (imageMat2.channels() != 3)
        {
            cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);
        }
        else
        {
            img2 = imageMat2;
        }

        cv::Mat lsd_outImg;
        std::vector<char> lsd_mask(good_matches.size(), 1);
        drawLineMatches(img1, octave0_1, img2, octave0_2, good_matches, lsd_outImg,
                        cv::Scalar::all(-1), cv::Scalar::all(-1), lsd_mask,
                        cv::line_descriptor::DrawLinesMatchesFlags::DEFAULT);

        int lowest = 0, highest = 255;
        int range = (highest - lowest) + 1;
        for (size_t k = 0; k < good_matches.size(); ++k)
        {
            cv::DMatch mt = good_matches[k];

            cv::line_descriptor::KeyLine line1 = octave0_1[mt.queryIdx]; // trainIdx
            cv::line_descriptor::KeyLine line2 = octave0_2[mt.trainIdx]; // queryIdx

            unsigned int r = lowest + int(rand() % range);
            unsigned int g = lowest + int(rand() % range);
            unsigned int b = lowest + int(rand() % range);
            cv::Point startPoint = cv::Point(int(line1.startPointX), int(line1.startPointY));
            cv::Point endPoint = cv::Point(int(line1.endPointX), int(line1.endPointY));
            cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b), 2, 8);

            cv::Point startPoint2 = cv::Point(int(line2.startPointX), int(line2.startPointY));
            cv::Point endPoint2 = cv::Point(int(line2.endPointX), int(line2.endPointY));
            cv::line(img2, startPoint2, endPoint2, cv::Scalar(r, g, b), 2, 8);

            // visualize the shift of ending points in image2
            cv::line(img2, startPoint, startPoint2, cv::Scalar(0, 0, 255), 1, 8);
            cv::line(img2, endPoint, endPoint2, cv::Scalar(0, 0, 255), 1, 8);
        }

        std::string name = std::to_string(_frame_num);
        std::string path = "/home/shu/Documents/image/";
        name = path + name + ".jpg";
        _frame_num++;
        cv::imwrite(name, lsd_outImg);

        /* plot matches */
        // cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);
        // cv::namedWindow("LSD matches");
        // cv::imshow("LSD matches", lsd_outImg);
        // // namedWindow("LSD matches1", CV_WINDOW_NORMAL);
        // cv::namedWindow("LSD matches2");
        // imshow("LSD matches1", img1);
        // cv::imshow("LSD matches2", img2);
        // cv::waitKey(1);
    }
} // namespace PLPSLAM
