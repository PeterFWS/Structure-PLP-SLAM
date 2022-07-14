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
#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/data/map_database.h"
#include "PLPSLAM/module/loop_bundle_adjuster.h"
#include "PLPSLAM/optimize/global_bundle_adjuster.h"

#include <thread>

#include <spdlog/spdlog.h>

namespace PLPSLAM
{
    namespace module
    {

        loop_bundle_adjuster::loop_bundle_adjuster(data::map_database *map_db, const unsigned int num_iter)
            : map_db_(map_db), num_iter_(num_iter)
        {
        }

        void loop_bundle_adjuster::set_mapping_module(mapping_module *mapper)
        {
            mapper_ = mapper;
        }

        void loop_bundle_adjuster::count_loop_BA_execution()
        {
            std::lock_guard<std::mutex> lock(mtx_thread_);
            ++num_exec_loop_BA_;
        }

        void loop_bundle_adjuster::abort()
        {
            std::lock_guard<std::mutex> lock(mtx_thread_);
            abort_loop_BA_ = true;
        }

        bool loop_bundle_adjuster::is_running() const
        {
            std::lock_guard<std::mutex> lock(mtx_thread_);
            return loop_BA_is_running_;
        }

        void loop_bundle_adjuster::optimize(const unsigned int identifier)
        {
            spdlog::info("start loop bundle adjustment");

            unsigned int num_exec_loop_BA = 0;
            {
                std::lock_guard<std::mutex> lock(mtx_thread_);
                loop_BA_is_running_ = true;
                abort_loop_BA_ = false;
                num_exec_loop_BA = num_exec_loop_BA_;
            }

            // FW: global BA with point and line
            const auto global_bundle_adjuster = optimize::global_bundle_adjuster(map_db_, num_iter_, false);
            global_bundle_adjuster.optimize(identifier, &abort_loop_BA_);

            {
                std::lock_guard<std::mutex> lock1(mtx_thread_);

                // if count_loop_BA_execution() was called during the loop BA or the loop BA was aborted,
                // cannot update the map
                if (num_exec_loop_BA != num_exec_loop_BA_ || abort_loop_BA_)
                {
                    spdlog::info("abort loop bundle adjustment");
                    loop_BA_is_running_ = false;
                    abort_loop_BA_ = false;
                    return;
                }

                spdlog::info("finish loop bundle adjustment");
                spdlog::info("updating the map with pose propagation");

                // stop mapping module
                mapper_->request_pause();
                while (!mapper_->is_paused() && !mapper_->is_terminated())
                {
                    std::this_thread::sleep_for(std::chrono::microseconds(1000));
                }

                std::lock_guard<std::mutex> lock2(data::map_database::mtx_database_);

                // update the camera pose along the spanning tree from the origin
                std::list<data::keyframe *> keyfrms_to_check;
                keyfrms_to_check.push_back(map_db_->origin_keyfrm_);
                while (!keyfrms_to_check.empty())
                {
                    auto parent = keyfrms_to_check.front();
                    const Mat44_t cam_pose_wp = parent->get_cam_pose_inv();

                    const auto children = parent->graph_node_->get_spanning_children();
                    for (auto child : children)
                    {
                        if (child->loop_BA_identifier_ != identifier)
                        {
                            // if `child` is NOT optimized by the loop BA
                            // propagate the pose correction from the spanning parent

                            // parent->child
                            const Mat44_t cam_pose_cp = child->get_cam_pose() * cam_pose_wp;
                            // world->child AFTER correction = parent->child * world->parent AFTER correction
                            child->cam_pose_cw_after_loop_BA_ = cam_pose_cp * parent->cam_pose_cw_after_loop_BA_;
                            // check as `child` has been corrected
                            child->loop_BA_identifier_ = identifier;
                        }

                        // need updating
                        keyfrms_to_check.push_back(child);
                    }

                    // temporally store the camera pose BEFORE correction (for correction of landmark positions)
                    parent->cam_pose_cw_before_BA_ = parent->get_cam_pose();
                    // update the camera pose
                    parent->set_cam_pose(parent->cam_pose_cw_after_loop_BA_);
                    // finish updating
                    keyfrms_to_check.pop_front();
                }

                // update the positions of the landmarks
                const auto landmarks = map_db_->get_all_landmarks();
                for (auto lm : landmarks)
                {
                    if (lm->will_be_erased())
                    {
                        continue;
                    }

                    if (lm->loop_BA_identifier_ == identifier)
                    {
                        // if `lm` is optimized by the loop BA

                        // update with the optimized position
                        lm->set_pos_in_world(lm->pos_w_after_global_BA_);
                    }
                    else
                    {
                        // if `lm` is NOT optimized by the loop BA

                        // correct the position according to the move of the camera pose of the reference keyframe
                        auto ref_keyfrm = lm->get_ref_keyframe();

                        assert(ref_keyfrm->loop_BA_identifier_ == identifier);

                        // convert the position to the camera-reference using the camera pose BEFORE the correction
                        const Mat33_t rot_cw_before_BA = ref_keyfrm->cam_pose_cw_before_BA_.block<3, 3>(0, 0);
                        const Vec3_t trans_cw_before_BA = ref_keyfrm->cam_pose_cw_before_BA_.block<3, 1>(0, 3);
                        const Vec3_t pos_c = rot_cw_before_BA * lm->get_pos_in_world() + trans_cw_before_BA;

                        // convert the position to the world-reference using the camera pose AFTER the correction
                        const Mat44_t cam_pose_wc = ref_keyfrm->get_cam_pose_inv();
                        const Mat33_t rot_wc = cam_pose_wc.block<3, 3>(0, 0);
                        const Vec3_t trans_wc = cam_pose_wc.block<3, 1>(0, 3);
                        lm->set_pos_in_world(rot_wc * pos_c + trans_wc);
                    }
                }

                // FW:
                // update the positions of the line cloud
                const auto landmarks_line = map_db_->get_all_landmarks_line();
                if (map_db_->_b_use_line_tracking && !landmarks_line.empty())
                {
                    for (auto lm_line : landmarks_line)
                    {
                        if (lm_line->will_be_erased())
                        {
                            continue;
                        }

                        if (lm_line->_loop_BA_identifier == identifier)
                        {
                            // if `lm` is optimized by the loop BA

                            // update with the optimized position
                            lm_line->set_PlueckerCoord_without_update_endpoints(lm_line->_pos_w_after_global_BA);

                            // endpoint trimming via its reference keyframe
                            Vec6_t updated_pose_w;
                            if (endpoint_trimming(lm_line, lm_line->_pos_w_after_global_BA, updated_pose_w))
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
                            // if `lm` is NOT optimized by the loop BA

                            // correct the position according to the move of the camera pose of the reference keyframe
                            auto ref_keyfrm = lm_line->get_ref_keyframe();

                            assert(ref_keyfrm->loop_BA_identifier_ == identifier);

                            // convert the position to the camera-reference using the camera pose BEFORE the correction
                            const Mat33_t rot_cw_before_BA = ref_keyfrm->cam_pose_cw_before_BA_.block<3, 3>(0, 0);
                            const Vec3_t trans_cw_before_BA = ref_keyfrm->cam_pose_cw_before_BA_.block<3, 1>(0, 3);

                            Mat66_t transformation_line_cw = Eigen::Matrix<double, 6, 6>::Zero();
                            transformation_line_cw.block<3, 3>(0, 0) = rot_cw_before_BA;
                            transformation_line_cw.block<3, 3>(3, 3) = rot_cw_before_BA;
                            transformation_line_cw.block<3, 3>(0, 3) = skew(trans_cw_before_BA) * rot_cw_before_BA;
                            const Vec6_t pos_c_pluecker = transformation_line_cw * lm_line->get_PlueckerCoord();

                            // convert the position to the world-reference using the camera pose AFTER the correction
                            const Mat44_t cam_pose_wc = ref_keyfrm->get_cam_pose_inv();
                            const Mat33_t rot_wc = cam_pose_wc.block<3, 3>(0, 0);
                            const Vec3_t trans_wc = cam_pose_wc.block<3, 1>(0, 3);

                            Mat66_t transformation_line_wc = Eigen::Matrix<double, 6, 6>::Zero();
                            transformation_line_wc.block<3, 3>(0, 0) = rot_wc;
                            transformation_line_wc.block<3, 3>(3, 3) = rot_wc;
                            transformation_line_wc.block<3, 3>(0, 3) = skew(trans_wc) * rot_wc;
                            Vec6_t pos_w_pluecker = transformation_line_wc * pos_c_pluecker;

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
                    }
                }

                mapper_->resume();
                loop_BA_is_running_ = false;

                spdlog::info("updated the map");
            }
        }

        // FW:
        bool loop_bundle_adjuster::endpoint_trimming(data::Line *local_lm_line,
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

    } // namespace module
} // namespace PLPSLAM
