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

#include "PLPSLAM/camera/perspective.h"
#include "PLPSLAM/camera/fisheye.h"
#include "PLPSLAM/camera/equirectangular.h"
#include "PLPSLAM/data/common.h"
#include "PLPSLAM/data/frame.h"
#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/data/map_database.h"
#include "PLPSLAM/data/bow_database.h"
#include "PLPSLAM/feature/orb_params.h"
#include "PLPSLAM/util/converter.h"

#include <nlohmann/json.hpp>

namespace PLPSLAM
{
    namespace data
    {

        std::atomic<unsigned int> keyframe::next_id_{0};

        // FW: from a frame
        keyframe::keyframe(const frame &frm, map_database *map_db, bow_database *bow_db)
            : // meta information
              id_(next_id_++), src_frm_id_(frm.id_), timestamp_(frm.timestamp_),
              // camera parameters
              camera_(frm.camera_), depth_thr_(frm.depth_thr_),
              // constant observations
              num_keypts_(frm.num_keypts_), keypts_(frm.keypts_), undist_keypts_(frm.undist_keypts_), bearings_(frm.bearings_),
              keypt_indices_in_cells_(frm.keypt_indices_in_cells_),
              stereo_x_right_(frm.stereo_x_right_), depths_(frm.depths_), descriptors_(frm.descriptors_.clone()),
              // BoW
              bow_vec_(frm.bow_vec_), bow_feat_vec_(frm.bow_feat_vec_),
              // covisibility graph node (connections is not assigned yet)
              graph_node_(std::unique_ptr<graph_node>(new graph_node(this, true))),
              // ORB scale pyramid
              num_scale_levels_(frm.num_scale_levels_), scale_factor_(frm.scale_factor_),
              log_scale_factor_(frm.log_scale_factor_), scale_factors_(frm.scale_factors_),
              level_sigma_sq_(frm.level_sigma_sq_), inv_level_sigma_sq_(frm.inv_level_sigma_sq_),
              // FW:
              _keylsd(frm._keylsd), _lbd_descr(frm._lbd_descr.clone()),
              _keyline_functions(frm._keyline_functions), _num_keylines(frm._num_keylines),
              _depths_cooresponding_to_keylines(frm._depths_cooresponding_to_keylines),
              _stereo_x_right_cooresponding_to_keylines(frm._stereo_x_right_cooresponding_to_keylines),
              // FW:
              _keylsd_right(frm._keylsd_right), _lbd_descr_right(frm._lbd_descr_right),
              _keyline_functions_right(frm._keyline_functions_right), _good_matches_stereo(frm._good_matches_stereo),
              // FW:
              _num_scale_levels_lsd(frm._num_scale_levels_lsd), _scale_factor_lsd(frm._scale_factor_lsd),
              _log_scale_factor_lsd(frm._log_scale_factor_lsd), _scale_factors_lsd(frm._scale_factors_lsd),
              _inv_scale_factors_lsd(frm._inv_scale_factors_lsd), _level_sigma_sq_lsd(frm._level_sigma_sq_lsd),
              _inv_level_sigma_sq_lsd(frm._inv_level_sigma_sq_lsd),
              // observations
              landmarks_(frm.landmarks_),
              // FW:
              _landmarks_line(frm._landmarks_line),
              // databases
              map_db_(map_db), bow_db_(bow_db), bow_vocab_(frm.bow_vocab_)
        {
            // set pose parameters (cam_pose_wc_, cam_center_) using frm.cam_pose_cw_
            set_cam_pose(frm.cam_pose_cw_);
        }

        // FW: from map loading
        keyframe::keyframe(const unsigned int id, const unsigned int src_frm_id, const double timestamp,
                           const Mat44_t &cam_pose_cw, camera::base *camera, const float depth_thr,
                           const unsigned int num_keypts, const std::vector<cv::KeyPoint> &keypts,
                           const std::vector<cv::KeyPoint> &undist_keypts, const eigen_alloc_vector<Vec3_t> &bearings,
                           const std::vector<float> &stereo_x_right, const std::vector<float> &depths, const cv::Mat &descriptors,
                           const unsigned int num_scale_levels, const float scale_factor,
                           bow_vocabulary *bow_vocab, bow_database *bow_db, map_database *map_db)
            : // meta information
              id_(id), src_frm_id_(src_frm_id), timestamp_(timestamp),
              // camera parameters
              camera_(camera), depth_thr_(depth_thr),
              // constant observations
              num_keypts_(num_keypts), keypts_(keypts), undist_keypts_(undist_keypts), bearings_(bearings),
              keypt_indices_in_cells_(assign_keypoints_to_grid(camera, undist_keypts)),
              stereo_x_right_(stereo_x_right), depths_(depths), descriptors_(descriptors.clone()),
              // graph node (connections is not assigned yet)
              graph_node_(std::unique_ptr<graph_node>(new graph_node(this, false))),
              // ORB scale pyramid
              num_scale_levels_(num_scale_levels), scale_factor_(scale_factor), log_scale_factor_(std::log(scale_factor)),
              scale_factors_(feature::orb_params::calc_scale_factors(num_scale_levels, scale_factor)),
              level_sigma_sq_(feature::orb_params::calc_level_sigma_sq(num_scale_levels, scale_factor)),
              inv_level_sigma_sq_(feature::orb_params::calc_inv_level_sigma_sq(num_scale_levels, scale_factor)),
              // others
              landmarks_(std::vector<landmark *>(num_keypts, nullptr)),
              // databases
              map_db_(map_db), bow_db_(bow_db), bow_vocab_(bow_vocab)
        {
            // compute BoW (bow_vec_, bow_feat_vec_) using descriptors_
            compute_bow();
            // set pose parameters (cam_pose_wc_, cam_center_) using cam_pose_cw_
            set_cam_pose(cam_pose_cw);

            // TODO: should set the pointers of landmarks_ using add_landmark()

            // TODO: should compute connected_keyfrms_and_weights_
            // TODO: should compute ordered_connected_keyfrms_
            // TODO: should compute ordered_weights_

            // TODO: should set spanning_parent_ using set_spanning_parent()
            // TODO: should set spanning_children_ using add_spanning_child()
            // TODO: should set loop_edges_ using add_loop_edge()
        }

        // FW: from map loading: points + lines
        keyframe::keyframe(const unsigned int id, const unsigned int src_frm_id, const double timestamp,
                           const Mat44_t &cam_pose_cw, camera::base *camera, const float depth_thr,
                           const unsigned int num_keypts, const std::vector<cv::KeyPoint> &keypts,
                           const std::vector<cv::KeyPoint> &undist_keypts, const eigen_alloc_vector<Vec3_t> &bearings,
                           const std::vector<float> &stereo_x_right, const std::vector<float> &depths, const cv::Mat &descriptors,
                           const unsigned int num_scale_levels, const float scale_factor,
                           const unsigned int num_keylines, const std::vector<cv::line_descriptor::KeyLine> &keylines,
                           const std::vector<std::pair<float, float>> &stereo_x_right_keylines,
                           const std::vector<std::pair<float, float>> &depths_keylines,
                           const cv::Mat &lbd_descriptors,
                           const unsigned int num_scale_levels_lsd, const float scale_factor_lsd,
                           bow_vocabulary *bow_vocab, bow_database *bow_db, map_database *map_db)
            : // meta information
              id_(id), src_frm_id_(src_frm_id), timestamp_(timestamp),
              // camera parameters
              camera_(camera), depth_thr_(depth_thr),
              // constant observations
              num_keypts_(num_keypts), keypts_(keypts), undist_keypts_(undist_keypts), bearings_(bearings),
              keypt_indices_in_cells_(assign_keypoints_to_grid(camera, undist_keypts)),
              stereo_x_right_(stereo_x_right), depths_(depths), descriptors_(descriptors.clone()),
              // graph node (connections is not assigned yet)
              graph_node_(std::unique_ptr<graph_node>(new graph_node(this, false))),
              // ORB scale pyramid
              num_scale_levels_(num_scale_levels), scale_factor_(scale_factor), log_scale_factor_(std::log(scale_factor)),
              scale_factors_(feature::orb_params::calc_scale_factors(num_scale_levels, scale_factor)),
              level_sigma_sq_(feature::orb_params::calc_level_sigma_sq(num_scale_levels, scale_factor)),
              inv_level_sigma_sq_(feature::orb_params::calc_inv_level_sigma_sq(num_scale_levels, scale_factor)),
              // FW:
              _num_keylines(num_keylines), _keylsd(keylines),
              _stereo_x_right_cooresponding_to_keylines(stereo_x_right_keylines),
              _depths_cooresponding_to_keylines(depths_keylines),
              _lbd_descr(lbd_descriptors),
              // FW:
              _num_scale_levels_lsd(num_scale_levels_lsd), _scale_factor_lsd(scale_factor_lsd),
              // others
              landmarks_(std::vector<landmark *>(num_keypts, nullptr)),
              // FW:
              _landmarks_line(std::vector<Line *>(num_keylines, nullptr)),
              // databases
              map_db_(map_db), bow_db_(bow_db), bow_vocab_(bow_vocab)
        {
            // compute BoW (bow_vec_, bow_feat_vec_) using descriptors_
            compute_bow();
            // set pose parameters (cam_pose_wc_, cam_center_) using cam_pose_cw_
            set_cam_pose(cam_pose_cw);

            // TODO: should set the pointers of landmarks_ using add_landmark()

            // TODO: should compute connected_keyfrms_and_weights_
            // TODO: should compute ordered_connected_keyfrms_
            // TODO: should compute ordered_weights_

            // TODO: should set spanning_parent_ using set_spanning_parent()
            // TODO: should set spanning_children_ using add_spanning_child()
            // TODO: should set loop_edges_ using add_loop_edge()

            // FW:
            initialize_lsd_scale_information();
        }

        nlohmann::json keyframe::to_json() const
        {
            // extract landmark IDs
            std::vector<int> landmark_ids(landmarks_.size(), -1);
            for (unsigned int i = 0; i < landmark_ids.size(); ++i)
            {
                if (landmarks_.at(i) && !landmarks_.at(i)->will_be_erased())
                {
                    landmark_ids.at(i) = landmarks_.at(i)->id_;
                }
            }

            // extract spanning tree parent
            auto spanning_parent = graph_node_->get_spanning_parent();

            // extract spanning tree children
            const auto spanning_children = graph_node_->get_spanning_children();
            std::vector<int> spanning_child_ids;
            spanning_child_ids.reserve(spanning_children.size());
            for (const auto spanning_child : spanning_children)
            {
                spanning_child_ids.push_back(spanning_child->id_);
            }

            // extract loop edges
            const auto loop_edges = graph_node_->get_loop_edges();
            std::vector<int> loop_edge_ids;
            for (const auto loop_edge : loop_edges)
            {
                loop_edge_ids.push_back(loop_edge->id_);
            }

            if (!map_db_->_b_use_line_tracking)
            {
                return {{"src_frm_id", src_frm_id_},
                        {"ts", timestamp_},
                        {"cam", camera_->name_},
                        {"depth_thr", depth_thr_},
                        // camera pose
                        {"rot_cw", convert_rotation_to_json(cam_pose_cw_.block<3, 3>(0, 0))},
                        {"trans_cw", convert_translation_to_json(cam_pose_cw_.block<3, 1>(0, 3))},
                        // features and observations
                        {"n_keypts", num_keypts_},
                        {"keypts", convert_keypoints_to_json(keypts_)},
                        {"undists", convert_undistorted_to_json(undist_keypts_)},
                        {"x_rights", stereo_x_right_},
                        {"depths", depths_},
                        {"descs", convert_descriptors_to_json(descriptors_)},
                        {"lm_ids", landmark_ids},
                        // orb scale information
                        {"n_scale_levels", num_scale_levels_},
                        {"scale_factor", scale_factor_},
                        // graph information
                        {"span_parent", spanning_parent ? spanning_parent->id_ : -1},
                        {"span_children", spanning_child_ids},
                        {"loop_edges", loop_edge_ids}};
            }
            else
            {
                // FW: extract landmark_line IDs
                std::vector<int> landmark_line_ids(_landmarks_line.size(), -1);
                for (unsigned int i = 0; i < landmark_line_ids.size(); ++i)
                {
                    if (_landmarks_line.at(i) && !_landmarks_line.at(i)->will_be_erased())
                    {
                        landmark_line_ids.at(i) = _landmarks_line.at(i)->_id;
                    }
                }

                return {
                    {"src_frm_id", src_frm_id_},
                    {"ts", timestamp_},
                    {"cam", camera_->name_},
                    {"depth_thr", depth_thr_},
                    // camera pose
                    {"rot_cw", convert_rotation_to_json(cam_pose_cw_.block<3, 3>(0, 0))},
                    {"trans_cw", convert_translation_to_json(cam_pose_cw_.block<3, 1>(0, 3))},
                    // features and observations
                    {"n_keypts", num_keypts_},
                    {"keypts", convert_keypoints_to_json(keypts_)},
                    {"undists", convert_undistorted_to_json(undist_keypts_)},
                    {"x_rights", stereo_x_right_},
                    {"depths", depths_},
                    {"descs", convert_descriptors_to_json(descriptors_)},
                    {"lm_ids", landmark_ids},
                    // orb scale information
                    {"n_scale_levels", num_scale_levels_},
                    {"scale_factor", scale_factor_},
                    // graph information
                    {"span_parent", spanning_parent ? spanning_parent->id_ : -1},
                    {"span_children", spanning_child_ids},
                    {"loop_edges", loop_edge_ids},
                    // 3D lines
                    {"n_keylines", _num_keylines},
                    {"keylines", convert_keylines_to_json(_keylsd)},
                    {"x_rights_keylines", _stereo_x_right_cooresponding_to_keylines},
                    {"depths_keylines", _depths_cooresponding_to_keylines},
                    {"descs_keylines", convert_lbd_descriptors_to_json(_lbd_descr)},
                    {"lm_line_ids", landmark_line_ids},
                    // LSD scale information
                    {"n_scale_levels_lsd", _num_scale_levels_lsd},
                    {"scale_factor_lsd", _scale_factor_lsd}};
            }
        }

        void keyframe::set_cam_pose(const Mat44_t &cam_pose_cw)
        {
            std::lock_guard<std::mutex> lock(mtx_pose_);
            cam_pose_cw_ = cam_pose_cw;

            const Mat33_t rot_cw = cam_pose_cw_.block<3, 3>(0, 0);
            const Vec3_t trans_cw = cam_pose_cw_.block<3, 1>(0, 3);
            const Mat33_t rot_wc = rot_cw.transpose();
            cam_center_ = -rot_wc * trans_cw;

            cam_pose_wc_ = Mat44_t::Identity();
            cam_pose_wc_.block<3, 3>(0, 0) = rot_wc;
            cam_pose_wc_.block<3, 1>(0, 3) = cam_center_;
        }

        void keyframe::set_cam_pose(const g2o::SE3Quat &cam_pose_cw)
        {
            set_cam_pose(util::converter::to_eigen_mat(cam_pose_cw));
        }

        Mat44_t keyframe::get_cam_pose() const
        {
            std::lock_guard<std::mutex> lock(mtx_pose_);
            return cam_pose_cw_;
        }

        Mat44_t keyframe::get_cam_pose_inv() const
        {
            std::lock_guard<std::mutex> lock(mtx_pose_);
            return cam_pose_wc_;
        }

        Vec3_t keyframe::get_cam_center() const
        {
            std::lock_guard<std::mutex> lock(mtx_pose_);
            return cam_center_;
        }

        Mat33_t keyframe::get_rotation() const
        {
            std::lock_guard<std::mutex> lock(mtx_pose_);
            return cam_pose_cw_.block<3, 3>(0, 0);
        }

        Vec3_t keyframe::get_translation() const
        {
            std::lock_guard<std::mutex> lock(mtx_pose_);
            return cam_pose_cw_.block<3, 1>(0, 3);
        }

        void keyframe::compute_bow()
        {
            if (bow_vec_.empty() || bow_feat_vec_.empty())
            {
#ifdef USE_DBOW2
                bow_vocab_->transform(util::converter::to_desc_vec(descriptors_), bow_vec_, bow_feat_vec_, 4);
#else
                bow_vocab_->transform(descriptors_, 4, bow_vec_, bow_feat_vec_);
#endif
            }
        }

        void keyframe::add_landmark(landmark *lm, const unsigned int idx)
        {
            std::lock_guard<std::mutex> lock(mtx_observations_);
            landmarks_.at(idx) = lm;
        }

        // FW:
        void keyframe::add_landmark_line(Line *line, const unsigned int idx)
        {
            std::lock_guard<std::mutex> lock(mtx_observations_);
            _landmarks_line.at(idx) = line;
        }

        void keyframe::erase_landmark_with_index(const unsigned int idx)
        {
            std::lock_guard<std::mutex> lock(mtx_observations_);
            landmarks_.at(idx) = nullptr;
        }

        // FW:
        void keyframe::erase_landmark_line_with_index(const unsigned int idx)
        {
            std::lock_guard<std::mutex> lock(mtx_observations_);
            _landmarks_line.at(idx) = nullptr;
        }

        void keyframe::erase_landmark(landmark *lm)
        {
            int idx = lm->get_index_in_keyframe(this);
            if (0 <= idx)
            {
                landmarks_.at(static_cast<unsigned int>(idx)) = nullptr;
            }
        }

        // FW:
        void keyframe::erase_landmark_line(Line *line)
        {
            int idx = line->get_index_in_keyframe(this);
            if (0 <= idx)
            {
                _landmarks_line.at(static_cast<unsigned int>(idx)) = nullptr;
            }
        }

        void keyframe::replace_landmark(landmark *lm, const unsigned int idx)
        {
            landmarks_.at(idx) = lm;
        }

        // FW:
        void keyframe::replace_landmark_line(Line *line, const unsigned int idx)
        {
            _landmarks_line.at(idx) = line;
        }

        std::vector<landmark *> keyframe::get_landmarks() const
        {
            std::lock_guard<std::mutex> lock(mtx_observations_);
            return landmarks_;
        }

        // FW:
        std::vector<Line *> keyframe::get_landmarks_line() const
        {
            std::lock_guard<std::mutex> lock(mtx_observations_);
            return _landmarks_line;
        }

        std::set<landmark *> keyframe::get_valid_landmarks() const
        {
            std::lock_guard<std::mutex> lock(mtx_observations_);
            std::set<landmark *> valid_landmarks;

            for (const auto lm : landmarks_)
            {
                if (!lm)
                {
                    continue;
                }
                if (lm->will_be_erased())
                {
                    continue;
                }

                valid_landmarks.insert(lm);
            }

            return valid_landmarks;
        }

        // FW:
        std::set<Line *> keyframe::get_valid_landmarks_line() const
        {
            std::lock_guard<std::mutex> lock(mtx_observations_);
            std::set<Line *> valid_landmarks;

            for (const auto lm : _landmarks_line)
            {
                if (!lm)
                {
                    continue;
                }
                if (lm->will_be_erased())
                {
                    continue;
                }

                valid_landmarks.insert(lm);
            }

            return valid_landmarks;
        }

        unsigned int keyframe::get_num_tracked_landmarks(const unsigned int min_num_obs_thr) const
        {
            std::lock_guard<std::mutex> lock(mtx_observations_);
            unsigned int num_tracked_lms = 0;

            if (0 < min_num_obs_thr)
            {
                for (const auto lm : landmarks_)
                {
                    if (!lm)
                    {
                        continue;
                    }
                    if (lm->will_be_erased())
                    {
                        continue;
                    }

                    if (min_num_obs_thr <= lm->num_observations())
                    {
                        ++num_tracked_lms;
                    }
                }
            }
            else
            {
                for (const auto lm : landmarks_)
                {
                    if (!lm)
                    {
                        continue;
                    }
                    if (lm->will_be_erased())
                    {
                        continue;
                    }

                    ++num_tracked_lms;
                }
            }

            return num_tracked_lms;
        }

        // FW:
        unsigned int keyframe::get_num_tracked_landmarks_line(const unsigned int min_num_obs_thr) const
        {
            std::lock_guard<std::mutex> lock(mtx_observations_);
            unsigned int num_tracked_lms = 0;

            if (0 < min_num_obs_thr)
            {
                for (const auto lm : _landmarks_line)
                {
                    if (!lm)
                    {
                        continue;
                    }
                    if (lm->will_be_erased())
                    {
                        continue;
                    }

                    if (min_num_obs_thr <= lm->num_observations())
                    {
                        ++num_tracked_lms;
                    }
                }
            }
            else
            {
                for (const auto lm : _landmarks_line)
                {
                    if (!lm)
                    {
                        continue;
                    }
                    if (lm->will_be_erased())
                    {
                        continue;
                    }

                    ++num_tracked_lms;
                }
            }

            return num_tracked_lms;
        }

        landmark *keyframe::get_landmark(const unsigned int idx) const
        {
            std::lock_guard<std::mutex> lock(mtx_observations_);
            return landmarks_.at(idx);
        }

        // FW:
        Line *keyframe::get_landmark_line(const unsigned int idx) const
        {
            std::lock_guard<std::mutex> lock(mtx_observations_);
            return _landmarks_line.at(idx);
        }

        std::vector<unsigned int> keyframe::get_keypoints_in_cell(const float ref_x, const float ref_y, const float margin) const
        {
            return data::get_keypoints_in_cell(camera_, undist_keypts_, keypt_indices_in_cells_, ref_x, ref_y, margin);
        }

        // FW: same from frame.cc
        std::vector<unsigned int> keyframe::get_keylines_in_cell(const float ref_x1, const float ref_y1,
                                                                 const float ref_x2, const float ref_y2,
                                                                 const float margin,
                                                                 const int min_level, const int max_level) const
        {
            return data::get_keylines_in_cell(_keylsd, ref_x1, ref_y1, ref_x2, ref_y2, margin, min_level, max_level);
        }

        Vec3_t keyframe::triangulate_stereo(const unsigned int idx) const
        {
            assert(camera_->setup_type_ != camera::setup_type_t::Monocular);

            switch (camera_->model_type_)
            {
            case camera::model_type_t::Perspective:
            {
                auto camera = static_cast<camera::perspective *>(camera_);

                const float depth = depths_.at(idx);
                if (0.0 < depth)
                {
                    const float x = undist_keypts_.at(idx).pt.x;
                    const float y = undist_keypts_.at(idx).pt.y;
                    const float unproj_x = (x - camera->cx_) * depth * camera->fx_inv_;
                    const float unproj_y = (y - camera->cy_) * depth * camera->fy_inv_;
                    const Vec3_t pos_c{unproj_x, unproj_y, depth};

                    std::lock_guard<std::mutex> lock(mtx_pose_);
                    return cam_pose_wc_.block<3, 3>(0, 0) * pos_c + cam_pose_wc_.block<3, 1>(0, 3);
                }
                else
                {
                    return Vec3_t::Zero();
                }
            }
            case camera::model_type_t::Fisheye:
            {
                auto camera = static_cast<camera::fisheye *>(camera_);

                const float depth = depths_.at(idx);
                if (0.0 < depth)
                {
                    const float x = undist_keypts_.at(idx).pt.x;
                    const float y = undist_keypts_.at(idx).pt.y;
                    const float unproj_x = (x - camera->cx_) * depth * camera->fx_inv_;
                    const float unproj_y = (y - camera->cy_) * depth * camera->fy_inv_;
                    const Vec3_t pos_c{unproj_x, unproj_y, depth};

                    std::lock_guard<std::mutex> lock(mtx_pose_);
                    return cam_pose_wc_.block<3, 3>(0, 0) * pos_c + cam_pose_wc_.block<3, 1>(0, 3);
                }
                else
                {
                    return Vec3_t::Zero();
                }
            }
            case camera::model_type_t::Equirectangular:
            {
                throw std::runtime_error("Not implemented: Stereo or RGBD of equirectangular camera model");
            }
            }

            return Vec3_t::Zero();
        }

        // FW: used for triangulation (RGB-D/Stereo)
        Vec6_t keyframe::triangulate_stereo_for_line(const unsigned int idx) const
        {
            assert(camera_->setup_type_ != camera::setup_type_t::Monocular);

            auto camera = static_cast<camera::perspective *>(camera_);

            // generate 3D lines using depth
            if (camera_->setup_type_ == camera::setup_type_t::RGBD)
            {
                const float depth_sp = _depths_cooresponding_to_keylines.at(idx).first;
                const float depth_ep = _depths_cooresponding_to_keylines.at(idx).second;

                if (0.0 < depth_sp && 0.0 < depth_ep)
                {
                    const float x_sp = _keylsd.at(idx).getStartPoint().x;
                    const float y_sp = _keylsd.at(idx).getStartPoint().y;
                    const float unproj_x_sp = (x_sp - camera->cx_) * depth_sp * camera->fx_inv_;
                    const float unproj_y_sp = (y_sp - camera->cy_) * depth_sp * camera->fy_inv_;

                    const float x_ep = _keylsd.at(idx).getEndPoint().x;
                    const float y_ep = _keylsd.at(idx).getEndPoint().y;
                    const float unproj_x_ep = (x_ep - camera->cx_) * depth_ep * camera->fx_inv_;
                    const float unproj_y_ep = (y_ep - camera->cy_) * depth_ep * camera->fy_inv_;

                    const Vec3_t pos_c_sp{unproj_x_sp, unproj_y_sp, depth_sp};
                    const Vec3_t pos_c_ep{unproj_x_ep, unproj_y_ep, depth_ep};

                    // Convert from camera coordinates to world coordinates
                    {
                        std::lock_guard<std::mutex> lock(mtx_pose_);
                        Vec3_t pos_w_sp = cam_pose_wc_.block<3, 3>(0, 0) * pos_c_sp + cam_pose_wc_.block<3, 1>(0, 3);
                        Vec3_t pos_w_ep = cam_pose_wc_.block<3, 3>(0, 0) * pos_c_ep + cam_pose_wc_.block<3, 1>(0, 3);

                        Vec6_t pos_w_line;
                        pos_w_line << pos_w_sp(0), pos_w_sp(1), pos_w_sp(2), pos_w_ep(0), pos_w_ep(1), pos_w_ep(2);

                        return pos_w_line;
                    }
                }
                else
                {
                    return Vec6_t::Zero();
                }
            }

            // triangulate 3D lins using stereo image pair (endpoints found by trimming)
            if (camera_->setup_type_ == camera::setup_type_t::Stereo)
            {
                if (!_good_matches_stereo.count(idx))
                {
                    return Vec6_t::Zero();
                }

                const int queryIdx = _good_matches_stereo.at(idx);

                cv::line_descriptor::KeyLine keyline1 = _keylsd[idx];
                cv::line_descriptor::KeyLine keyline2 = _keylsd_right[queryIdx];

                // the two projection matrix of the stereo image pair, after rectification
                Mat34_t P1, P2;
                P1 << camera->fx_, 0, camera->cx_, 0,
                    0, camera->fy_, camera->cy_, 0,
                    0, 0, 1.0, 0;

                P2 << camera->fx_, 0, camera->cx_, -camera_->focal_x_baseline_,
                    0, camera->fy_, camera->cy_, 0,
                    0, 0, 1.0, 0;

                // construct two planes
                Vec3_t xs_1{keyline1.getStartPoint().x, keyline1.getStartPoint().y, 1.0};
                Vec3_t xe_1{keyline1.getEndPoint().x, keyline1.getEndPoint().y, 1.0};
                Vec3_t line_1 = xs_1.cross(xe_1);
                Vec4_t plane_1 = line_1.transpose() * P1;

                Vec3_t xs_2{keyline2.getStartPoint().x, keyline2.getStartPoint().y, 1.0};
                Vec3_t xe_2{keyline2.getEndPoint().x, keyline2.getEndPoint().y, 1.0};
                Vec3_t line_2 = xs_2.cross(xe_2);
                Vec4_t plane_2 = line_2.transpose() * P2;

                // calculate dual Pluecker matrix via two plane intersection
                Mat44_t L_star = plane_1 * plane_2.transpose() - plane_2 * plane_1.transpose();

                // extract Pluecker coordinates of the 3D line (infinite line representation)
                Mat33_t d_skew = L_star.block<3, 3>(0, 0);
                Vec3_t d;
                d << d_skew(2, 1), d_skew(0, 2), d_skew(1, 0); // the direction vector of the line
                Vec3_t m = L_star.block<3, 1>(0, 3);           // the moment vector of the line

                Vec6_t plucker_coord;
                plucker_coord << m(0), m(1), m(2), d(0), d(1), d(2);

                // endpoints trimming (using keyframe 1)
                Mat66_t transformation_line_cw = Eigen::Matrix<double, 6, 6>::Zero();
                transformation_line_cw.block<3, 3>(0, 0) = Mat33_t::Identity();
                transformation_line_cw.block<3, 3>(3, 3) = Mat33_t::Identity();
                transformation_line_cw.block<3, 3>(0, 3) = Mat33_t::Zero();

                Mat33_t _K;
                _K << camera->fy_, 0.0, 0.0,
                    0.0, camera->fx_, 0.0,
                    -camera->fy_ * camera->cx_, -camera->fx_ * camera->cy_, camera->fx_ * camera->fy_;

                Vec3_t reproj_line_function;
                reproj_line_function = _K * (transformation_line_cw * plucker_coord).block<3, 1>(0, 0);

                double l1 = reproj_line_function(0);
                double l2 = reproj_line_function(1);
                double l3 = reproj_line_function(2);

                // calculate closet point on the re-projected line
                auto sp = keyline1.getStartPoint();
                auto ep = keyline1.getEndPoint();
                double x_sp_closet = -(sp.y - (l2 / l1) * sp.x + (l3 / l2)) * ((l1 * l2) / (l1 * l1 + l2 * l2));
                double y_sp_closet = -(l1 / l2) * x_sp_closet - (l3 / l2);

                double x_ep_closet = -(ep.y - (l2 / l1) * ep.x + (l3 / l2)) * ((l1 * l2) / (l1 * l1 + l2 * l2));
                double y_ep_closet = -(l1 / l2) * x_ep_closet - (l3 / l2);

                // calculate another point
                double x_0sp = 0;
                double y_0sp = sp.y - (l2 / l1) * sp.x;

                double x_0ep = 0;
                double y_0ep = ep.y - (l2 / l1) * ep.x;

                // calculate 3D plane (using keyframe 1)
                Vec3_t point2d_sp_closet{x_sp_closet, y_sp_closet, 1.0};
                Vec3_t point2d_0sp{x_0sp, y_0sp, 1.0};
                Vec3_t line_temp_sp = point2d_sp_closet.cross(point2d_0sp);
                Vec4_t plane3d_temp_sp = P1.transpose() * line_temp_sp;

                Vec3_t point2d_ep_closet{x_ep_closet, y_ep_closet, 1.0};
                Vec3_t point2d_0ep{x_0ep, y_0ep, 1.0};
                Vec3_t line_temp_ep = point2d_ep_closet.cross(point2d_0ep);
                Vec4_t plane3d_temp_ep = P1.transpose() * line_temp_ep;

                // calculate intersection of the 3D plane and 3d line
                Mat44_t line3d_pluecker_matrix = Eigen::Matrix<double, 4, 4>::Zero();
                line3d_pluecker_matrix.block<3, 3>(0, 0) = skew(m);
                line3d_pluecker_matrix.block<3, 1>(0, 3) = d;
                line3d_pluecker_matrix.block<1, 3>(3, 0) = -d.transpose();

                Vec4_t intersect_endpoint_sp, intersect_endpoint_ep;
                intersect_endpoint_sp = line3d_pluecker_matrix * plane3d_temp_sp;
                intersect_endpoint_ep = line3d_pluecker_matrix * plane3d_temp_ep;

                Vec3_t sp_c_3D, ep_c_3D;
                sp_c_3D << intersect_endpoint_sp(0) / intersect_endpoint_sp(3),
                    intersect_endpoint_sp(1) / intersect_endpoint_sp(3),
                    intersect_endpoint_sp(2) / intersect_endpoint_sp(3);

                ep_c_3D << intersect_endpoint_ep(0) / intersect_endpoint_ep(3),
                    intersect_endpoint_ep(1) / intersect_endpoint_ep(3),
                    intersect_endpoint_ep(2) / intersect_endpoint_ep(3);

                // Convert from camera coordinates to world coordinates
                {
                    std::lock_guard<std::mutex> lock(mtx_pose_);
                    Vec3_t pos_w_sp = cam_pose_wc_.block<3, 3>(0, 0) * sp_c_3D + cam_pose_wc_.block<3, 1>(0, 3);
                    Vec3_t pos_w_ep = cam_pose_wc_.block<3, 3>(0, 0) * ep_c_3D + cam_pose_wc_.block<3, 1>(0, 3);

                    Vec6_t pos_w_line;
                    pos_w_line << pos_w_sp(0), pos_w_sp(1), pos_w_sp(2), pos_w_ep(0), pos_w_ep(1), pos_w_ep(2);

                    if (0 < pos_w_sp(2) && 0 < pos_w_ep(2))
                    {
                        return pos_w_line;
                    }
                    else
                    {
                        return Vec6_t::Zero();
                    }
                }
            }

            return Vec6_t::Zero();
        }

        float keyframe::compute_median_depth(const bool abs) const
        { // camera_->model_type_ == camera::model_type_t::Equirectangular -> bool abs
            std::vector<landmark *> landmarks;
            Mat44_t cam_pose_cw;

            {
                std::lock_guard<std::mutex> lock1(mtx_observations_);
                std::lock_guard<std::mutex> lock2(mtx_pose_);
                landmarks = landmarks_;
                cam_pose_cw = cam_pose_cw_;
            }

            std::vector<float> depths;
            depths.reserve(num_keypts_);
            const Vec3_t rot_cw_z_row = cam_pose_cw.block<1, 3>(2, 0); // the 3rd row of rotation matrix
            const float trans_cw_z = cam_pose_cw(2, 3);                // z from the translation matrix

            for (const auto lm : landmarks)
            {
                if (!lm)
                {
                    continue;
                }
                const Vec3_t pos_w = lm->get_pos_in_world();
                // transfrom the landmark from world coordinates to current (keyframe) camera coordinates, and aggregate all the z value
                const auto pos_c_z = rot_cw_z_row.dot(pos_w) + trans_cw_z; // z_c = (R31*X + R32*Y + R33*Z) + t_z
                depths.push_back(abs ? std::abs(pos_c_z) : pos_c_z);
            }

            std::sort(depths.begin(), depths.end());

            return depths.at((depths.size() - 1) / 2);
        }

        void keyframe::set_not_to_be_erased()
        {
            cannot_be_erased_ = true;
        }

        void keyframe::set_to_be_erased()
        {
            if (!graph_node_->has_loop_edge())
            {
                cannot_be_erased_ = false;
            }
        }

        void keyframe::prepare_for_erasing()
        {
            // cannot erase the origin
            if (*this == *(map_db_->origin_keyfrm_))
            {
                return;
            }

            // cannot erase if the frag is raised
            if (cannot_be_erased_)
            {
                return;
            }

            // 1. raise the flag which indicates it has been erased

            will_be_erased_ = true;

            // 2. remove associations between keypoints and landmarks

            for (const auto lm : landmarks_)
            {
                if (!lm)
                {
                    continue;
                }
                lm->erase_observation(this);
            }

            // FW: also remove the associations between keylines and 3D lines
            for (const auto lm_line : _landmarks_line)
            {
                if (!lm_line)
                {
                    continue;
                }
                lm_line->erase_observation(this);
            }

            // 3. recover covisibility graph and spanning tree

            // remove covisibility information
            graph_node_->erase_all_connections();
            // recover spanning tree
            graph_node_->recover_spanning_connections();

            // 3. update frame statistics

            map_db_->replace_reference_keyframe(this, graph_node_->get_spanning_parent());

            // 4. remove myself from the databased

            map_db_->erase_keyframe(this);
            bow_db_->erase_keyframe(this);
        }

        bool keyframe::will_be_erased()
        {
            return will_be_erased_;
        }

        void keyframe::set_segmentation_mask(const cv::Mat &img_seg_mask)
        {
            _img_seg_mask = img_seg_mask;
        }

        cv::Mat keyframe::get_segmentation_mask() const
        {
            return _img_seg_mask;
        }

        void keyframe::set_depth_map(const cv::Mat &depth_img)
        {
            _depth_img = depth_img;
        }

        cv::Mat keyframe::get_depth_map() const
        {
            return _depth_img;
        }

        void keyframe::set_img_rgb(const cv::Mat &img_rgb)
        {
            _img_rgb = img_rgb;
        }

        cv::Mat keyframe::get_img_rgb() const
        {
            return _img_rgb;
        }

        // FW:
        void keyframe::initialize_lsd_scale_information()
        {
            // calculate scale_factors at each level of image pyramid
            std::vector<float> scale_factors(_num_scale_levels_lsd, 1.0);
            if (_num_scale_levels_lsd > 1)
            {
                for (unsigned int level = 1; level < _num_scale_levels_lsd; ++level)
                {
                    scale_factors.at(level) = _scale_factor_lsd * scale_factors.at(level - 1);
                }
                _scale_factors_lsd = scale_factors;
            }
            else
            {
                _scale_factors_lsd = scale_factors;
            }

            // calculate inv_scale_factors
            std::vector<float> inv_scale_factors(_num_scale_levels_lsd, 1.0);
            if (_num_scale_levels_lsd > 1)
            {
                for (unsigned int level = 1; level < _num_scale_levels_lsd; ++level)
                {
                    inv_scale_factors.at(level) = (1.0f / _scale_factor_lsd) * inv_scale_factors.at(level - 1);
                }
                _inv_scale_factors_lsd = inv_scale_factors;
            }
            else
            {
                _inv_scale_factors_lsd = inv_scale_factors;
            }

            // calculate level_sigma_sq
            std::vector<float> level_sigma_sq(_num_scale_levels_lsd, 1.0);
            float scale_factor_at_level = 1.0;
            if (_num_scale_levels_lsd > 1)
            {
                for (unsigned int level = 1; level < _num_scale_levels_lsd; ++level)
                {
                    scale_factor_at_level = _scale_factor_lsd * scale_factor_at_level;
                    level_sigma_sq.at(level) = scale_factor_at_level * scale_factor_at_level;
                }
                _level_sigma_sq_lsd = level_sigma_sq;
            }
            else
            {
                _level_sigma_sq_lsd = level_sigma_sq;
            }

            // calculate inv_level_sigma_sq
            std::vector<float> inv_level_sigma_sq(_num_scale_levels_lsd, 1.0);
            scale_factor_at_level = 1.0;
            if (_num_scale_levels_lsd > 1)
            {
                for (unsigned int level = 1; level < _num_scale_levels_lsd; ++level)
                {
                    scale_factor_at_level = _scale_factor_lsd * scale_factor_at_level;
                    inv_level_sigma_sq.at(level) = 1.0f / (scale_factor_at_level * scale_factor_at_level);
                }
                _inv_level_sigma_sq_lsd = inv_level_sigma_sq;
            }
            else
            {
                _inv_level_sigma_sq_lsd = inv_level_sigma_sq;
            }
        }

    } // namespace data
} // namespace PLPSLAM
