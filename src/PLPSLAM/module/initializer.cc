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

#include "PLPSLAM/config.h"
#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/data/map_database.h"
#include "PLPSLAM/initialize/bearing_vector.h"
#include "PLPSLAM/initialize/perspective.h"
#include "PLPSLAM/match/area.h"
#include "PLPSLAM/module/initializer.h"
#include "PLPSLAM/optimize/global_bundle_adjuster.h"

#include <spdlog/spdlog.h>
#include "PLPSLAM/module/two_view_triangulator_line.h"

namespace PLPSLAM
{
    namespace module
    {

        initializer::initializer(const camera::setup_type_t setup_type,
                                 data::map_database *map_db, data::bow_database *bow_db,
                                 const YAML::Node &yaml_node)
            : setup_type_(setup_type), map_db_(map_db), bow_db_(bow_db),
              num_ransac_iters_(yaml_node["Initializer.num_ransac_iterations"].as<unsigned int>(100)),
              min_num_triangulated_(yaml_node["Initializer.num_min_triangulated_pts"].as<unsigned int>(50)),
              parallax_deg_thr_(yaml_node["Initializer.parallax_deg_threshold"].as<float>(1.0)),
              reproj_err_thr_(yaml_node["Initializer.reprojection_error_threshold"].as<float>(4.0)),
              num_ba_iters_(yaml_node["Initializer.num_ba_iterations"].as<unsigned int>(20)),
              scaling_factor_(yaml_node["Initializer.scaling_factor"].as<float>(1.0))
        {
            spdlog::debug("CONSTRUCT: module::initializer");
        }

        initializer::~initializer()
        {
            spdlog::debug("DESTRUCT: module::initializer");
        }

        void initializer::reset()
        {
            initializer_base_.reset(nullptr);
            state_ = initializer_state_t::NotReady;
            init_frm_id_ = 0;
        }

        initializer_state_t initializer::get_state() const
        {
            return state_;
        }

        std::vector<cv::KeyPoint> initializer::get_initial_keypoints() const
        {
            return init_frm_.keypts_;
        }

        std::vector<int> initializer::get_initial_matches() const
        {
            return init_matches_;
        }

        unsigned int initializer::get_initial_frame_id() const
        {
            return init_frm_id_;
        }

        bool initializer::initialize(data::frame &curr_frm)
        {

            switch (setup_type_)
            {
            case camera::setup_type_t::Monocular:
            {
                // construct an initializer if not constructed
                if (state_ == initializer_state_t::NotReady)
                {
                    create_initializer(curr_frm);
                    return false;
                }

                // try to initialize
                if (!try_initialize_for_monocular(curr_frm))
                {
                    // failed
                    return false;
                }

                // create new map if succeeded, which means we found first two image to initialize the map
                create_map_for_monocular(curr_frm);
                break;
            }
            case camera::setup_type_t::Stereo:
            case camera::setup_type_t::RGBD:
            {
                state_ = initializer_state_t::Initializing;

                // try to initialize
                if (!try_initialize_for_stereo(curr_frm))
                {
                    // failed
                    return false;
                }

                // create new map if succeeded
                create_map_for_stereo(curr_frm);
                break;
            }
            default:
            {
                throw std::runtime_error("Undefined camera setup");
            }
            }

            // check the state is succeeded or not
            if (state_ == initializer_state_t::Succeeded)
            {
                init_frm_id_ = curr_frm.id_;
                return true;
            }
            else
            {
                return false;
            }
        }

        void initializer::create_initializer(data::frame &curr_frm)
        {
            // set the initial frame
            init_frm_ = data::frame(curr_frm);

            // initialize the previously matched coordinates
            prev_matched_coords_.resize(init_frm_.undist_keypts_.size()); // std::vector<cv::Point2f>
            for (unsigned int i = 0; i < init_frm_.undist_keypts_.size(); ++i)
            {
                prev_matched_coords_.at(i) = init_frm_.undist_keypts_.at(i).pt;
            }

            // initialize matchings (init_idx -> curr_idx)
            std::fill(init_matches_.begin(), init_matches_.end(), -1);

            // build a initializer
            initializer_base_.reset(nullptr); //  std::unique_ptr<initialize::base>
            switch (init_frm_.camera_->model_type_)
            {
            case camera::model_type_t::Perspective:
            case camera::model_type_t::Fisheye:
            {
                initializer_base_ = std::unique_ptr<initialize::perspective>(new initialize::perspective(init_frm_,
                                                                                                         num_ransac_iters_, min_num_triangulated_,
                                                                                                         parallax_deg_thr_, reproj_err_thr_));
                break;
            }
            case camera::model_type_t::Equirectangular:
            {
                initializer_base_ = std::unique_ptr<initialize::bearing_vector>(new initialize::bearing_vector(init_frm_,
                                                                                                               num_ransac_iters_, min_num_triangulated_,
                                                                                                               parallax_deg_thr_, reproj_err_thr_));
                break;
            }
            }

            state_ = initializer_state_t::Initializing;
        }

        bool initializer::try_initialize_for_monocular(data::frame &curr_frm)
        {
            assert(state_ == initializer_state_t::Initializing);

            // find initial matches between init_frm_ and curr_frm -> init_matches_
            match::area matcher(0.9, true); // FW: true -> check orientation
            unsigned int num_matches = matcher.match_in_consistent_area(init_frm_, curr_frm, prev_matched_coords_, init_matches_, 100);

            if (num_matches < min_num_triangulated_)
            {
                // rebuild the initializer with the next frame
                reset();
                return false;
            }

            // try to initialize with the current frame
            // FW: see -> perspective::initialize() -> initialize system with H/F
            // with initial triangulation of points, see -> base::check_pose() -> triangulator::triangulate()
            assert(initializer_base_);
            return initializer_base_->initialize(curr_frm, init_matches_);
        }

        bool initializer::create_map_for_monocular(data::frame &curr_frm)
        {
            assert(state_ == initializer_state_t::Initializing);

            eigen_alloc_vector<Vec3_t> init_triangulated_pts;
            {
                assert(initializer_base_);
                init_triangulated_pts = initializer_base_->get_triangulated_pts();
                const auto is_triangulated = initializer_base_->get_triangulated_flags();

                // make invalid the matchings which have not been triangulated
                for (unsigned int i = 0; i < init_matches_.size(); ++i)
                {
                    if (init_matches_.at(i) < 0)
                    {
                        continue;
                    }
                    if (is_triangulated.at(i))
                    {
                        continue;
                    }
                    init_matches_.at(i) = -1;
                }

                // set the camera poses
                init_frm_.set_cam_pose(Mat44_t::Identity());
                Mat44_t cam_pose_cw = Mat44_t::Identity();
                cam_pose_cw.block<3, 3>(0, 0) = initializer_base_->get_rotation_ref_to_cur();
                cam_pose_cw.block<3, 1>(0, 3) = initializer_base_->get_translation_ref_to_cur();
                curr_frm.set_cam_pose(cam_pose_cw);

                // destruct the initializer
                initializer_base_.reset(nullptr);
            }

            // create initial keyframes
            auto init_keyfrm = new data::keyframe(init_frm_, map_db_, bow_db_);
            auto curr_keyfrm = new data::keyframe(curr_frm, map_db_, bow_db_);

            // FW: also pass the corresponding segmentation mask
            if (map_db_->_b_seg_or_not)
            {
                init_keyfrm->set_segmentation_mask(init_frm_._img_seg_mask);
                curr_keyfrm->set_segmentation_mask(curr_frm._img_seg_mask);
            }

            // compute BoW representations
            init_keyfrm->compute_bow();
            curr_keyfrm->compute_bow();

            // add the keyframes to the map DB
            map_db_->add_keyframe(init_keyfrm);
            map_db_->add_keyframe(curr_keyfrm);

            // update the frame statistics
            init_frm_.ref_keyfrm_ = init_keyfrm;
            curr_frm.ref_keyfrm_ = curr_keyfrm;
            map_db_->update_frame_statistics(init_frm_, false);
            map_db_->update_frame_statistics(curr_frm, false);

            // assign 2D-3D associations
            for (unsigned int init_idx = 0; init_idx < init_matches_.size(); init_idx++)
            {
                const auto curr_idx = init_matches_.at(init_idx);
                if (curr_idx < 0)
                {
                    continue;
                }

                // construct a landmark: point
                auto lm = new data::landmark(init_triangulated_pts.at(init_idx), curr_keyfrm, map_db_);

                // set the assocications to the new keyframes
                init_keyfrm->add_landmark(lm, init_idx);
                curr_keyfrm->add_landmark(lm, curr_idx);
                lm->add_observation(init_keyfrm, init_idx);
                lm->add_observation(curr_keyfrm, curr_idx);

                // update the descriptor
                lm->compute_descriptor();
                // update the geometry
                lm->update_normal_and_depth();

                // set the 2D-3D assocications to the current frame
                curr_frm.landmarks_.at(curr_idx) = lm;
                curr_frm.outlier_flags_.at(curr_idx) = false;

                // add the landmark to the map DB
                map_db_->add_landmark(lm);
            }

            // FW: triangulate lines in monocular initialization
            if (map_db_->_b_use_line_tracking)
            {
                triangulate_line_with_two_keyframes(curr_keyfrm, init_keyfrm, curr_frm);
            }

            // global bundle adjustment
            const auto global_bundle_adjuster = optimize::global_bundle_adjuster(map_db_, num_ba_iters_, true);
            global_bundle_adjuster.optimize();

            // scale the map so that the median of depths is 1.0
            const auto median_depth = init_keyfrm->compute_median_depth(init_keyfrm->camera_->model_type_ == camera::model_type_t::Equirectangular);
            const auto inv_median_depth = 1.0 / median_depth;
            if (curr_keyfrm->get_num_tracked_landmarks(1) < min_num_triangulated_ && median_depth < 0)
            {
                spdlog::info("seems to be wrong initialization, resetting");
                state_ = initializer_state_t::Wrong;
                return false;
            }
            scale_map(init_keyfrm, curr_keyfrm, inv_median_depth * scaling_factor_);

            // FW: detect plane after Global BA, after monocular initialization
            if (map_db_->_b_seg_or_not)
            {
                spdlog::info("-- Initializer[Mono] -- trying to initialize plane");

                const auto planar_mapper = new Planar_Mapping_module(map_db_, true);
                planar_mapper->initialize_map_with_plane(init_keyfrm);
                planar_mapper->initialize_map_with_plane(curr_keyfrm);

                delete planar_mapper;
            }

            // update the current frame pose
            curr_frm.set_cam_pose(curr_keyfrm->get_cam_pose());

            // set the origin keyframe
            map_db_->origin_keyfrm_ = init_keyfrm;

            spdlog::info("new map created with {} points: frame {} - frame {}", map_db_->get_num_landmarks(), init_frm_.id_, curr_frm.id_);
            state_ = initializer_state_t::Succeeded;

            // FW:
            if (map_db_->_b_use_line_tracking)
            {
                spdlog::info("new map created with {} lines: frame {} - frame {}", map_db_->get_num_landmarks_line(), init_frm_.id_, curr_frm.id_);
            }

            // FW: experiments -> save map point clouds into a txt file: each line indicates x, y, z
            //  {
            //      auto all_map_points = map_db_->get_all_landmarks();
            //      std::string path = "/home/shu/pytorch_ex/pointplaneslam/evaluation/pointclouds_mono.txt";
            //      FILE *pFile;
            //      pFile = fopen(path.c_str(), "w");
            //      for (auto pt : all_map_points)
            //      {
            //          auto position = pt->get_pos_in_world();
            //          fprintf(pFile, "%lf %lf %lf \n",
            //                  position(0), position(1), position(2));
            //      }

            //     fclose(pFile);
            // }

            return true;
        }

        void initializer::scale_map(data::keyframe *init_keyfrm, data::keyframe *curr_keyfrm, const double scale)
        {
            // scaling keyframes
            Mat44_t cam_pose_cw = curr_keyfrm->get_cam_pose();
            cam_pose_cw.block<3, 1>(0, 3) *= scale;
            curr_keyfrm->set_cam_pose(cam_pose_cw);

            // scaling landmarks
            const auto landmarks = init_keyfrm->get_landmarks();
            for (auto lm : landmarks)
            {
                if (!lm)
                {
                    continue;
                }
                lm->set_pos_in_world(lm->get_pos_in_world() * scale);
            }

            // FW: scaling the landmarks line
            if (map_db_->_b_use_line_tracking && init_keyfrm->get_num_tracked_landmarks(1) > 0)
            {
                const auto landmarks_line = init_keyfrm->get_landmarks_line();
                if (!landmarks_line.empty())
                {
                    for (auto lm_line : landmarks_line)
                    {
                        if (!lm_line)
                        {
                            continue;
                        }
                        lm_line->set_pos_in_world(lm_line->get_pos_in_world() * scale);
                    }
                }
            }
        }

        bool initializer::try_initialize_for_stereo(data::frame &curr_frm)
        {
            assert(state_ == initializer_state_t::Initializing);
            // count the number of valid depths
            unsigned int num_valid_depths = std::count_if(curr_frm.depths_.begin(), curr_frm.depths_.end(),
                                                          [](const float depth)
                                                          {
                                                              return 0 < depth;
                                                          });
            return min_num_triangulated_ <= num_valid_depths;
        }

        bool initializer::create_map_for_stereo(data::frame &curr_frm)
        {
            assert(state_ == initializer_state_t::Initializing);

            // create an initial keyframe
            curr_frm.set_cam_pose(Mat44_t::Identity());
            auto curr_keyfrm = new data::keyframe(curr_frm, map_db_, bow_db_);

            // FW: also pass the corresponding segmentation mask, depth image, and rgb image
            if (map_db_->_b_seg_or_not)
            {
                curr_keyfrm->set_segmentation_mask(curr_frm._img_seg_mask); // for planar mapping
                curr_keyfrm->set_depth_map(curr_frm._depth_img);            // for dense reconstruction
            }

            // FW:
            curr_keyfrm->set_img_rgb(curr_frm._img_rgb); // for assigning rgb color to the dense point cloud

            // compute BoW representation
            curr_keyfrm->compute_bow();

            // add to the map DB
            map_db_->add_keyframe(curr_keyfrm);

            // update the frame statistics
            curr_frm.ref_keyfrm_ = curr_keyfrm;
            map_db_->update_frame_statistics(curr_frm, false);

            for (unsigned int idx = 0; idx < curr_frm.num_keypts_; ++idx)
            {
                // add a new landmark if tht corresponding depth is valid
                const auto z = curr_frm.depths_.at(idx);
                if (z <= 0)
                {
                    continue;
                }

                // build a landmark
                const Vec3_t pos_w = curr_frm.triangulate_stereo(idx);
                auto lm = new data::landmark(pos_w, curr_keyfrm, map_db_);

                // set the associations to the new keyframe
                lm->add_observation(curr_keyfrm, idx);
                curr_keyfrm->add_landmark(lm, idx);

                // update the descriptor
                lm->compute_descriptor();
                // update the geometry
                lm->update_normal_and_depth();

                // set the 2D-3D associations to the current frame
                curr_frm.landmarks_.at(idx) = lm;
                curr_frm.outlier_flags_.at(idx) = false;

                // add the landmark to the map DB
                map_db_->add_landmark(lm);
            }

            // FW: (RGB-D/Stereo) create initial landmarks for 3D lines
            if (map_db_->_b_use_line_tracking)
            {
                if (setup_type_ == camera::setup_type_t::RGBD)
                {
                    for (unsigned int idx_l = 0; idx_l < curr_frm._num_keylines; ++idx_l)
                    {
                        const auto z_sp = curr_frm._depths_cooresponding_to_keylines.at(idx_l).first;
                        const auto z_ep = curr_frm._depths_cooresponding_to_keylines.at(idx_l).second;
                        if (z_sp <= 0 || z_ep <= 0)
                        {
                            continue;
                        }

                        // build 3D line via depth
                        const Vec6_t pos_w_line = curr_frm.triangulate_stereo_for_line(idx_l);
                        auto lm_line = new data::Line(pos_w_line, curr_keyfrm, map_db_);

                        // set the associations to the new keyframe
                        lm_line->add_observation(curr_keyfrm, idx_l);
                        curr_keyfrm->add_landmark_line(lm_line, idx_l);

                        // update the descriptor
                        lm_line->compute_descriptor();
                        // update the geometry
                        lm_line->update_information();

                        // set the 2D-3D associations to the current frame
                        curr_frm._landmarks_line.at(idx_l) = lm_line;
                        curr_frm._outlier_flags_line.at(idx_l) = false;

                        // add the landmark to the map DB
                        map_db_->add_landmark_line(lm_line);
                    }
                }

                if (setup_type_ == camera::setup_type_t::Stereo)
                {
                    for (unsigned int idx_l = 0; idx_l < curr_frm._num_keylines; ++idx_l)
                    {
                        // build 3D line via two-view triangulation and endpoints trimming
                        const Vec6_t pos_w_line = curr_frm.triangulate_stereo_for_line(idx_l);
                        auto lm_line = new data::Line(pos_w_line, curr_keyfrm, map_db_);

                        // set the associations to the new keyframe
                        lm_line->add_observation(curr_keyfrm, idx_l);
                        curr_keyfrm->add_landmark_line(lm_line, idx_l);

                        // update the descriptor
                        lm_line->compute_descriptor();
                        // update the geometry
                        lm_line->update_information();

                        // set the 2D-3D associations to the current frame
                        curr_frm._landmarks_line.at(idx_l) = lm_line;
                        curr_frm._outlier_flags_line.at(idx_l) = false;

                        // add the landmark to the map DB
                        map_db_->add_landmark_line(lm_line);
                    }
                }
            }

            // FW: detect plane after RGB-D initialization
            if (map_db_->_b_seg_or_not)
            {
                spdlog::info("-- Initializer[RGB-D&Stereo] -- trying to initialize plane");

                const auto planar_mapper = new Planar_Mapping_module(map_db_, false);
                planar_mapper->initialize_map_with_plane(curr_keyfrm);

                delete planar_mapper;
            }

            // set the origin keyframe
            map_db_->origin_keyfrm_ = curr_keyfrm;

            spdlog::info("new map created with {} points: frame {}", map_db_->get_num_landmarks(), curr_frm.id_);

            // FW:
            if (map_db_->_b_seg_or_not)
            {
                spdlog::info("new map created with {} planes: frame {}", map_db_->get_num_landmark_planes(), curr_frm.id_);
            }

            // FW:
            if (map_db_->_b_use_line_tracking)
            {
                spdlog::info("new map created with {} lines: frame {}", map_db_->get_num_landmarks_line(), curr_frm.id_);
            }

            state_ = initializer_state_t::Succeeded;

            // FW: experiments -> save map point clouds into a txt file: each line indicates x, y, z
            //  {
            //      auto all_map_points = map_db_->get_all_landmarks();
            //      std::string path = "/home/shu/pytorch_ex/pointplaneslam/evaluation/pointclouds_rgbd.txt";
            //      FILE *pFile;
            //      pFile = fopen(path.c_str(), "w");
            //      for (auto pt : all_map_points)
            //      {
            //          auto position = pt->get_pos_in_world();
            //          fprintf(pFile, "%lf %lf %lf \n",
            //                  position(0), position(1), position(2));
            //      }

            //     fclose(pFile);
            // }

            return true;
        }

        // FW:
        void initializer::triangulate_line_with_two_keyframes(data::keyframe *cur_keyfrm, data::keyframe *ngh_keyfrm, data::frame &curr_frm)
        {
            if (cur_keyfrm->_keylsd.size() == 0 || ngh_keyfrm->_keylsd.size() == 0)
            {
                return;
            }

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
                if (lsd_matches[j].distance < 30)
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

                    // select good matches, we give bigger thresholds as we match across keyframes
                    if (distance_s < 200 && distance_e < 200 && angle < 5)
                    {
                        good_matches.push_back(lsd_matches[j]);
                    }
                }
            }

            // initialize the two view triangulator for line
            const module::two_view_triangulator_line triangulator_line(cur_keyfrm, ngh_keyfrm, 1.0);
            for (unsigned k = 0; k < good_matches.size(); k++)
            {
                // variables should be all in Eigen, not in cv::Mat
                // get matched line segments
                cv::DMatch mt = good_matches[k];

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

                // set assocications in the curr_frame
                curr_frm._landmarks_line.at(mt.queryIdx) = lm_line;
                curr_frm._outlier_flags_line.at(mt.queryIdx) = false;

                // add landmark to the map database
                map_db_->add_landmark_line(lm_line);
            }
        }
    } // namespace module
} // namespace PLPSLAM
