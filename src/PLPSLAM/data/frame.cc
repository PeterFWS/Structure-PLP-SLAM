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
#include "PLPSLAM/feature/orb_extractor.h"
#include "PLPSLAM/match/stereo.h"

#include <thread>

#include <spdlog/spdlog.h>

// FW:
#include "PLPSLAM/feature/line_extractor.h"

namespace PLPSLAM
{
    namespace data
    {

        std::atomic<unsigned int> frame::next_id_{0};

        // (default) Monocular
        frame::frame(const cv::Mat &img_gray, const double timestamp,
                     feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                     camera::base *camera, const float depth_thr,
                     const cv::Mat &mask)
            : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor), extractor_right_(nullptr),
              timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr)
        {
            // Get ORB scale
            update_orb_info();

            // Extract ORB feature
            extract_orb(img_gray, mask);
            num_keypts_ = keypts_.size();
            if (keypts_.empty())
            {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Ignore stereo parameters
            stereo_x_right_ = std::vector<float>(num_keypts_, -1);
            depths_ = std::vector<float>(num_keypts_, -1);

            // FW: Ignore stereo parameters of line feature
            _stereo_x_right_cooresponding_to_keylines = std::vector<std::pair<float, float>>(num_keypts_, std::make_pair(-1, -1));
            _depths_cooresponding_to_keylines = std::vector<std::pair<float, float>>(num_keypts_, std::make_pair(-1, -1));

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
        }

        // FW: Monocular + planar segmentation
        frame::frame(const cv::Mat &img_gray, const double timestamp, const cv::Mat &img_seg_mask,
                     feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                     camera::base *camera, const float depth_thr,
                     const cv::Mat &mask)
            : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor), extractor_right_(nullptr),
              timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr),
              _img_seg_mask(img_seg_mask)
        {
            // Get ORB scale
            update_orb_info();

            // Extract ORB feature
            extract_orb(img_gray, mask);
            num_keypts_ = keypts_.size();
            if (keypts_.empty())
            {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Ignore stereo parameters
            stereo_x_right_ = std::vector<float>(num_keypts_, -1);
            depths_ = std::vector<float>(num_keypts_, -1);

            // FW: Ignore stereo parameters of line feature
            _stereo_x_right_cooresponding_to_keylines = std::vector<std::pair<float, float>>(num_keypts_, std::make_pair(-1, -1));
            _depths_cooresponding_to_keylines = std::vector<std::pair<float, float>>(num_keypts_, std::make_pair(-1, -1));

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
        }

        // FW: Monocular frame + planar segmentation + line segments extractor
        frame::frame(const cv::Mat &img_gray, const double timestamp, const cv::Mat &img_seg_mask,
                     feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                     feature::LineFeatureTracker *line_extractor,
                     camera::base *camera, const float depth_thr,
                     const cv::Mat &mask)
            : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor), extractor_right_(nullptr),
              timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr),
              _img_seg_mask(img_seg_mask),
              _line_extractor(line_extractor)
        {
            // Get ORB scale
            update_orb_info();

            // FW: extract ORB feature and Line features in parallel
            std::thread thread_for_Point(&frame::extract_orb, this, img_gray, mask, image_side::Left);
            std::thread thread_for_Line(&frame::extract_line, this, img_gray, std::ref(_keylsd), std::ref(_lbd_descr), std::ref(_keyline_functions));
            thread_for_Point.join();
            thread_for_Line.join();

            num_keypts_ = keypts_.size();
            if (keypts_.empty())
            {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Ignore stereo parameters
            stereo_x_right_ = std::vector<float>(num_keypts_, -1);
            depths_ = std::vector<float>(num_keypts_, -1);

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);

            // FW:
            if (!_keylsd.empty() && _line_extractor)
            {
                update_lsd_info();

                _num_keylines = _keylsd.size();

                // Ignore stereo parameters of line feature
                _stereo_x_right_cooresponding_to_keylines = std::vector<std::pair<float, float>>(_num_keylines, std::make_pair(-1, -1));
                _depths_cooresponding_to_keylines = std::vector<std::pair<float, float>>(_num_keylines, std::make_pair(-1, -1));

                // initialize association with 3D lines
                _landmarks_line = std::vector<Line *>(_num_keylines, nullptr);
                _outlier_flags_line = std::vector<bool>(_num_keylines, false);
            }
        }

        // FW: Monocular frame + line segments extractor
        frame::frame(const cv::Mat &img_gray, const double timestamp,
                     feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                     feature::LineFeatureTracker *line_extractor,
                     camera::base *camera, const float depth_thr,
                     const cv::Mat &mask)
            : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor), extractor_right_(nullptr),
              timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr),
              _line_extractor(line_extractor)
        {
            // Get ORB scale
            update_orb_info();

            // FW: extract ORB feature and Line features in parallel
            std::thread thread_for_Point(&frame::extract_orb, this, img_gray, mask, image_side::Left);
            std::thread thread_for_Line(&frame::extract_line, this, img_gray, std::ref(_keylsd), std::ref(_lbd_descr), std::ref(_keyline_functions));
            thread_for_Point.join();
            thread_for_Line.join();

            num_keypts_ = keypts_.size();
            if (keypts_.empty())
            {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Ignore stereo parameters
            stereo_x_right_ = std::vector<float>(num_keypts_, -1);
            depths_ = std::vector<float>(num_keypts_, -1);

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);

            // FW:
            if (!_keylsd.empty() && _line_extractor)
            {
                update_lsd_info();

                _num_keylines = _keylsd.size();

                // Ignore stereo parameters of line feature
                _stereo_x_right_cooresponding_to_keylines = std::vector<std::pair<float, float>>(_num_keylines, std::make_pair(-1, -1));
                _depths_cooresponding_to_keylines = std::vector<std::pair<float, float>>(_num_keylines, std::make_pair(-1, -1));

                // initialize association with 3D lines
                _landmarks_line = std::vector<Line *>(_num_keylines, nullptr);
                _outlier_flags_line = std::vector<bool>(_num_keylines, false);
            }
        }

        //-----------------------------------------
        // (default) Stereo
        frame::frame(const cv::Mat &left_img_gray, const cv::Mat &right_img_gray, const double timestamp,
                     feature::orb_extractor *extractor_left, feature::orb_extractor *extractor_right,
                     bow_vocabulary *bow_vocab, camera::base *camera, const float depth_thr,
                     const cv::Mat &mask)
            : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor_left), extractor_right_(extractor_right),
              timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr)
        {
            // Get ORB scale
            update_orb_info();

            // Extract ORB feature
            std::thread thread_left(&frame::extract_orb, this, left_img_gray, mask, image_side::Left);
            std::thread thread_right(&frame::extract_orb, this, right_img_gray, mask, image_side::Right);
            thread_left.join();
            thread_right.join();
            num_keypts_ = keypts_.size();
            if (keypts_.empty())
            {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Estimate depth with stereo match
            match::stereo stereo_matcher(extractor_left->image_pyramid_, extractor_right_->image_pyramid_,
                                         keypts_, keypts_right_, descriptors_, descriptors_right_,
                                         scale_factors_, inv_scale_factors_,
                                         camera->focal_x_baseline_, camera_->true_baseline_);
            stereo_matcher.compute(stereo_x_right_, depths_);

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
        }

        // FW: stereo + planar segmentation
        frame::frame(const cv::Mat &left_img_gray, const cv::Mat &right_img_gray, const cv::Mat &img_seg_mask, const double timestamp,
                     feature::orb_extractor *extractor_left, feature::orb_extractor *extractor_right, bow_vocabulary *bow_vocab,
                     camera::base *camera, const float depth_thr,
                     const cv::Mat &mask)
            : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor_left), extractor_right_(extractor_right),
              timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr), _img_seg_mask(img_seg_mask)
        {
            // Get ORB scale
            update_orb_info();

            // Extract ORB feature
            std::thread thread_left(&frame::extract_orb, this, left_img_gray, mask, image_side::Left);
            std::thread thread_right(&frame::extract_orb, this, right_img_gray, mask, image_side::Right);
            thread_left.join();
            thread_right.join();
            num_keypts_ = keypts_.size();
            if (keypts_.empty())
            {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Estimate depth with stereo match
            match::stereo stereo_matcher(extractor_left->image_pyramid_, extractor_right_->image_pyramid_,
                                         keypts_, keypts_right_, descriptors_, descriptors_right_,
                                         scale_factors_, inv_scale_factors_,
                                         camera->focal_x_baseline_, camera_->true_baseline_);
            stereo_matcher.compute(stereo_x_right_, depths_);

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
        }

        // FW: stereo + planar segmentation + line segments extractor
        frame::frame(const cv::Mat &left_img_gray, const cv::Mat &right_img_gray, const cv::Mat &img_seg_mask, const double timestamp,
                     feature::orb_extractor *extractor_left, feature::orb_extractor *extractor_right, bow_vocabulary *bow_vocab,
                     feature::LineFeatureTracker *line_extractor, feature::LineFeatureTracker *line_extractor_right,
                     camera::base *camera, const float depth_thr,
                     const cv::Mat &mask)
            : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor_left), extractor_right_(extractor_right),
              _line_extractor(line_extractor), _line_extractor_right(line_extractor_right),
              timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr), _img_seg_mask(img_seg_mask)
        {
            // Get ORB scale
            update_orb_info();

            // FW: extract ORB feature and Line features in parallel
            std::thread thread_left(&frame::extract_orb, this, left_img_gray, mask, image_side::Left);
            std::thread thread_right(&frame::extract_orb, this, right_img_gray, mask, image_side::Right);
            std::thread thread_for_Line_left(&frame::extract_line_stereo, this, left_img_gray, std::ref(_keylsd), std::ref(_lbd_descr), std::ref(_keyline_functions), image_side::Left);
            std::thread thread_for_Line_right(&frame::extract_line_stereo, this, right_img_gray, std::ref(_keylsd_right), std::ref(_lbd_descr_right), std::ref(_keyline_functions_right), image_side::Right);
            thread_left.join();
            thread_right.join();
            thread_for_Line_left.join();
            thread_for_Line_right.join();

            num_keypts_ = keypts_.size();
            if (keypts_.empty())
            {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Estimate depth with stereo match
            match::stereo stereo_matcher(extractor_left->image_pyramid_, extractor_right_->image_pyramid_,
                                         keypts_, keypts_right_, descriptors_, descriptors_right_,
                                         scale_factors_, inv_scale_factors_,
                                         camera->focal_x_baseline_, camera_->true_baseline_);
            stereo_matcher.compute(stereo_x_right_, depths_);

            // FW: estimate depth according to the line matches between stereo image pair
            if (!_keylsd.empty() && _line_extractor)
            {
                update_lsd_info();

                // initialize association with 3D lines
                _num_keylines = _keylsd.size();
                _landmarks_line = std::vector<Line *>(_num_keylines, nullptr);
                _outlier_flags_line = std::vector<bool>(_num_keylines, false);
                _depths_cooresponding_to_keylines = std::vector<std::pair<float, float>>(_num_keylines, std::make_pair(-1, -1));
                _stereo_x_right_cooresponding_to_keylines = std::vector<std::pair<float, float>>(_num_keylines, std::make_pair(-1, -1));

                // match lines from stereo image pair:
                // if a line has a good match -> assign _stereo_x_right_cooresponding_to_keylines > 0
                // later on this line will be checked if using triangulation from stereo image pair
                std::vector<cv::DMatch> lsd_matches;
                cv::Ptr<cv::line_descriptor::BinaryDescriptorMatcher> binary_descriptor_matcher;
                binary_descriptor_matcher = cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

                // left image (query), right (train)
                binary_descriptor_matcher->match(_lbd_descr, _lbd_descr_right, lsd_matches);

                // select best matches
                std::vector<cv::line_descriptor::KeyLine> good_Keylines;
                for (unsigned int j = 0; j < lsd_matches.size(); j++)
                {
                    if (lsd_matches[j].distance < 30)
                    {
                        cv::DMatch mt = lsd_matches[j];
                        cv::line_descriptor::KeyLine line1 = _keylsd[mt.queryIdx];
                        cv::line_descriptor::KeyLine line2 = _keylsd_right[mt.trainIdx];

                        // check the distance
                        cv::Point2f serr = line1.getStartPoint() - line2.getStartPoint();
                        cv::Point2f eerr = line1.getEndPoint() - line2.getEndPoint();
                        const float distance_s = sqrt(serr.dot(serr));
                        const float distance_e = sqrt(eerr.dot(eerr));

                        // check the angle
                        const float angle = abs((abs(line1.angle) - abs(line2.angle))) * 180 / 3.14;

                        // select good matches
                        if (distance_s < 200 && distance_e < 200 && angle < 5)
                        {
                            _good_matches_stereo[mt.queryIdx] = mt.trainIdx;

                            // update stereo information
                            // here, we simply assign a value > 0.0 as an indicator. This value will not be used for triangulation.
                            _depths_cooresponding_to_keylines.at(mt.queryIdx) = std::make_pair(1.0, 1.0);
                            _stereo_x_right_cooresponding_to_keylines.at(mt.queryIdx) = std::make_pair(1.0, 1.0);
                        }
                    }
                }
            }

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
        }

        // FW: Stereo frame + line segment extractor
        frame::frame(const cv::Mat &left_img_gray, const cv::Mat &right_img_gray, const double timestamp,
                     feature::orb_extractor *extractor_left, feature::orb_extractor *extractor_right, bow_vocabulary *bow_vocab,
                     feature::LineFeatureTracker *line_extractor, feature::LineFeatureTracker *line_extractor_right,
                     camera::base *camera, const float depth_thr,
                     const cv::Mat &mask)
            : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor_left), extractor_right_(extractor_right),
              _line_extractor(line_extractor), _line_extractor_right(line_extractor_right),
              timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr)
        {
            // Get ORB scale
            update_orb_info();

            // FW: extract ORB feature and Line features in parallel
            std::thread thread_left(&frame::extract_orb, this, left_img_gray, mask, image_side::Left);
            std::thread thread_right(&frame::extract_orb, this, right_img_gray, mask, image_side::Right);
            std::thread thread_for_Line_left(&frame::extract_line_stereo, this, left_img_gray, std::ref(_keylsd), std::ref(_lbd_descr), std::ref(_keyline_functions), image_side::Left);
            std::thread thread_for_Line_right(&frame::extract_line_stereo, this, right_img_gray, std::ref(_keylsd_right), std::ref(_lbd_descr_right), std::ref(_keyline_functions_right), image_side::Right);
            thread_left.join();
            thread_right.join();
            thread_for_Line_left.join();
            thread_for_Line_right.join();

            num_keypts_ = keypts_.size();
            if (keypts_.empty())
            {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Estimate depth with stereo match
            match::stereo stereo_matcher(extractor_left->image_pyramid_, extractor_right_->image_pyramid_,
                                         keypts_, keypts_right_, descriptors_, descriptors_right_,
                                         scale_factors_, inv_scale_factors_,
                                         camera->focal_x_baseline_, camera_->true_baseline_);
            stereo_matcher.compute(stereo_x_right_, depths_);

            // FW: estimate depth according to the line matches between stereo image pair
            if (!_keylsd.empty() && _line_extractor)
            {
                update_lsd_info();

                // initialize association with 3D lines
                _num_keylines = _keylsd.size();
                _landmarks_line = std::vector<Line *>(_num_keylines, nullptr);
                _outlier_flags_line = std::vector<bool>(_num_keylines, false);
                _depths_cooresponding_to_keylines = std::vector<std::pair<float, float>>(_num_keylines, std::make_pair(-1, -1));
                _stereo_x_right_cooresponding_to_keylines = std::vector<std::pair<float, float>>(_num_keylines, std::make_pair(-1, -1));

                // match lines from stereo image pair:
                // if a line has a good match -> assign _stereo_x_right_cooresponding_to_keylines > 0
                // later on this line will be checked if using triangulation from stereo image pair
                std::vector<cv::DMatch> lsd_matches;
                cv::Ptr<cv::line_descriptor::BinaryDescriptorMatcher> binary_descriptor_matcher;
                binary_descriptor_matcher = cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

                // left image (query), right (train)
                binary_descriptor_matcher->match(_lbd_descr, _lbd_descr_right, lsd_matches);

                // select best matches
                std::vector<cv::line_descriptor::KeyLine> good_Keylines;
                for (unsigned int j = 0; j < lsd_matches.size(); j++)
                {
                    if (lsd_matches[j].distance < 30)
                    {
                        cv::DMatch mt = lsd_matches[j];
                        cv::line_descriptor::KeyLine line1 = _keylsd[mt.queryIdx];
                        cv::line_descriptor::KeyLine line2 = _keylsd_right[mt.trainIdx];

                        // check the distance
                        cv::Point2f serr = line1.getStartPoint() - line2.getStartPoint();
                        cv::Point2f eerr = line1.getEndPoint() - line2.getEndPoint();
                        const float distance_s = sqrt(serr.dot(serr));
                        const float distance_e = sqrt(eerr.dot(eerr));

                        // check the angle
                        const float angle = abs((abs(line1.angle) - abs(line2.angle))) * 180 / 3.14;

                        // select good matches
                        if (distance_s < 200 && distance_e < 200 && angle < 5)
                        {
                            _good_matches_stereo[mt.queryIdx] = mt.trainIdx;

                            // update stereo information
                            // here, we simply assign a value > 0.0 as an indicator. This value will not be used for triangulation.
                            _depths_cooresponding_to_keylines.at(mt.queryIdx) = std::make_pair(1.0, 1.0);
                            _stereo_x_right_cooresponding_to_keylines.at(mt.queryIdx) = std::make_pair(1.0, 1.0);
                        }
                    }
                }
            }

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
        }

        //-----------------------------------------
        // (default) RGB-D
        frame::frame(const cv::Mat &img_gray, const cv::Mat &img_depth, const double timestamp,
                     feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                     camera::base *camera, const float depth_thr,
                     const cv::Mat &mask)
            : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor), extractor_right_(nullptr),
              timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr)
        {
            // Get ORB scale
            update_orb_info();

            // Extract ORB feature
            extract_orb(img_gray, mask);
            num_keypts_ = keypts_.size();
            if (keypts_.empty())
            {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Calculate disparity from depth
            compute_stereo_from_depth(img_depth);

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
        }

        // FW: RGB-D + planar segmentation
        frame::frame(const cv::Mat &img_gray, const cv::Mat &img_depth, const cv::Mat &img_seg_mask, const double timestamp,
                     feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                     camera::base *camera, const float depth_thr,
                     const cv::Mat &mask)
            : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor), extractor_right_(nullptr),
              timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr),
              _img_seg_mask(img_seg_mask), _depth_img(img_depth)
        {
            // Get ORB scale
            update_orb_info();

            // Extract ORB feature
            extract_orb(img_gray, mask);
            num_keypts_ = keypts_.size();
            if (keypts_.empty())
            {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Calculate disparity from depth
            compute_stereo_from_depth(img_depth);

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
        }

        // FW: RGB-D + planar segmentation + line segments extractor
        frame::frame(const cv::Mat &img_gray, const cv::Mat &img_depth, const cv::Mat &img_seg_mask, const double timestamp,
                     feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                     feature::LineFeatureTracker *line_extractor,
                     camera::base *camera, const float depth_thr,
                     const cv::Mat &mask)
            : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor), extractor_right_(nullptr),
              timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr),
              _img_seg_mask(img_seg_mask), _depth_img(img_depth),
              _line_extractor(line_extractor)
        {
            // Get ORB scale
            update_orb_info();

            // FW: extract ORB feature and Line features in parallel
            std::thread thread_for_Point(&frame::extract_orb, this, img_gray, mask, image_side::Left);
            std::thread thread_for_Line(&frame::extract_line, this, img_gray, std::ref(_keylsd), std::ref(_lbd_descr), std::ref(_keyline_functions));
            thread_for_Point.join();
            thread_for_Line.join();

            num_keypts_ = keypts_.size();
            if (keypts_.empty())
            {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // FW:
            if (!_keylsd.empty() && _line_extractor)
            {
                update_lsd_info();

                // initialize association with 3D lines
                _num_keylines = _keylsd.size();
                _landmarks_line = std::vector<Line *>(_num_keylines, nullptr);
                _outlier_flags_line = std::vector<bool>(_num_keylines, false);
                _depths_cooresponding_to_keylines = std::vector<std::pair<float, float>>(_num_keylines, std::make_pair(-1, -1));
                _stereo_x_right_cooresponding_to_keylines = std::vector<std::pair<float, float>>(_num_keylines, std::make_pair(-1, -1));
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Calculate disparity from depth
            compute_stereo_from_depth(img_depth);

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
        }

        // FW: RGB-D frame + line segments extractor
        frame::frame(const cv::Mat &img_gray, const cv::Mat &img_depth, const double timestamp,
                     feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                     feature::LineFeatureTracker *line_extractor, // add line extractor
                     camera::base *camera, const float depth_thr,
                     const cv::Mat &mask)
            : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor), extractor_right_(nullptr),
              timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr),
              _line_extractor(line_extractor)
        {
            // Get ORB scale
            update_orb_info();

            // FW: extract ORB feature and Line features in parallel
            std::thread thread_for_Point(&frame::extract_orb, this, img_gray, mask, image_side::Left);
            std::thread thread_for_Line(&frame::extract_line, this, img_gray, std::ref(_keylsd), std::ref(_lbd_descr), std::ref(_keyline_functions));
            thread_for_Point.join();
            thread_for_Line.join();

            num_keypts_ = keypts_.size();
            if (keypts_.empty())
            {
                spdlog::warn("frame {}: cannot extract any keypoints", id_);
            }

            // FW:
            if (!_keylsd.empty() && _line_extractor)
            {
                update_lsd_info();

                // initialize association with 3D lines
                _num_keylines = _keylsd.size();
                _landmarks_line = std::vector<Line *>(_num_keylines, nullptr);
                _outlier_flags_line = std::vector<bool>(_num_keylines, false);
                _depths_cooresponding_to_keylines = std::vector<std::pair<float, float>>(_num_keylines, std::make_pair(-1, -1));
                _stereo_x_right_cooresponding_to_keylines = std::vector<std::pair<float, float>>(_num_keylines, std::make_pair(-1, -1));
            }

            // Undistort keypoints
            camera_->undistort_keypoints(keypts_, undist_keypts_);

            // Calculate disparity from depth
            compute_stereo_from_depth(img_depth);

            // Convert to bearing vector
            camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

            // Initialize association with 3D points
            landmarks_ = std::vector<landmark *>(num_keypts_, nullptr);
            outlier_flags_ = std::vector<bool>(num_keypts_, false);

            // Assign all the keypoints into grid
            assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
        }

        //-----------------------------------------
        void frame::set_cam_pose(const Mat44_t &cam_pose_cw)
        {
            cam_pose_cw_is_valid_ = true;
            cam_pose_cw_ = cam_pose_cw;
            update_pose_params();
        }

        void frame::set_cam_pose(const g2o::SE3Quat &cam_pose_cw)
        {
            set_cam_pose(util::converter::to_eigen_mat(cam_pose_cw));
        }

        void frame::update_pose_params()
        {
            rot_cw_ = cam_pose_cw_.block<3, 3>(0, 0);
            rot_wc_ = rot_cw_.transpose();
            trans_cw_ = cam_pose_cw_.block<3, 1>(0, 3);
            cam_center_ = -rot_cw_.transpose() * trans_cw_;
        }

        Vec3_t frame::get_cam_center() const
        {
            return cam_center_;
        }

        Mat33_t frame::get_rotation_inv() const
        {
            return rot_wc_;
        }

        void frame::update_orb_info()
        {
            num_scale_levels_ = extractor_->get_num_scale_levels();
            scale_factor_ = extractor_->get_scale_factor();
            log_scale_factor_ = std::log(scale_factor_);
            scale_factors_ = extractor_->get_scale_factors();
            inv_scale_factors_ = extractor_->get_inv_scale_factors();
            level_sigma_sq_ = extractor_->get_level_sigma_sq();
            inv_level_sigma_sq_ = extractor_->get_inv_level_sigma_sq();
        }

        void frame::update_lsd_info()
        {
            _num_scale_levels_lsd = _line_extractor->get_num_scale_levels();
            _scale_factor_lsd = _line_extractor->get_scale_factor();
            _log_scale_factor_lsd = std::log(_scale_factor_lsd);
            _scale_factors_lsd = _line_extractor->get_scale_factors();
            _inv_scale_factors_lsd = _line_extractor->get_inv_scale_factors();
            _level_sigma_sq_lsd = _line_extractor->get_level_sigma_sq();
            _inv_level_sigma_sq_lsd = _line_extractor->get_inv_level_sigma_sq();
        }

        void frame::compute_bow()
        {
            if (bow_vec_.empty())
            {
#ifdef USE_DBOW2
                bow_vocab_->transform(util::converter::to_desc_vec(descriptors_), bow_vec_, bow_feat_vec_, 4);
#else
                bow_vocab_->transform(descriptors_, 4, bow_vec_, bow_feat_vec_);
#endif
            }
        }

        bool frame::can_observe(landmark *lm, const float ray_cos_thr,
                                Vec2_t &reproj, float &x_right, unsigned int &pred_scale_level) const
        {
            const Vec3_t pos_w = lm->get_pos_in_world();

            const bool in_image = camera_->reproject_to_image(rot_cw_, trans_cw_, pos_w, reproj, x_right);
            if (!in_image)
            {
                return false;
            }

            const Vec3_t cam_to_lm_vec = pos_w - cam_center_;
            const auto cam_to_lm_dist = cam_to_lm_vec.norm();
            if (!lm->is_inside_in_orb_scale(cam_to_lm_dist))
            {
                return false;
            }

            const Vec3_t obs_mean_normal = lm->get_obs_mean_normal();
            const auto ray_cos = cam_to_lm_vec.dot(obs_mean_normal) / cam_to_lm_dist;
            if (ray_cos < ray_cos_thr)
            {
                return false;
            }

            pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, this);
            return true;
        }

        // FW: check if the 3D line within the frustum of the image
        bool frame::can_observe_line(Line *line,
                                     Vec2_t &reproj_sp, float &x_right_sp,
                                     Vec2_t &reproj_ep, float &x_right_ep, unsigned int &pred_scale_level) const
        {
            const Vec6_t pos_w = line->get_pos_in_world();
            Vec3_t pos_w_sp = pos_w.head(3);
            Vec3_t pos_w_ep = pos_w.tail(3);

            // check if the two points are within the image frustum
            const bool in_image_sp = camera_->reproject_to_image(rot_cw_, trans_cw_, pos_w_sp, reproj_sp, x_right_sp); // starting point
            const bool in_image_ep = camera_->reproject_to_image(rot_cw_, trans_cw_, pos_w_ep, reproj_ep, x_right_ep); // ending point

            // two endpoints both are out of range
            if (!in_image_sp && !in_image_ep)
            {
                return false;
            }

            // check if the 3D line is partially occluded
            bool partial_occlusion = false;
            if (!in_image_sp || !in_image_ep)
            {
                // check if the middle point are within the image frustum
                Vec3_t pos_w_mp = 0.5 * (pos_w_sp + pos_w_ep);
                Vec2_t reproj_mp;
                float x_right_mp;
                const bool in_image_mp = camera_->reproject_to_image(rot_cw_, trans_cw_, pos_w_mp, reproj_mp, x_right_mp);

                if (in_image_mp)
                {
                    // partially occluded is ok, it would be good, if the middle point is visible
                    partial_occlusion = true;
                }
                else
                {
                    // if the middle point is not even visible, discard this landmark
                    return false;
                }
            }

            // check if the distance between 3D line and camera center within the scale
            const Vec3_t cam_to_lm_vec = 0.5 * (pos_w_sp + pos_w_ep) - cam_center_;
            const auto cam_to_lm_dist = cam_to_lm_vec.norm();
            if (!line->is_inside_in_feature_scale(cam_to_lm_dist))
            {
                return false;
            }

            // predict scale in the image
            pred_scale_level = line->predict_scale_level(cam_to_lm_dist, this->_log_scale_factor_lsd, this->_num_scale_levels_lsd);

            return true;
        }

        std::vector<unsigned int> frame::get_keypoints_in_cell(const float ref_x, const float ref_y, const float margin, const int min_level, const int max_level) const
        {
            return data::get_keypoints_in_cell(camera_, undist_keypts_, keypt_indices_in_cells_, ref_x, ref_y, margin, min_level, max_level);
        }

        // FW:
        std::vector<unsigned int> frame::get_keylines_in_cell(const float ref_x1, const float ref_y1,
                                                              const float ref_x2, const float ref_y2,
                                                              const float margin,
                                                              const int min_level, const int max_level) const
        {
            return data::get_keylines_in_cell(_keylsd, ref_x1, ref_y1, ref_x2, ref_y2, margin, min_level, max_level);
        }

        Vec3_t frame::triangulate_stereo(const unsigned int idx) const
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

                    // Convert from camera coordinates to world coordinates
                    return rot_wc_ * pos_c + cam_center_;
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

                    // Convert from camera coordinates to world coordinates
                    return rot_wc_ * pos_c + cam_center_;
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

        // FW:
        Vec6_t frame::triangulate_stereo_for_line(const unsigned int idx) const
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
                    Vec3_t pos_w_sp = rot_wc_ * pos_c_sp + cam_center_;
                    Vec3_t pos_w_ep = rot_wc_ * pos_c_ep + cam_center_;

                    Vec6_t pos_w_line;
                    pos_w_line << pos_w_sp(0), pos_w_sp(1), pos_w_sp(2), pos_w_ep(0), pos_w_ep(1), pos_w_ep(2);

                    return pos_w_line;
                }
                else
                {
                    return Vec6_t::Zero();
                }
            }

            // triangulate 3D lins using stereo image pair
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
                Vec3_t pos_w_sp = rot_wc_ * sp_c_3D + cam_center_;
                Vec3_t pos_w_ep = rot_wc_ * ep_c_3D + cam_center_;

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

            return Vec6_t::Zero();
        }

        void frame::extract_orb(const cv::Mat &img, const cv::Mat &mask, const image_side &img_side)
        {
            switch (img_side)
            {
            case image_side::Left:
            {
                extractor_->extract(img, mask, keypts_, descriptors_);
                break;
            }
            case image_side::Right:
            {
                extractor_right_->extract(img, mask, keypts_right_, descriptors_right_);
                break;
            }
            }
        }

        // FW:
        void frame::extract_line(const cv::Mat &img, std::vector<cv::line_descriptor::KeyLine> &keylsd,
                                 cv::Mat &lbd_descr, std::vector<Vec3_t> &keyline_functions)
        {
            _line_extractor->extract_LSD_LBD(img, keylsd, lbd_descr, keyline_functions);
        }

        // FW:
        void frame::extract_line_stereo(const cv::Mat &img, std::vector<cv::line_descriptor::KeyLine> &keylsd,
                                        cv::Mat &lbd_descr, std::vector<Vec3_t> &keyline_functions,
                                        const image_side &img_side)
        {
            switch (img_side)
            {
            case image_side::Left:
            {
                _line_extractor->extract_LSD_LBD(img, keylsd, lbd_descr, keyline_functions);
                break;
            }
            case image_side::Right:
            {
                _line_extractor_right->extract_LSD_LBD(img, keylsd, lbd_descr, keyline_functions);
                break;
            }
            }
        }

        void frame::compute_stereo_from_depth(const cv::Mat &right_img_depth)
        {
            assert(camera_->setup_type_ == camera::setup_type_t::RGBD);

            // Initialize with invalid value
            stereo_x_right_ = std::vector<float>(num_keypts_, -1);
            depths_ = std::vector<float>(num_keypts_, -1);

            for (unsigned int idx = 0; idx < num_keypts_; idx++)
            {
                const auto &keypt = keypts_.at(idx);
                const auto &undist_keypt = undist_keypts_.at(idx);

                const float x = keypt.pt.x;
                const float y = keypt.pt.y;

                const float depth = right_img_depth.at<float>(y, x);

                if (depth <= 0)
                {
                    continue;
                }

                depths_.at(idx) = depth;
                stereo_x_right_.at(idx) = undist_keypt.pt.x - camera_->focal_x_baseline_ / depth;
            }

            // FW: generate depths corresponding to keylines
            // for RGB-D SLAM initialization/triangulation
            if (_num_keylines != 0)
            {
                for (unsigned int idx_l = 0; idx_l < _num_keylines; ++idx_l)
                {
                    const auto &keyline = _keylsd.at(idx_l);
                    const auto &sp = keyline.getStartPoint();
                    const auto &ep = keyline.getEndPoint();

                    const float depth_sp = right_img_depth.at<float>(sp.y, sp.x);
                    const float depth_ep = right_img_depth.at<float>(ep.y, ep.x);

                    if (depth_sp < 0 || depth_ep < 0)
                    {
                        continue;
                    }

                    _depths_cooresponding_to_keylines.at(idx_l) = std::make_pair(depth_sp, depth_ep);
                    _stereo_x_right_cooresponding_to_keylines.at(idx_l) = std::make_pair(sp.x - camera_->focal_x_baseline_ / depth_sp,
                                                                                         ep.x - camera_->focal_x_baseline_ / depth_ep);
                }
            }
        }

    } // namespace data
} // namespace PLPSLAM
