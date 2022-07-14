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

#ifndef PLPSLAM_DATA_KEYFRAME_H
#define PLPSLAM_DATA_KEYFRAME_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/camera/base.h"
#include "PLPSLAM/data/graph_node.h"
#include "PLPSLAM/data/bow_vocabulary.h"

#include <set>
#include <mutex>
#include <atomic>

#include <g2o/types/sba/types_six_dof_expmap.h>
#include <nlohmann/json_fwd.hpp>

#ifdef USE_DBOW2
#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>
#else
#include <fbow/fbow.h>
#endif

// FW: for LSD ...
#include <opencv2/features2d.hpp>
#include "PLPSLAM/feature/line_descriptor/line_descriptor_custom.hpp"
#include "PLPSLAM/data/landmark_line.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace PLPSLAM
{

       namespace camera
       {
              class base;
       }

       namespace data
       {

              class frame;
              class landmark;
              class map_database;
              class bow_database;
              class Line; // FW:

              class keyframe
              {
              public:
                     EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                     // operator overrides
                     bool operator==(const keyframe &keyfrm) const { return id_ == keyfrm.id_; }
                     bool operator!=(const keyframe &keyfrm) const { return !(*this == keyfrm); }
                     bool operator<(const keyframe &keyfrm) const { return id_ < keyfrm.id_; }
                     bool operator<=(const keyframe &keyfrm) const { return id_ <= keyfrm.id_; }
                     bool operator>(const keyframe &keyfrm) const { return id_ > keyfrm.id_; }
                     bool operator>=(const keyframe &keyfrm) const { return id_ >= keyfrm.id_; }

                     /**
                      * Constructor for building from a frame
                      */
                     keyframe(const frame &frm, map_database *map_db, bow_database *bow_db);

                     /**
                      * Constructor for map loading
                      * (NOTE: some variables must be recomputed after the construction. See the definition.)
                      */
                     keyframe(const unsigned int id, const unsigned int src_frm_id, const double timestamp,
                              const Mat44_t &cam_pose_cw, camera::base *camera, const float depth_thr,
                              const unsigned int num_keypts, const std::vector<cv::KeyPoint> &keypts,
                              const std::vector<cv::KeyPoint> &undist_keypts, const eigen_alloc_vector<Vec3_t> &bearings,
                              const std::vector<float> &stereo_x_right, const std::vector<float> &depths, const cv::Mat &descriptors,
                              const unsigned int num_scale_levels, const float scale_factor,
                              bow_vocabulary *bow_vocab, bow_database *bow_db, map_database *map_db);

                     // FW: map loading (points + lines)
                     keyframe(const unsigned int id, const unsigned int src_frm_id, const double timestamp,
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
                              bow_vocabulary *bow_vocab, bow_database *bow_db, map_database *map_db);

                     /**
                      * Encode this keyframe information as JSON
                      */
                     nlohmann::json to_json() const;

                     //-----------------------------------------
                     // camera pose

                     /**
                      * Set camera pose
                      */
                     void set_cam_pose(const Mat44_t &cam_pose_cw);

                     /**
                      * Set camera pose
                      */
                     void set_cam_pose(const g2o::SE3Quat &cam_pose_cw);

                     /**
                      * Get the camera pose
                      */
                     Mat44_t get_cam_pose() const;

                     /**
                      * Get the inverse of the camera pose
                      */
                     Mat44_t get_cam_pose_inv() const;

                     /**
                      * Get the camera center
                      */
                     Vec3_t get_cam_center() const;

                     /**
                      * Get the rotation of the camera pose
                      */
                     Mat33_t get_rotation() const;

                     /**
                      * Get the translation of the camera pose
                      */
                     Vec3_t get_translation() const;

                     //-----------------------------------------
                     // features and observations

                     /**
                      * Compute BoW representation
                      */
                     void compute_bow();

                     /**
                      * Add a landmark observed by myself at keypoint idx
                      */
                     void add_landmark(landmark *lm, const unsigned int idx);

                     // FW: for line landmark
                     void add_landmark_line(Line *line, const unsigned int idx);

                     /**
                      * Erase a landmark observed by myself at keypoint idx
                      */
                     void erase_landmark_with_index(const unsigned int idx);

                     // FW: for line landmark
                     void erase_landmark_line_with_index(const unsigned int idx);

                     /**
                      * Erase a landmark
                      */
                     void erase_landmark(landmark *lm);

                     // FW: for line landmark
                     void erase_landmark_line(Line *line);

                     /**
                      * Replace the landmark
                      */
                     void replace_landmark(landmark *lm, const unsigned int idx);

                     // FW: for line landmark
                     void replace_landmark_line(Line *line, const unsigned int idx);

                     /**
                      * Get all of the landmarks
                      * (NOTE: including nullptr)
                      */
                     std::vector<landmark *> get_landmarks() const;

                     // FW: for line landmark
                     std::vector<Line *> get_landmarks_line() const;

                     /**
                      * Get the valid landmarks
                      */
                     std::set<landmark *> get_valid_landmarks() const;

                     // FW: for line landmark
                     std::set<Line *> get_valid_landmarks_line() const;

                     /**
                      * Get the number of tracked landmarks which have observers equal to or greater than the threshold
                      */
                     unsigned int get_num_tracked_landmarks(const unsigned int min_num_obs_thr) const;

                     // FW: for line landmark
                     unsigned int get_num_tracked_landmarks_line(const unsigned int min_num_obs_thr) const;

                     /**
                      * Get the landmark associated keypoint idx
                      */
                     landmark *get_landmark(const unsigned int idx) const;

                     // FW: for line landmark
                     Line *get_landmark_line(const unsigned int idx) const;

                     /**
                      * Get the keypoint indices in the cell which reference point is located
                      */
                     std::vector<unsigned int> get_keypoints_in_cell(const float ref_x, const float ref_y, const float margin) const;

                     // FW: same as frame.h/cc
                     std::vector<unsigned int> get_keylines_in_cell(const float ref_x1, const float ref_y1,
                                                                    const float ref_x2, const float ref_y2,
                                                                    const float margin,
                                                                    const int min_level = -1, const int max_level = -1) const;

                     /**
                      * Triangulate the keypoint using the disparity
                      * NOTE: RGB-D camera only
                      */
                     Vec3_t triangulate_stereo(const unsigned int idx) const;

                     // FW: for line landmark, used for triangulation (RGB-D/Stereo)
                     Vec6_t triangulate_stereo_for_line(const unsigned int idx) const;

                     /**
                      * Compute median of depths
                      */
                     float compute_median_depth(const bool abs = false) const;

                     //-----------------------------------------
                     // flags

                     /**
                      * Set this keyframe as non-erasable
                      */
                     void set_not_to_be_erased();

                     /**
                      * Set this keyframe as erasable
                      */
                     void set_to_be_erased();

                     /**
                      * Erase this keyframe
                      */
                     void prepare_for_erasing();

                     /**
                      * Whether this keyframe will be erased shortly or not
                      */
                     bool will_be_erased();

                     //-----------------------------------------
                     // for local map update

                     //! identifier for local map update
                     unsigned int local_map_update_identifier = 0;

                     //-----------------------------------------
                     // for loop BA

                     //! identifier for loop BA
                     unsigned int loop_BA_identifier_ = 0;
                     //! camera pose AFTER loop BA
                     Mat44_t cam_pose_cw_after_loop_BA_;
                     //! camera pose BEFORE loop BA
                     Mat44_t cam_pose_cw_before_BA_;

                     //-----------------------------------------
                     // meta information

                     //! keyframe ID
                     unsigned int id_;
                     //! next keyframe ID
                     static std::atomic<unsigned int> next_id_;

                     //! source frame ID
                     const unsigned int src_frm_id_;

                     //! timestamp in seconds
                     const double timestamp_;

                     //-----------------------------------------
                     // camera parameters

                     //! camera model
                     camera::base *camera_;
                     //! depth threshold
                     const float depth_thr_;

                     //-----------------------------------------
                     // constant observations

                     //! number of keypoints
                     const unsigned int num_keypts_;

                     //! keypoints of monocular or stereo left image
                     const std::vector<cv::KeyPoint> keypts_;
                     //! undistorted keypoints of monocular or stereo left image
                     const std::vector<cv::KeyPoint> undist_keypts_;
                     //! bearing vectors
                     const eigen_alloc_vector<Vec3_t> bearings_;

                     //! keypoint indices in each of the cells
                     const std::vector<std::vector<std::vector<unsigned int>>> keypt_indices_in_cells_;

                     //! disparities
                     const std::vector<float> stereo_x_right_;
                     //! depths
                     const std::vector<float> depths_;

                     //! descriptors
                     const cv::Mat descriptors_;

                     //! BoW features (DBoW2 or FBoW)
#ifdef USE_DBOW2
                     DBoW2::BowVector bow_vec_;
                     DBoW2::FeatureVector bow_feat_vec_;
#else
                     fbow::BoWVector bow_vec_;
                     fbow::BoWFeatVector bow_feat_vec_;
#endif

                     //-----------------------------------------
                     // covisibility graph

                     //! graph node
                     const std::unique_ptr<graph_node> graph_node_ = nullptr;

                     //-----------------------------------------
                     // ORB scale pyramid information

                     //! number of scale levels
                     const unsigned int num_scale_levels_;
                     //! scale factor
                     const float scale_factor_;
                     //! log scale factor
                     const float log_scale_factor_;
                     //! list of scale factors
                     const std::vector<float> scale_factors_;
                     //! list of sigma^2 (sigma=1.0 at scale=0) for optimization
                     const std::vector<float> level_sigma_sq_;
                     //! list of 1 / sigma^2 for optimization
                     const std::vector<float> inv_level_sigma_sq_;

                     //-----------------------------------------
                     // FW: for planar_mapping_module
                     void set_segmentation_mask(const cv::Mat &img_seg_mask);
                     cv::Mat get_segmentation_mask() const;

                     void set_depth_map(const cv::Mat &depth_img);
                     cv::Mat get_depth_map() const;

                     void set_img_rgb(const cv::Mat &img_rgb);
                     cv::Mat get_img_rgb() const;

                     //-----------------------------------------
                     // FW: for line tracking
                     // monocular / RGB-D / left of stereo
                     const std::vector<cv::line_descriptor::KeyLine> _keylsd;
                     const cv::Mat _lbd_descr;
                     const std::vector<Vec3_t> _keyline_functions;
                     unsigned int _num_keylines;

                     // pair of <starting point (sp), ending point (ep)>
                     // used only for RGB-D, a trivial value 1.0 is assigned for stereo as indicator
                     const std::vector<std::pair<float, float>> _depths_cooresponding_to_keylines;         // depths
                     const std::vector<std::pair<float, float>> _stereo_x_right_cooresponding_to_keylines; // disparities

                     // right image of stereo
                     const std::vector<cv::line_descriptor::KeyLine> _keylsd_right;
                     const cv::Mat _lbd_descr_right;
                     const std::vector<Vec3_t> _keyline_functions_right;
                     const std::unordered_map<int, int> _good_matches_stereo; // 2D-2D line matches between stereo image pair

                     // FW:
                     // LSD scale pyramid information
                     //! number of scale levels
                     unsigned int _num_scale_levels_lsd;
                     //! scale factor
                     float _scale_factor_lsd;
                     //! log scale factor
                     float _log_scale_factor_lsd;
                     //! list of scale factors
                     std::vector<float> _scale_factors_lsd;
                     //! list of inverse of scale factors
                     std::vector<float> _inv_scale_factors_lsd;
                     //! list of sigma^2 (sigma=1.0 at scale=0) for optimization
                     std::vector<float> _level_sigma_sq_lsd;
                     //! list of 1 / sigma^2 for optimization
                     std::vector<float> _inv_level_sigma_sq_lsd;

                     void initialize_lsd_scale_information();

                     inline Mat33_t skew(const Vec3_t &t) const
                     {
                            Mat33_t S;
                            S << 0, -t.z(), t.y(), t.z(), 0, -t.x(), -t.y(), t.x(), 0;
                            return S;
                     }

              private:
                     //-----------------------------------------
                     // camera pose

                     //! need mutex for access to poses
                     mutable std::mutex mtx_pose_;
                     //! camera pose from the world to the current
                     Mat44_t cam_pose_cw_;
                     //! camera pose from the current to the world
                     Mat44_t cam_pose_wc_;
                     //! camera center
                     Vec3_t cam_center_;

                     //-----------------------------------------
                     // observations

                     //! need mutex for access to observations
                     mutable std::mutex mtx_observations_;
                     //! observed landmarks
                     std::vector<landmark *> landmarks_;

                     //-----------------------------------------
                     // FW: for planar mapping module
                     cv::Mat _img_seg_mask; // for planar mapping
                     cv::Mat _depth_img;    // for dense reconstruction (demo)
                     cv::Mat _img_rgb;      // for assigning color to the dense point cloud (demo)

                     //-----------------------------------------
                     // FW: for line tracking
                     std::vector<Line *> _landmarks_line;

                     //-----------------------------------------
                     // databases

                     //! map database
                     map_database *map_db_;
                     //! BoW database
                     bow_database *bow_db_;
                     //! BoW vocabulary
                     bow_vocabulary *bow_vocab_;

                     //-----------------------------------------
                     // flags

                     //! flag which indicates this keyframe is erasable or not
                     std::atomic<bool> cannot_be_erased_{false};

                     //! flag which indicates this keyframe will be erased
                     std::atomic<bool> will_be_erased_{false};
              };

       } // namespace data
} // namespace PLPSLAM

#endif // PLPSLAM_DATA_KEYFRAME_H
