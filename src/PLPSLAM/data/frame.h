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

#ifndef PLPSLAM_DATA_FRAME_H
#define PLPSLAM_DATA_FRAME_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/camera/base.h"
#include "PLPSLAM/util/converter.h"
#include "PLPSLAM/data/bow_vocabulary.h"

#include <vector>
#include <atomic>

#include <opencv2/core.hpp>
#include <Eigen/Core>

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

      namespace feature
      {
            class orb_extractor;
            class LineFeatureTracker; // FW:

      }

      namespace data
      {

            class keyframe;
            class landmark;
            class Line; // FW:

            class frame
            {
            public:
                  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                  frame() = default;

                  bool operator==(const frame &frm) { return this->id_ == frm.id_; }
                  bool operator!=(const frame &frm) { return !(*this == frm); }

                  //-----------------------------------------
                  // (default) Constructor for monocular frame
                  frame(const cv::Mat &img_gray, const double timestamp,
                        feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                        camera::base *camera, const float depth_thr,
                        const cv::Mat &mask = cv::Mat{});

                  // FW: Monocular frame + planar segmentation
                  frame(const cv::Mat &img_gray, const double timestamp, const cv::Mat &img_seg_mask,
                        feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                        camera::base *camera, const float depth_thr,
                        const cv::Mat &mask = cv::Mat{});

                  // FW: Monocular frame + planar segmentation + line segments extractor
                  frame(const cv::Mat &img_gray, const double timestamp, const cv::Mat &img_seg_mask,
                        feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                        feature::LineFeatureTracker *line_extractor,
                        camera::base *camera, const float depth_thr,
                        const cv::Mat &mask = cv::Mat{});

                  // FW: Monocular frame + line segments extractor
                  frame(const cv::Mat &img_gray, const double timestamp,
                        feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                        feature::LineFeatureTracker *line_extractor,
                        camera::base *camera, const float depth_thr,
                        const cv::Mat &mask = cv::Mat{});

                  //-----------------------------------------
                  // (default) Constructor for stereo frame
                  frame(const cv::Mat &left_img_gray, const cv::Mat &right_img_gray, const double timestamp,
                        feature::orb_extractor *extractor_left, feature::orb_extractor *extractor_right, bow_vocabulary *bow_vocab,
                        camera::base *camera, const float depth_thr,
                        const cv::Mat &mask = cv::Mat{});

                  // FW: stereo + planar segmentation
                  frame(const cv::Mat &left_img_gray, const cv::Mat &right_img_gray, const cv::Mat &img_seg_mask, const double timestamp,
                        feature::orb_extractor *extractor_left, feature::orb_extractor *extractor_right, bow_vocabulary *bow_vocab,
                        camera::base *camera, const float depth_thr,
                        const cv::Mat &mask = cv::Mat{});

                  // FW: stereo + planar segmentation + line segments extractor
                  frame(const cv::Mat &left_img_gray, const cv::Mat &right_img_gray, const cv::Mat &img_seg_mask, const double timestamp,
                        feature::orb_extractor *extractor_left, feature::orb_extractor *extractor_right, bow_vocabulary *bow_vocab,
                        feature::LineFeatureTracker *line_extractor, feature::LineFeatureTracker *line_extractor_right,
                        camera::base *camera, const float depth_thr,
                        const cv::Mat &mask = cv::Mat{});

                  // FW: Stereo frame + line segment extractor
                  frame(const cv::Mat &left_img_gray, const cv::Mat &right_img_gray, const double timestamp,
                        feature::orb_extractor *extractor_left, feature::orb_extractor *extractor_right, bow_vocabulary *bow_vocab,
                        feature::LineFeatureTracker *line_extractor, feature::LineFeatureTracker *line_extractor_right,
                        camera::base *camera, const float depth_thr,
                        const cv::Mat &mask = cv::Mat{});

                  //-----------------------------------------
                  // (default) Constructor for RGBD frame
                  frame(const cv::Mat &img_gray, const cv::Mat &img_depth, const double timestamp,
                        feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                        camera::base *camera, const float depth_thr,
                        const cv::Mat &mask = cv::Mat{});

                  // FW: RGB-D frame with planar segmentation
                  frame(const cv::Mat &img_gray, const cv::Mat &img_depth, const cv::Mat &img_seg_mask, const double timestamp,
                        feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                        camera::base *camera, const float depth_thr,
                        const cv::Mat &mask = cv::Mat{});

                  // FW: RGB-D frame with planar segmentation + line segments extractor
                  frame(const cv::Mat &img_gray, const cv::Mat &img_depth, const cv::Mat &img_seg_mask, const double timestamp,
                        feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                        feature::LineFeatureTracker *line_extractor,
                        camera::base *camera, const float depth_thr,
                        const cv::Mat &mask = cv::Mat{});

                  // FW: RGB-D frame with line segments extractor
                  frame(const cv::Mat &img_gray, const cv::Mat &img_depth, const double timestamp,
                        feature::orb_extractor *extractor, bow_vocabulary *bow_vocab,
                        feature::LineFeatureTracker *line_extractor,
                        camera::base *camera, const float depth_thr,
                        const cv::Mat &mask = cv::Mat{});

                  //-----------------------------------------
                  /**
                   * Set camera pose and refresh rotation and translation
                   * @param cam_pose_cw
                   */
                  void set_cam_pose(const Mat44_t &cam_pose_cw);

                  /**
                   * Set camera pose and refresh rotation and translation
                   * @param cam_pose_cw
                   */
                  void set_cam_pose(const g2o::SE3Quat &cam_pose_cw);

                  /**
                   * Update rotation and translation using cam_pose_cw_
                   */
                  void update_pose_params();

                  /**
                   * Get camera center
                   * @return
                   */
                  Vec3_t get_cam_center() const;

                  /**
                   * Get inverse of rotation
                   * @return
                   */
                  Mat33_t get_rotation_inv() const;

                  /**
                   * Update ORB information
                   */
                  void update_orb_info();

                  // FW:
                  void update_lsd_info();

                  /**
                   * Compute BoW representation
                   */
                  void compute_bow();

                  /**
                   * Check observability of the landmark
                   */
                  // FW: same function as "bool isInFrustum()" in ORB-SLAM2
                  bool can_observe(landmark *lm, const float ray_cos_thr,
                                   Vec2_t &reproj, float &x_right, unsigned int &pred_scale_level) const;

                  // FW: check if the 3D line is in the frustum
                  // used in tracking_module::search_local_landmarks_line()
                  // notice: for 3D line we do not check ray angle
                  bool can_observe_line(Line *line,
                                        Vec2_t &reproj_sp, float &x_right_sp,
                                        Vec2_t &reproj_ep, float &x_right_ep, unsigned int &pred_scale_level) const;

                  /**
                   * Get keypoint indices in the cell which reference point is located
                   * @param ref_x
                   * @param ref_y
                   * @param margin
                   * @param min_level
                   * @param max_level
                   * @return
                   */
                  // FW: same function as "vector<size_t> GetFeaturesInArea()" in ORB-SLAM2
                  // find the feature point in the circle with x, y as center, whithin radius as r (margin) and scale in [minLevel, maxLevel]
                  std::vector<unsigned int> get_keypoints_in_cell(const float ref_x, const float ref_y, const float margin, const int min_level = -1, const int max_level = -1) const;

                  // FW: find the feature line in the circle with middle point of the line as center, within radius as r (margin) and scale in [minLevel, maxLevel]
                  // compare to the point landmark, we are missing:
                  // (1) assigning keylines into grid/cell
                  // (2) find keylines in the cell based on the information from (1)
                  std::vector<unsigned int> get_keylines_in_cell(const float ref_x1, const float ref_y1,
                                                                 const float ref_x2, const float ref_y2,
                                                                 const float margin,
                                                                 const int min_level = -1, const int max_level = -1) const;

                  /**
                   * Perform stereo triangulation of the keypoint
                   * @param idx
                   * @return
                   */
                  // FW: used in initializer::create_map_for_stereo() and keyframe_inserter::insert_new_keyframe()
                  Vec3_t triangulate_stereo(const unsigned int idx) const;

                  // FW: used in initializer::create_map_for_stereo()
                  Vec6_t triangulate_stereo_for_line(const unsigned int idx) const;

                  //! current frame ID
                  unsigned int id_;

                  //! next frame ID
                  static std::atomic<unsigned int> next_id_;

                  //! BoW vocabulary (DBoW2 or FBoW)
                  bow_vocabulary *bow_vocab_ = nullptr;

                  // ORB extractor
                  //! ORB extractor for monocular or stereo left image
                  feature::orb_extractor *extractor_ = nullptr;
                  //! ORB extractor for stereo right image
                  feature::orb_extractor *extractor_right_ = nullptr;

                  //! timestamp
                  double timestamp_;

                  //! camera model
                  camera::base *camera_ = nullptr;

                  // if a stereo-triangulated point is farther than this threshold, it is invalid
                  //! depth threshold
                  float depth_thr_;

                  //! number of keypoints
                  unsigned int num_keypts_ = 0;

                  // keypoints
                  //! keypoints of monocular or stereo left image
                  std::vector<cv::KeyPoint> keypts_;
                  //! keypoints of stereo right image
                  std::vector<cv::KeyPoint> keypts_right_;
                  //! undistorted keypoints of monocular or stereo left image
                  std::vector<cv::KeyPoint> undist_keypts_;
                  //! bearing vectors
                  eigen_alloc_vector<Vec3_t> bearings_;

                  //! disparities
                  std::vector<float> stereo_x_right_;
                  //! depths
                  std::vector<float> depths_;

                  // FW: pair of <starting point (sp), ending point (ep)>
                  std::vector<std::pair<float, float>> _depths_cooresponding_to_keylines;         // depths
                  std::vector<std::pair<float, float>> _stereo_x_right_cooresponding_to_keylines; // disparities

                  //! BoW features (DBoW2 or FBoW)
#ifdef USE_DBOW2
                  DBoW2::BowVector bow_vec_;
                  DBoW2::FeatureVector bow_feat_vec_;
#else
                  fbow::BoWVector bow_vec_;
                  fbow::BoWFeatVector bow_feat_vec_;
#endif

                  // ORB descriptors
                  //! ORB descriptors of monocular or stereo left image
                  cv::Mat descriptors_;
                  //! ORB descriptors of stereo right image
                  cv::Mat descriptors_right_;

                  //! landmarks, whose nullptr indicates no-association
                  std::vector<landmark *> landmarks_;

                  //! outlier flags, which are mainly used in pose optimization and bundle adjustment
                  std::vector<bool> outlier_flags_;

                  //! cells for storing keypoint indices
                  std::vector<std::vector<std::vector<unsigned int>>> keypt_indices_in_cells_;

                  //! camera pose: world -> camera
                  bool cam_pose_cw_is_valid_ = false;
                  Mat44_t cam_pose_cw_;

                  //! reference keyframe for tracking
                  keyframe *ref_keyfrm_ = nullptr;

                  // ORB scale pyramid information
                  //! number of scale levels
                  unsigned int num_scale_levels_;
                  //! scale factor
                  float scale_factor_;
                  //! log scale factor
                  float log_scale_factor_;
                  //! list of scale factors
                  std::vector<float> scale_factors_;
                  //! list of inverse of scale factors
                  std::vector<float> inv_scale_factors_;
                  //! list of sigma^2 (sigma=1.0 at scale=0) for optimization
                  std::vector<float> level_sigma_sq_;
                  //! list of 1 / sigma^2 for optimization
                  std::vector<float> inv_level_sigma_sq_;

                  //-----------------------------------------
                  // FW: for planar mapping module
                  cv::Mat _img_seg_mask; // passed to keyframe, for planar mapping
                  cv::Mat _depth_img;    // passed to keyframe, for dense reconstruction (demo)
                  cv::Mat _img_rgb;      // passed to keyframe, for assigning color to the dense point cloud (demo), this value is only used in monocular/rgb-d + segmentation (tracking_module.cc)

                  //-----------------------------------------
                  // FW: LSD/LBD extractor
                  feature::LineFeatureTracker *_line_extractor = nullptr; // passed from tracking_module
                  feature::LineFeatureTracker *_line_extractor_right = nullptr;

                  // FW: used for monocular / rgb-d / left of stereo
                  unsigned int _num_keylines = 0;                    // passed to keyframe
                  std::vector<cv::line_descriptor::KeyLine> _keylsd; // passed to keyframe
                  cv::Mat _lbd_descr;                                // passed to keyframe
                  std::vector<Vec3_t> _keyline_functions;            // passed to keyframe

                  // FW: used for right of stereo
                  std::vector<cv::line_descriptor::KeyLine> _keylsd_right; // passed to keyframe
                  cv::Mat _lbd_descr_right;                                // passed to keyframe
                  std::vector<Vec3_t> _keyline_functions_right;            // passed to keyframe
                  std::unordered_map<int, int> _good_matches_stereo;       // passed to keyframe

                  // FW:
                  std::vector<Line *> _landmarks_line;
                  std::vector<bool> _outlier_flags_line;

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

                  inline Mat33_t skew(const Vec3_t &t) const
                  {
                        Mat33_t S;
                        S << 0, -t.z(), t.y(), t.z(), 0, -t.x(), -t.y(), t.x(), 0;
                        return S;
                  }

            private:
                  //! enumeration to control the behavior of extract_orb()
                  enum class image_side
                  {
                        Left,
                        Right
                  };

                  /**
                   * Extract ORB feature according to img_size
                   * @param img
                   * @param mask
                   * @param img_side
                   */
                  void extract_orb(const cv::Mat &img, const cv::Mat &mask, const image_side &img_side = image_side::Left);

                  /**
                   * Compute disparities from depth information in depthmap
                   * @param right_img_depth
                   */
                  // FW: extended for keylines
                  void compute_stereo_from_depth(const cv::Mat &right_img_depth);

                  //-----------------------------------------
                  // FW: Monocular/RGB-D
                  void extract_line(const cv::Mat &img, std::vector<cv::line_descriptor::KeyLine> &keylsd,
                                    cv::Mat &lbd_descr, std::vector<Vec3_t> &keyline_functions);

                  // FW: Stereo
                  void extract_line_stereo(const cv::Mat &img, std::vector<cv::line_descriptor::KeyLine> &keylsd,
                                           cv::Mat &lbd_descr, std::vector<Vec3_t> &keyline_functions,
                                           const image_side &img_side);

                  //-----------------------------------------
                  //! Camera pose
                  //! rotation: world -> camera
                  Mat33_t rot_cw_;
                  //! translation: world -> camera
                  Vec3_t trans_cw_;
                  //! rotation: camera -> world
                  Mat33_t rot_wc_;
                  //! translation: camera -> world
                  Vec3_t cam_center_;
            };

      } // namespace data
} // namespace PLPSLAM

#endif // PLPSLAM_DATA_FRAME_H
