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

#ifndef PLPSLAM_DATA_COMMON_H
#define PLPSLAM_DATA_COMMON_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/camera/base.h"

#include <opencv2/core.hpp>
#include <nlohmann/json_fwd.hpp>

// FW: for LSD ...
#include <opencv2/features2d.hpp>
#include "PLPSLAM/feature/line_descriptor/line_descriptor_custom.hpp"
#include "PLPSLAM/data/landmark_line.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace PLPSLAM
{
       namespace data
       {

              nlohmann::json convert_rotation_to_json(const Mat33_t &rot_cw);

              Mat33_t convert_json_to_rotation(const nlohmann::json &json_rot_cw);

              nlohmann::json convert_translation_to_json(const Vec3_t &trans_cw);

              Vec3_t convert_json_to_translation(const nlohmann::json &json_trans_cw);

              nlohmann::json convert_keypoints_to_json(const std::vector<cv::KeyPoint> &keypts);

              // FW:
              nlohmann::json convert_keylines_to_json(const std::vector<cv::line_descriptor::KeyLine> &keylines);

              std::vector<cv::KeyPoint> convert_json_to_keypoints(const nlohmann::json &json_keypts);

              // FW:
              std::vector<cv::line_descriptor::KeyLine> convert_json_to_keylines(const nlohmann::json &json_keylines);

              nlohmann::json convert_undistorted_to_json(const std::vector<cv::KeyPoint> &undist_keypts);

              std::vector<cv::KeyPoint> convert_json_to_undistorted(const nlohmann::json &json_undist_keypts, const std::vector<cv::KeyPoint> &keypts = {});

              nlohmann::json convert_descriptors_to_json(const cv::Mat &descriptors);

              // FW:
              nlohmann::json convert_lbd_descriptors_to_json(const cv::Mat &descriptors);

              cv::Mat convert_json_to_descriptors(const nlohmann::json &json_descriptors);

              // FW:
              cv::Mat convert_json_to_lbd_descriptors(const nlohmann::json &json_lbd_descriptors);

              /**
               * Assign all keypoints to cells to accelerate projection matching
               * @param camera
               * @param undist_keypts
               * @param keypt_indices_in_cells
               */
              void assign_keypoints_to_grid(camera::base *camera, const std::vector<cv::KeyPoint> &undist_keypts,
                                            std::vector<std::vector<std::vector<unsigned int>>> &keypt_indices_in_cells);

              /**
               * Assign all keypoints to cells to accelerate projection matching
               * @param camera
               * @param undist_keypts
               * @return
               */
              auto assign_keypoints_to_grid(camera::base *camera, const std::vector<cv::KeyPoint> &undist_keypts)
                  -> std::vector<std::vector<std::vector<unsigned int>>>;

              /**
               * Get x-y index of the cell in which the specified keypoint is assigned
               * @param camera
               * @param keypt
               * @param cell_idx_x
               * @param cell_idx_y
               * @return
               */
              inline bool get_cell_indices(camera::base *camera, const cv::KeyPoint &keypt, int &cell_idx_x, int &cell_idx_y)
              {
                     cell_idx_x = cvFloor((keypt.pt.x - camera->img_bounds_.min_x_) * camera->inv_cell_width_);
                     cell_idx_y = cvFloor((keypt.pt.y - camera->img_bounds_.min_y_) * camera->inv_cell_height_);
                     return (0 <= cell_idx_x && cell_idx_x < static_cast<int>(camera->num_grid_cols_) && 0 <= cell_idx_y && cell_idx_y < static_cast<int>(camera->num_grid_rows_));
              }

              /**
               * Get keypoint indices in cell(s) in which the specified point is located
               * @param camera
               * @param undist_keypts
               * @param keypt_indices_in_cells
               * @param ref_x
               * @param ref_y
               * @param margin
               * @param min_level
               * @param max_level
               * @return
               */
              std::vector<unsigned int> get_keypoints_in_cell(camera::base *camera, const std::vector<cv::KeyPoint> &undist_keypts,
                                                              const std::vector<std::vector<std::vector<unsigned int>>> &keypt_indices_in_cells,
                                                              const float ref_x, const float ref_y, const float margin,
                                                              const int min_level = -1, const int max_level = -1);

              // FW:
              std::vector<unsigned int> get_keylines_in_cell(const std::vector<cv::line_descriptor::KeyLine> &keylines,
                                                             const float ref_x1, const float ref_y1,
                                                             const float ref_x2, const float ref_y2,
                                                             const float margin,
                                                             const int min_level = -1, const int max_level = -1);

       } // namespace data
} // namespace PLPSLAM

#endif // PLPSLAM_DATA_COMMON_H
