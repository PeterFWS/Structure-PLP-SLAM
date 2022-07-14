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

#ifndef PLPSLAM_PUBLISH_FRAME_PUBLISHER_H
#define PLPSLAM_PUBLISH_FRAME_PUBLISHER_H

#include "PLPSLAM/config.h"
#include "PLPSLAM/tracking_module.h"

#include <mutex>
#include <vector>

#include <opencv2/core/core.hpp>

// FW: for LSD ...
#include <opencv2/features2d.hpp>
#include "PLPSLAM/feature/line_descriptor/line_descriptor_custom.hpp"
#include "PLPSLAM/data/landmark_line.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace PLPSLAM
{

    class tracking_module;

    namespace data
    {
        class map_database;
    } // namespace data

    namespace publish
    {

        class frame_publisher
        {
        public:
            /**
             * Constructor
             */
            frame_publisher(const std::shared_ptr<config> &cfg, data::map_database *map_db,
                            const unsigned int img_width = 1024);

            /**
             * Destructor
             */
            virtual ~frame_publisher();

            /**
             * Update tracking information
             * NOTE: should be accessed from system thread
             */
            void update(tracking_module *tracker);

            /**
             * Get the current image with tracking information
             * NOTE: should be accessed from viewer thread
             */
            cv::Mat draw_frame(const bool draw_text = true);

            // FW: functionality for segmentation input
            cv::Mat draw_seg_mask();

        protected:
            unsigned int draw_initial_points(cv::Mat &img, const std::vector<cv::KeyPoint> &init_keypts,
                                             const std::vector<int> &init_matches, const std::vector<cv::KeyPoint> &curr_keypts,
                                             const float mag = 1.0) const;

            unsigned int draw_tracked_points(cv::Mat &img, const std::vector<cv::KeyPoint> &curr_keypts,
                                             const std::vector<bool> &is_tracked, const bool mapping_is_enabled,
                                             const float mag = 1.0) const;

            // FW:
            unsigned int draw_tracked_lines(cv::Mat &img, const std::vector<cv::line_descriptor::KeyLine> &curr_keylines,
                                            const std::vector<bool> &is_tracked_line, const bool mapping_is_enabled,
                                            const float mag = 1.0) const;

            void draw_info_text(cv::Mat &img, const tracker_state_t tracking_state, const unsigned int num_tracked, const unsigned int num_tracked_line,
                                const double elapsed_ms, const bool mapping_is_enabled) const;

            // colors (BGR)
            const cv::Scalar mapping_color_{0, 255, 255};
            const cv::Scalar localization_color_{255, 255, 0};

            //! config
            std::shared_ptr<config> cfg_;
            //! map database
            data::map_database *map_db_;
            //! maximum size of output images
            const int img_width_;

            // -------------------------------------------
            //! mutex to access variables below
            std::mutex mtx_;

            //! raw img
            cv::Mat img_;
            //! tracking state
            tracker_state_t tracking_state_;

            //! initial keypoints
            std::vector<cv::KeyPoint> init_keypts_;
            //! matching between initial frame and current frame
            std::vector<int> init_matches_;

            //! current keypoints
            std::vector<cv::KeyPoint> curr_keypts_;

            // FW:
            std::vector<cv::line_descriptor::KeyLine> _curr_keylines;

            //! elapsed time for tracking
            double elapsed_ms_ = 0.0;

            //! mapping module status
            bool mapping_is_enabled_;

            //! tracking flag for each current keypoint
            std::vector<bool> is_tracked_;

            // FW:
            std::vector<bool> _is_tracked_line;

            // FW: flag of segmentation input
            cv::Mat _seg_mask_img;
        };

    } // namespace publish
} // namespace PLPSLAM

#endif // PLPSLAM_PUBLISH_FRAME_PUBLISHER_H
