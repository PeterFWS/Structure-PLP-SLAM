/**
 * This file is part of Structure PLP-SLAM.
 *
 * Copyright 2022 DFKI (German Research Center for Artificial Intelligence)
 * Developed by Fangwen Shu <Fangwen.Shu@dfki.de>
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

#ifndef PLPSLAM_FEATURE_LINE_EXTRACTOR_H
#define PLPSLAM_FEATURE_LINE_EXTRACTOR_H

#include <iostream>
#include <queue>
#include <memory>

#include <opencv2/features2d.hpp>
#include "PLPSLAM/feature/line_descriptor/line_descriptor_custom.hpp"
#include "PLPSLAM/camera/base.h"
#include "PLPSLAM/camera/perspective.h"
#include <eigen3/Eigen/Dense>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "PLPSLAM/type.h"

#include "PLPSLAM/data/landmark_line.h"

namespace PLPSLAM
{

    namespace camera
    {
        class base;
    } // namespace camera

    namespace data
    {
        class Line;
        // class FrameLines;
    }

    namespace feature
    {
        // FW: The purpose of this class is for extracting LSD and LBD from a given image
        class LineFeatureTracker
        {
        public:
            // Constructor
            LineFeatureTracker(camera::base *camera);

            // Get _undist_map1, _undist_map2, _K for un-distortion
            void readIntrinsicParameter(const cv::Mat &img);

            // Extracting line segments and calculate LBD, per frame
            void extract_LSD_LBD(const cv::Mat &img,
                                 std::vector<cv::line_descriptor::KeyLine> &frame_keylsd,
                                 cv::Mat &frame_lbd_descr,
                                 std::vector<Vec3_t> &keyline_functions);

            void initialize();

            unsigned int get_num_scale_levels() const;
            float get_scale_factor() const;
            std::vector<float> get_scale_factors() const;
            std::vector<float> get_inv_scale_factors() const;
            std::vector<float> get_level_sigma_sq() const;
            std::vector<float> get_inv_level_sigma_sq() const;

        private:
            camera::base *_camera = nullptr; // initialized within constructor
            int _frame_cnt;                  // initialized within constructor

            // used for un-distortion of image
            cv::Mat _undist_map1, _undist_map2, _K;

            // apply histogram equalization on the image before extracting LSD
            bool EQUALIZE = false;

            // image pyramid parameters
            unsigned int _num_levels;
            float _scale_factor;
            std::vector<float> _scale_factors;
            std::vector<float> _inv_scale_factors;
            std::vector<float> _level_sigma_sq;
            std::vector<float> _inv_level_sigma_sq;

            double _min_line_length = 0.125; // line segments shorter than that are rejected
        };

    }
}

#endif