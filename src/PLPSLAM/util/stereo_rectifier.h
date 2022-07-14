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

#ifndef PLPSLAM_UTIL_STEREO_RECTIFIER_H
#define PLPSLAM_UTIL_STEREO_RECTIFIER_H

#include "PLPSLAM/config.h"
#include "PLPSLAM/camera/base.h"

#include <memory>

namespace PLPSLAM
{
    namespace util
    {

        class stereo_rectifier
        {
        public:
            //! Constructor
            explicit stereo_rectifier(const std::shared_ptr<config> &cfg);

            //! Constructor
            stereo_rectifier(camera::base *camera, const YAML::Node &yaml_node);

            //! Destructor
            virtual ~stereo_rectifier();

            //! Apply stereo-rectification
            void rectify(const cv::Mat &in_img_l, const cv::Mat &in_img_r,
                         cv::Mat &out_img_l, cv::Mat &out_img_r) const;

        private:
            //! Parse std::vector as cv::Mat
            static cv::Mat parse_vector_as_mat(const cv::Size &shape, const std::vector<double> &vec);

            //! Load model type before rectification from YAML
            static camera::model_type_t load_model_type(const YAML::Node &yaml_node);

            //! camera model type before rectification
            const camera::model_type_t model_type_;

            //! undistortion map for x-axis in left image
            cv::Mat undist_map_x_l_;
            //! undistortion map for y-axis in left image
            cv::Mat undist_map_y_l_;
            //! undistortion map for x-axis in right image
            cv::Mat undist_map_x_r_;
            //! undistortion map for y-axis in right image
            cv::Mat undist_map_y_r_;
        };

    } // namespace util
} // namespace PLPSLAM

#endif // PLPSLAM_UTIL_STEREO_RECTIFIER_H
