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

#include "PLPSLAM/util/image_converter.h"

#include <opencv2/imgproc.hpp>

namespace PLPSLAM
{
    namespace util
    {

        void convert_to_grayscale(cv::Mat &img, const camera::color_order_t in_color_order)
        {
            if (img.channels() == 3)
            {
                switch (in_color_order)
                {
                case camera::color_order_t::Gray:
                {
                    break;
                }
                case camera::color_order_t::RGB:
                {
                    cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
                    break;
                }
                case camera::color_order_t::BGR:
                {
                    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
                    break;
                }
                }
            }
            else if (img.channels() == 4)
            {
                switch (in_color_order)
                {
                case camera::color_order_t::Gray:
                {
                    break;
                }
                case camera::color_order_t::RGB:
                {
                    cv::cvtColor(img, img, cv::COLOR_RGBA2GRAY);
                    break;
                }
                case camera::color_order_t::BGR:
                {
                    cv::cvtColor(img, img, cv::COLOR_BGRA2GRAY);
                    break;
                }
                }
            }
        }

        void convert_to_true_depth(cv::Mat &img, const double depthmap_factor)
        {
            img.convertTo(img, CV_32F, 1.0 / depthmap_factor);
        }

        void equalize_histogram(cv::Mat &img)
        {
            assert(img.type() == CV_8UC1 || img.type() == CV_16UC1);
            if (img.type() == CV_16UC1)
            {
                std::vector<unsigned short> vec(img.begin<unsigned short>(), img.end<unsigned short>());
                std::sort(vec.begin(), vec.end());
                const auto l = vec.at(static_cast<unsigned int>(0.05 * vec.size()));
                const auto h = vec.at(static_cast<unsigned int>(0.95 * vec.size()));
                img.convertTo(img, CV_8UC1, 255.0 / (h - l), -255.0 * l / (h - l)); // 255*(img-l)/(h-l)
            }
            else if (img.type() == CV_8UC1)
            {
                std::vector<unsigned char> vec(img.begin<unsigned char>(), img.end<unsigned char>());
                std::sort(vec.begin(), vec.end());
                const auto l = vec.at(static_cast<unsigned int>(0.05 * vec.size()));
                const auto h = vec.at(static_cast<unsigned int>(0.95 * vec.size()));
                img.convertTo(img, CV_8UC1, 255.0 / (h - l), -255.0 * l / (h - l)); // 255*(img-l)/(h-l)
            }
        }

    } // namespace util
} // namespace PLPSLAM
