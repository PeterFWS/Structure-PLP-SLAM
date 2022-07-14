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

#ifndef PLPSLAM_FEATURE_ORB_PARAMS_H
#define PLPSLAM_FEATURE_ORB_PARAMS_H

#include <yaml-cpp/yaml.h>

namespace PLPSLAM
{
    namespace feature
    {

        struct orb_params
        {
            orb_params() = default;

            //! Constructor
            orb_params(const unsigned int max_num_keypts, const float scale_factor, const unsigned int num_levels,
                       const unsigned int ini_fast_thr, const unsigned int min_fast_thr,
                       const std::vector<std::vector<float>> &mask_rects = {});

            //! Constructor
            explicit orb_params(const YAML::Node &yaml_node);

            //! Destructor
            virtual ~orb_params() = default;

            //! Dump parameter values to the standard output
            void show_parameters() const;

            unsigned int max_num_keypts_ = 2000; // number of features wanted to detect
            float scale_factor_ = 1.2;           // the scaling factor of the image pyramid
            unsigned int num_levels_ = 8;        // specify the image pyramid layer for which feature points need to be extracted
            unsigned int ini_fast_thr_ = 20;     // initial default FAST response value threshold
            unsigned int min_fast_thr = 7;       // a smaller FAST response threshold

            //! A vector of keypoint area represents mask area
            //! Each areas are denoted as form of [x_min / cols, x_max / cols, y_min / rows, y_max / rows]
            std::vector<std::vector<float>> mask_rects_;

            //! Calculate scale factors
            static std::vector<float> calc_scale_factors(const unsigned int num_scale_levels, const float scale_factor);

            //! Calculate inverses of scale factors
            static std::vector<float> calc_inv_scale_factors(const unsigned int num_scale_levels, const float scale_factor);

            //! Calculate squared sigmas at all levels
            static std::vector<float> calc_level_sigma_sq(const unsigned int num_scale_levels, const float scale_factor);

            //! Calculate inverses of squared sigmas at all levels
            static std::vector<float> calc_inv_level_sigma_sq(const unsigned int num_scale_levels, const float scale_factor);
        };

    } // namespace feature
} // namespace PLPSLAM

#endif // PLPSLAM_FEATURE_ORB_PARAMS_H
