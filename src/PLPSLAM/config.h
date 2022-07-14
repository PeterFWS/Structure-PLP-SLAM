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

#ifndef PLPSLAM_CONFIG_H
#define PLPSLAM_CONFIG_H

#include "PLPSLAM/camera/base.h"
#include "PLPSLAM/feature/orb_params.h"

#include <yaml-cpp/yaml.h>

namespace PLPSLAM
{

    class config
    {
    public:
        //! Constructor
        explicit config(const std::string &config_file_path);
        explicit config(const YAML::Node &yaml_node, const std::string &config_file_path = "");

        //! Destructor
        ~config();

        friend std::ostream &operator<<(std::ostream &os, const config &cfg);

        //! path to config YAML file
        const std::string config_file_path_;

        //! YAML node
        const YAML::Node yaml_node_;

        //! Camera model
        camera::base *camera_ = nullptr;

        //! ORB feature parameters
        feature::orb_params orb_params_;

        //! depth threshold
        double true_depth_thr_ = 40.0;

        //! depthmap factor (pixel_value / depthmap_factor = true_depth)
        double depthmap_factor_ = 1.0;
    };

} // namespace PLPSLAM

#endif // PLPSLAM_CONFIG_H
