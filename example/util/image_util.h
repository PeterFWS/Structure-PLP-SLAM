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

#ifndef EXAMPLE_UTIL_IMAGE_UTIL_H
#define EXAMPLE_UTIL_IMAGE_UTIL_H

#include <string>
#include <vector>

class image_sequence
{
public:
    struct frame
    {
        frame(const std::string &img_path, const double timestamp)
            : img_path_(img_path), timestamp_(timestamp){};

        const std::string img_path_;
        const double timestamp_;
    };

    image_sequence(const std::string &img_dir_path, const double fps);

    virtual ~image_sequence() = default;

    std::vector<frame> get_frames() const;

private:
    const double fps_;

    std::vector<std::string> img_file_paths_;
};

#endif // EXAMPLE_UTIL_IMAGE_UTIL_H
