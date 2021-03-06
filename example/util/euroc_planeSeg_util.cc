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

#include "euroc_planeSeg_util.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <algorithm>

euroc_sequence::euroc_sequence(const std::string &seq_dir_path)
{
    const std::string timestamp_file_path = seq_dir_path + "/cam0/data.csv";
    const std::string left_img_dir_path = seq_dir_path + "/cam0/data/";
    const std::string right_img_dir_path = seq_dir_path + "/cam1/data/";

    // FW:
    const std::string mask_img_dir_path = seq_dir_path + "/cam0/seg/";

    timestamps_.clear();
    left_img_file_paths_.clear();
    right_img_file_paths_.clear();

    // FW:
    _mask_img_file_paths.clear();

    // load timestamps
    std::ifstream ifs_timestamp;
    ifs_timestamp.open(timestamp_file_path.c_str());
    if (!ifs_timestamp)
    {
        throw std::runtime_error("Could not load a timestamp file from " + timestamp_file_path);
    }

    // load header row
    std::string s;
    getline(ifs_timestamp, s);

    while (!ifs_timestamp.eof())
    {
        getline(ifs_timestamp, s);
        std::replace(s.begin(), s.end(), ',', ' ');
        if (!s.empty())
        {
            std::stringstream ss;
            ss << s;
            unsigned long long timestamp;
            ss >> timestamp;
            timestamps_.push_back(timestamp / static_cast<double>(1E9));
            left_img_file_paths_.push_back(left_img_dir_path + std::to_string(timestamp) + ".png");
            right_img_file_paths_.push_back(right_img_dir_path + std::to_string(timestamp) + ".png");

            // FW:
            _mask_img_file_paths.push_back(mask_img_dir_path + std::to_string(timestamp) + ".png");
        }
    }

    ifs_timestamp.close();
}

std::vector<euroc_sequence::frame> euroc_sequence::get_frames() const
{
    std::vector<frame> frames;
    assert(timestamps_.size() == left_img_file_paths_.size());
    assert(timestamps_.size() == right_img_file_paths_.size());
    assert(left_img_file_paths_.size() == right_img_file_paths_.size());

    // FW:
    assert(timestamps_.size() == _mask_img_file_paths.size());

    for (unsigned int i = 0; i < timestamps_.size(); ++i)
    {
        frames.emplace_back(frame{left_img_file_paths_.at(i), right_img_file_paths_.at(i), timestamps_.at(i), _mask_img_file_paths.at(i)});
    }
    return frames;
}
