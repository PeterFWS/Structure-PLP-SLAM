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

#ifndef PLPSLAM_UTIL_STRING_H
#define PLPSLAM_UTIL_STRING_H

#include <vector>
#include <string>
#include <sstream>

namespace PLPSLAM
{
    namespace util
    {

        inline std::vector<std::string> split_string(const std::string &str, const char del)
        {
            std::vector<std::string> splitted_strs;
            std::stringstream ss(str);
            std::string item;
            while (std::getline(ss, item, del))
            {
                if (!item.empty())
                {
                    splitted_strs.push_back(item);
                }
            }
            return splitted_strs;
        }

        inline bool string_startswith(const std::string &str, const std::string &qry)
        {
            return str.size() >= qry.size() && std::equal(std::begin(qry), std::end(qry), std::begin(str));
        }

    } // namespace util
} // namespace PLPSLAM

#endif // PLPSLAM_UTIL_STRING_H
