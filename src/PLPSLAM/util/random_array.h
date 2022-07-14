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

#ifndef PLPSLAM_UTIL_RANDOM_ARRAY_H
#define PLPSLAM_UTIL_RANDOM_ARRAY_H

#include <vector>
#include <random>

namespace PLPSLAM
{
    namespace util
    {

        std::mt19937 create_random_engine();

        template <typename T>
        std::vector<T> create_random_array(const size_t size, const T rand_min, const T rand_max);

    } // namespace util
} // namespace PLPSLAM

#endif // PLPSLAM_UTIL_RANDOM_ARRAY_H
