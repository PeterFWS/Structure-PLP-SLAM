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

#ifndef PLPSLAM_UTIL_FANCY_INDEX_H
#define PLPSLAM_UTIL_FANCY_INDEX_H

#include "PLPSLAM/type.h"

#include <vector>
#include <type_traits>

namespace PLPSLAM
{
    namespace util
    {

        template <typename T, typename U>
        std::vector<T> resample_by_indices(const std::vector<T> &elements, const std::vector<U> &indices)
        {
            static_assert(std::is_integral<U>(), "the element type of indices must be integer");

            std::vector<T> resampled;
            resampled.reserve(elements.size());
            for (const auto idx : indices)
            {
                resampled.push_back(elements.at(idx));
            }

            return resampled;
        }

        template <typename T, typename U>
        eigen_alloc_vector<T> resample_by_indices(const eigen_alloc_vector<T> &elements, const std::vector<U> &indices)
        {
            static_assert(std::is_integral<U>(), "the element type of indices must be integer");

            eigen_alloc_vector<T> resampled;
            resampled.reserve(elements.size());
            for (const auto idx : indices)
            {
                resampled.push_back(elements.at(idx));
            }

            return resampled;
        }

        template <typename T>
        std::vector<T> resample_by_indices(const std::vector<T> &elements, const std::vector<bool> &indices)
        {
            assert(elements.size() == indices.size());

            std::vector<T> resampled;
            resampled.reserve(elements.size());
            for (unsigned int idx = 0; idx < elements.size(); ++idx)
            {
                if (indices.at(idx))
                {
                    resampled.push_back(elements.at(idx));
                }
            }

            return resampled;
        }

        template <typename T>
        eigen_alloc_vector<T> resample_by_indices(const eigen_alloc_vector<T> &elements, const std::vector<bool> &indices)
        {
            assert(elements.size() == indices.size());

            eigen_alloc_vector<T> resampled;
            resampled.reserve(elements.size());
            for (unsigned int idx = 0; idx < elements.size(); ++idx)
            {
                if (indices.at(idx))
                {
                    resampled.push_back(elements.at(idx));
                }
            }

            return resampled;
        }

    } // namespace util
} // namespace PLPSLAM

#endif // PLPSLAM_UTIL_FANCY_INDEX_H
