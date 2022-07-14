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

#ifndef PLPSLAM_MATCH_BASE_H
#define PLPSLAM_MATCH_BASE_H

#include <array>
#include <algorithm>
#include <numeric>

#include <opencv2/core.hpp>

namespace PLPSLAM
{
    namespace match
    {

        static constexpr unsigned int HAMMING_DIST_THR_LOW = 50;
        static constexpr unsigned int HAMMING_DIST_THR_HIGH = 100;
        static constexpr unsigned int MAX_HAMMING_DIST = 256;

        //! Calculate the Hamming distance between ORB features
        inline unsigned int compute_descriptor_distance_32(const cv::Mat &desc_1, const cv::Mat &desc_2)
        {
            // FW: Counting bits set, in parallel:
            //  http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

            constexpr uint32_t mask_1 = 0x55555555U;
            constexpr uint32_t mask_2 = 0x33333333U;
            constexpr uint32_t mask_3 = 0x0F0F0F0FU;
            constexpr uint32_t mask_4 = 0x01010101U;

            const auto *pa = desc_1.ptr<uint32_t>();
            const auto *pb = desc_2.ptr<uint32_t>();

            unsigned int dist = 0;

            for (unsigned int i = 0; i < 8; ++i, ++pa, ++pb)
            {
                auto v = *pa ^ *pb;
                v -= ((v >> 1) & mask_1);
                v = (v & mask_2) + ((v >> 2) & mask_2);
                dist += (((v + (v >> 4)) & mask_3) * mask_4) >> 24;
            }

            return dist;
        }

        //! Calculate the Hamming distance between ORB features
        inline unsigned int compute_descriptor_distance_64(const cv::Mat &desc_1, const cv::Mat &desc_2)
        {
            // https://stackoverflow.com/questions/21826292/t-sql-hamming-distance-function-capable-of-decimal-string-uint64?lq=1

            constexpr uint64_t mask_1 = 0x5555555555555555UL;
            constexpr uint64_t mask_2 = 0x3333333333333333UL;
            constexpr uint64_t mask_3 = 0x0F0F0F0F0F0F0F0FUL;
            constexpr uint64_t mask_4 = 0x0101010101010101UL;

            const auto *pa = desc_1.ptr<uint64_t>();
            const auto *pb = desc_2.ptr<uint64_t>();

            unsigned int dist = 0;

            for (unsigned int i = 0; i < 4; ++i, ++pa, ++pb)
            {
                auto v = *pa ^ *pb;
                v -= (v >> 1) & mask_1;
                v = (v & mask_2) + ((v >> 2) & mask_2);
                dist += (((v + (v >> 4)) & mask_3) * mask_4) >> 56;
            }

            return dist;
        }

        class base
        {
        public:
            base(const float lowe_ratio, const bool check_orientation)
                : lowe_ratio_(lowe_ratio), check_orientation_(check_orientation) {}

            virtual ~base() = default;

        protected:
            const float lowe_ratio_;
            const bool check_orientation_;
        };

    } // namespace match
} // namespace PLPSLAM

#endif // PLPSLAM_MATCH_BASE_H
