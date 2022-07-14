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

#ifndef PLPSLAM_UTIL_TRIGONOMETRIC_H
#define PLPSLAM_UTIL_TRIGONOMETRIC_H

#include <cmath>

#include <opencv2/core/fast_math.hpp>

namespace PLPSLAM
{
    namespace util
    {

        static constexpr float _PI = 3.14159265358979f;
        static constexpr float _PI_2 = _PI / 2.0f;
        static constexpr float _TWO_PI = 2.0f * _PI;
        static constexpr float _INV_TWO_PI = 1.0f / _TWO_PI;
        static constexpr float _THREE_PI_2 = 3.0f * _PI_2;

        inline float _cos(float v)
        {
            constexpr float c1 = 0.99940307f;
            constexpr float c2 = -0.49558072f;
            constexpr float c3 = 0.03679168f;

            const float v2 = v * v;
            return c1 + v2 * (c2 + c3 * v2);
        }

        inline float cos(float v)
        {
            v = v - cvFloor(v * _INV_TWO_PI) * _TWO_PI;
            v = (0.0f < v) ? v : -v;

            if (v < _PI_2)
            {
                return _cos(v);
            }
            else if (v < _PI)
            {
                return -_cos(_PI - v);
            }
            else if (v < _THREE_PI_2)
            {
                return -_cos(v - _PI);
            }
            else
            {
                return _cos(_TWO_PI - v);
            }
        }

        inline float sin(float v)
        {
            return PLPSLAM::util::cos(_PI_2 - v);
        }

    } // namespace util
} // namespace PLPSLAM

#endif // PLPSLAM_UTIL_TRIGONOMETRIC_H
