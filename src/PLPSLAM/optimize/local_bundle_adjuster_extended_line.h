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

#ifndef PLPSLAM_OPTIMIZE_LOCAL_BUNDLE_ADJUSTER_EXTENDED_LINE_H
#define PLPSLAM_OPTIMIZE_LOCAL_BUNDLE_ADJUSTER_EXTENDED_LINE_H

#include "PLPSLAM/type.h"

namespace PLPSLAM
{
    namespace data
    {
        class keyframe;
        class map_database;
        class Line; // FW:
    }

    namespace optimize
    {
        class local_bundle_adjuster_extended_line
        {
        public:
            explicit local_bundle_adjuster_extended_line(data::map_database *map_db,
                                                         const unsigned int num_first_iter = 5,
                                                         const unsigned int num_second_iter = 10);

            virtual ~local_bundle_adjuster_extended_line() = default;

            void optimize(data::keyframe *curr_keyfrm, bool *const force_stop_flag) const;

            // FW:
            inline Mat33_t skew(const Vec3_t &t) const
            {
                Mat33_t S;
                S << 0, -t.z(), t.y(), t.z(), 0, -t.x(), -t.y(), t.x(), 0;
                return S;
            }

            // FW: re-estimate two endpoints for visualization 3D line in the map
            bool endpoint_trimming(data::Line *local_lm_line,
                                   const Vec6_t &plucker_coord,
                                   Vec6_t &updated_pose_w) const;

        private:
            data::map_database *_map_db;

            //! number of iterations of first optimization
            const unsigned int num_first_iter_; // 5
            //! number of iterations of second optimization
            const unsigned int num_second_iter_; // 10

            // FW: print Debug info in the terminal
            const bool _setVerbose = false;
        };

    } // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_OPTIMIZE_LOCAL_BUNDLE_ADJUSTER_H
