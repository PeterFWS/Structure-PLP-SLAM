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

#ifndef PLPSLAM_OPTIMIZE_GLOBAL_BUNDLE_ADJUSTER_H
#define PLPSLAM_OPTIMIZE_GLOBAL_BUNDLE_ADJUSTER_H

#include "PLPSLAM/type.h"

namespace PLPSLAM
{

    namespace data
    {
        class map_database;
        class Line; // FW:
    }

    namespace optimize
    {

        class global_bundle_adjuster
        {
        public:
            /**
             * Constructor
             * @param map_db
             * @param num_iter
             * @param use_huber_kernel
             */
            explicit global_bundle_adjuster(data::map_database *map_db,
                                            const unsigned int num_iter = 10,
                                            const bool use_huber_kernel = true);

            /**
             * Destructor
             */
            virtual ~global_bundle_adjuster() = default;

            /**
             * Perform optimization
             * @param lead_keyfrm_id_in_global_BA
             * @param force_stop_flag
             */
            void optimize(const unsigned int lead_keyfrm_id_in_global_BA = 0, bool *const force_stop_flag = nullptr) const;

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
            //! map database
            const data::map_database *map_db_;

            //! number of iterations of optimization
            unsigned int num_iter_;

            //! use Huber loss or not
            const bool use_huber_kernel_;
        };

    } // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_OPTIMIZE_GLOBAL_BUNDLE_ADJUSTER_H
