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

#ifndef PLPSLAM_OPTIMIZE_LOCAL_BUNDLE_ADJUSTER_H
#define PLPSLAM_OPTIMIZE_LOCAL_BUNDLE_ADJUSTER_H

namespace PLPSLAM
{

    namespace data
    {
        class keyframe;
        class map_database;
    } // namespace data

    namespace optimize
    {
        class local_bundle_adjuster
        {
        public:
            /**
             * Constructor
             * @param map_db
             * @param num_first_iter
             * @param num_second_iter
             */
            explicit local_bundle_adjuster(data::map_database *map_db,
                                           const unsigned int num_first_iter = 5,
                                           const unsigned int num_second_iter = 10);

            /**
             * Destructor
             */
            virtual ~local_bundle_adjuster() = default;

            /**
             * Perform optimization, local BA for keyframe
             * @param curr_keyfrm
             * @param force_stop_flag
             */
            void optimize(data::keyframe *curr_keyfrm, bool *const force_stop_flag) const;

        private:
            data::map_database *_map_db;

            //! number of iterations of first optimization
            const unsigned int num_first_iter_; // 5
            //! number of iterations of second optimization
            const unsigned int num_second_iter_; // 10
        };

    } // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_OPTIMIZE_LOCAL_BUNDLE_ADJUSTER_H
