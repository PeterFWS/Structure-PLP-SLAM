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

#ifndef PLPSLAM_IO_TRAJECTORY_IO_H
#define PLPSLAM_IO_TRAJECTORY_IO_H

#include <string>

namespace PLPSLAM
{

    namespace data
    {
        class map_database;
    } // namespace data

    namespace io
    {

        class trajectory_io
        {
        public:
            /**
             * Constructor
             */
            explicit trajectory_io(data::map_database *map_db);

            /**
             * Destructor
             */
            ~trajectory_io() = default;

            /**
             * Save the frame trajectory in the specified format
             */
            void save_frame_trajectory(const std::string &path, const std::string &format) const;

            /**
             * Save the keyframe trajectory in the specified format
             */
            void save_keyframe_trajectory(const std::string &path, const std::string &format) const;

        private:
            //! map_database
            data::map_database *const map_db_ = nullptr;
        };

    } // namespace io
} // namespace PLPSLAM

#endif // PLPSLAM_IO_TRAJECTORY_IO_H
