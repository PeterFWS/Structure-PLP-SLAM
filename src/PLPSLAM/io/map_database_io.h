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

#ifndef PLPSLAM_IO_MAP_DATABASE_IO_H
#define PLPSLAM_IO_MAP_DATABASE_IO_H

#include "PLPSLAM/data/bow_vocabulary.h"

#include <string>

namespace PLPSLAM
{

    namespace data
    {
        class camera_database;
        class bow_database;
        class map_database;
    } // namespace data

    namespace io
    {

        class map_database_io
        {
        public:
            /**
             * Constructor
             */
            map_database_io(data::camera_database *cam_db, data::map_database *map_db,
                            data::bow_database *bow_db, data::bow_vocabulary *bow_vocab);

            /**
             * Destructor
             */
            ~map_database_io() = default;

            /**
             * Save the map database as MessagePack
             */
            void save_message_pack(const std::string &path);

            /**
             * Load the map database from MessagePack
             */
            void load_message_pack(const std::string &path);

        private:
            //! camera database
            data::camera_database *const cam_db_ = nullptr;
            //! map_database
            data::map_database *const map_db_ = nullptr;
            //! BoW database
            data::bow_database *const bow_db_ = nullptr;
            //! BoW vocabulary
            data::bow_vocabulary *const bow_vocab_ = nullptr;
        };

    } // namespace io
} // namespace PLPSLAM

#endif // PLPSLAM_IO_MAP_DATABASE_IO_H
