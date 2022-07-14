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

#ifndef PLPSLAM_DATA_CAMERA_DATABASE_H
#define PLPSLAM_DATA_CAMERA_DATABASE_H

#include <mutex>
#include <unordered_map>

#include <nlohmann/json_fwd.hpp>

namespace PLPSLAM
{

    namespace camera
    {
        class base;
    } // namespace camera

    namespace data
    {

        class camera_database
        {
        public:
            explicit camera_database(camera::base *curr_camera);

            ~camera_database();

            camera::base *get_camera(const std::string &camera_name) const;

            void from_json(const nlohmann::json &json_cameras);

            nlohmann::json to_json() const;

        private:
            //-----------------------------------------
            //! mutex to access the database
            mutable std::mutex mtx_database_;
            //! pointer to the camera which used in the current tracking
            //! (NOTE: the object is owned by config class,
            //!  thus this class does NOT delete the object of curr_camera_)
            camera::base *curr_camera_ = nullptr;
            //! database (key: camera name, value: pointer of camera::base)
            //! (NOTE: tracking camera must NOT be contained in the database)
            std::unordered_map<std::string, camera::base *> database_;
        };

    } // namespace data
} // namespace PLPSLAM

#endif // PLPSLAM_DATA_CAMERA_DATABASE_H
