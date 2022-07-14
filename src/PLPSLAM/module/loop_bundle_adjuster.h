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

#ifndef PLPSLAM_MODULE_LOOP_BUNDLE_ADJUSTER_H
#define PLPSLAM_MODULE_LOOP_BUNDLE_ADJUSTER_H

namespace PLPSLAM
{

       class mapping_module;

       namespace data
       {
              class map_database;
       } // namespace data

       namespace module
       {

              class loop_bundle_adjuster
              {
              public:
                     /**
                      * Constructor
                      */
                     explicit loop_bundle_adjuster(data::map_database *map_db, const unsigned int num_iter = 10);

                     /**
                      * Destructor
                      */
                     ~loop_bundle_adjuster() = default;

                     /**
                      * Set the mapping module
                      */
                     void set_mapping_module(mapping_module *mapper);

                     /**
                      * Count the number of loop BA execution
                      */
                     void count_loop_BA_execution();

                     /**
                      * Abort loop BA externally
                      */
                     void abort();

                     /**
                      * Loop BA is running or not
                      */
                     bool is_running() const;

                     /**
                      * Run loop BA
                      */
                     void optimize(const unsigned int identifier);

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
                     data::map_database *map_db_ = nullptr;

                     //! mapping module
                     mapping_module *mapper_ = nullptr;

                     //! number of iteration for optimization
                     const unsigned int num_iter_ = 10;

                     //-----------------------------------------
                     // thread management

                     //! mutex for access to pause procedure
                     mutable std::mutex mtx_thread_;

                     //! number of times loop BA is performed
                     unsigned int num_exec_loop_BA_ = 0;

                     //! flag to abort loop BA
                     bool abort_loop_BA_ = false;

                     //! flag which indicates loop BA is running or not
                     bool loop_BA_is_running_ = false;
              };

       } // namespace module
} // namespace PLPSLAM

#endif // PLPSLAM_MODULE_LOOP_BUNDLE_ADJUSTER_H
