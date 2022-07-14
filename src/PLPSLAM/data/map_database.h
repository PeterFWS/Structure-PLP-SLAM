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

#ifndef PLPSLAM_DATA_MAP_DATABASE_H
#define PLPSLAM_DATA_MAP_DATABASE_H

#include "PLPSLAM/data/bow_vocabulary.h"
#include "PLPSLAM/data/frame_statistics.h"

#include <mutex>
#include <vector>
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

              class frame;
              class keyframe;
              class landmark;
              class camera_database;
              class bow_database;

              // FW: extended semantic structure for plane
              class Plane;

              // FW: extended semantic structure for line segment
              class Line;

              class map_database
              {
              public:
                     /**
                      * Constructor
                      */
                     map_database();

                     /**
                      * Destructor
                      */
                     ~map_database();

                     /**
                      * Add keyframe to the database
                      * @param keyfrm
                      */
                     void add_keyframe(keyframe *keyfrm);

                     /**
                      * Erase keyframe from the database
                      * @param keyfrm
                      */
                     void erase_keyframe(keyframe *keyfrm);

                     /**
                      * Add landmark to the database
                      * @param lm
                      */
                     void add_landmark(landmark *lm);

                     /**
                      * Erase landmark from the database
                      * @param lm
                      */
                     void erase_landmark(landmark *lm);

                     /**
                      * Set local landmarks
                      * @param local_lms
                      */
                     void set_local_landmarks(const std::vector<landmark *> &local_lms);

                     /**
                      * Get local landmarks
                      * @return
                      */
                     std::vector<landmark *> get_local_landmarks() const;

                     /**
                      * Get all of the keyframes in the database
                      * @return
                      */
                     std::vector<keyframe *> get_all_keyframes() const;

                     /**
                      * Get the number of keyframes
                      * @return
                      */
                     unsigned get_num_keyframes() const;

                     /**
                      * Get all of the landmarks in the database
                      * @return
                      */
                     std::vector<landmark *> get_all_landmarks() const;

                     /**
                      * Get the number of landmarks
                      * @return
                      */
                     unsigned int get_num_landmarks() const;

                     /**
                      * Get the maximum keyframe ID
                      * @return
                      */
                     unsigned int get_max_keyframe_id() const;

                     /**
                      * Update frame statistics
                      * @param frm
                      * @param is_lost
                      */
                     void update_frame_statistics(const data::frame &frm, const bool is_lost)
                     {
                            std::lock_guard<std::mutex> lock(mtx_map_access_);
                            frm_stats_.update_frame_statistics(frm, is_lost);
                     }

                     /**
                      * Replace a keyframe which will be erased in frame statistics
                      * @param old_keyfrm
                      * @param new_keyfrm
                      */
                     void replace_reference_keyframe(data::keyframe *old_keyfrm, data::keyframe *new_keyfrm)
                     {
                            std::lock_guard<std::mutex> lock(mtx_map_access_);
                            frm_stats_.replace_reference_keyframe(old_keyfrm, new_keyfrm);
                     }

                     /**
                      * Get frame statistics
                      * @return
                      */
                     frame_statistics get_frame_statistics() const
                     {
                            std::lock_guard<std::mutex> lock(mtx_map_access_);
                            return frm_stats_;
                     }

                     /**
                      * Clear the database
                      */
                     void clear();

                     /**
                      * Load keyframes and landmarks from JSON
                      * @param cam_db
                      * @param bow_vocab
                      * @param bow_db
                      * @param json_keyfrms
                      * @param json_landmarks
                      */
                     void from_json(camera_database *cam_db, bow_vocabulary *bow_vocab, bow_database *bow_db,
                                    const nlohmann::json &json_keyfrms, const nlohmann::json &json_landmarks);

                     // FW: load point-line map
                     void from_json(camera_database *cam_db, bow_vocabulary *bow_vocab, bow_database *bow_db,
                                    const nlohmann::json &json_keyfrms, const nlohmann::json &json_landmarks, const nlohmann::json &json_landmarks_line);

                     /**
                      * Dump keyframes and landmarks as JSON
                      * @param json_keyfrms
                      * @param json_landmarks
                      */
                     void to_json(nlohmann::json &json_keyfrms, nlohmann::json &json_landmarks);

                     // FW: save point-line map
                     void to_json(nlohmann::json &json_keyfrms, nlohmann::json &json_landmarks, nlohmann::json &json_landmarks_line);

                     //! origin keyframe
                     keyframe *origin_keyfrm_ = nullptr;

                     //! mutex for locking ALL access to the database
                     //! (NOTE: cannot used in map_database class)
                     static std::mutex mtx_database_;

                     //-----------------------------------------
                     // FW: some functionalities for plane
                     bool _b_seg_or_not = false;                                                              // global flag which indicates semantic SLAM is running
                     void add_landmark_plane(Plane *pl);                                                      // add plane to unordered_map
                     std::vector<Plane *> get_all_landmark_planes() const;                                    // return a vector, not unordered_map
                     std::unordered_map<unsigned int, Plane *> get_all_landmark_planes_unordered_map() const; // return unordered_map
                     void erase_landmark_plane(Plane *pl);                                                    // erase a plane from unordered_map
                     void erase_landmark_plane(unsigned id);                                                  // erase a plane from unordered_map, according to id
                     unsigned int get_num_landmark_planes() const;                                            // return the number of planes

                     //-----------------------------------------
                     // FW: some functionalities for line segments
                     bool _b_use_line_tracking = false;
                     void add_landmark_line(Line *line);
                     void erase_landmark_line(Line *line);
                     std::vector<Line *> get_all_landmarks_line() const;
                     unsigned int get_num_landmarks_line() const;
                     void set_local_landmarks_line(const std::vector<Line *> &local_lms_line);
                     std::vector<Line *> get_local_landmarks_line() const;

              private:
                     /**
                      * Decode JSON and register keyframe information to the map database
                      * (NOTE: objects which are not constructed yet will be set as nullptr)
                      * @param cam_db
                      * @param bow_vocab
                      * @param bow_db
                      * @param id
                      * @param json_keyfrm
                      */
                     void register_keyframe(camera_database *cam_db, bow_vocabulary *bow_vocab, bow_database *bow_db,
                                            const unsigned int id, const nlohmann::json &json_keyfrm);

                     /**
                      * Decode JSON and register landmark information to the map database
                      * (NOTE: objects which are not constructed yet will be set as nullptr)
                      * @param id
                      * @param json_landmark
                      */
                     void register_landmark(const unsigned int id, const nlohmann::json &json_landmark);

                     // FW:
                     void register_landmark_line(const unsigned int id, const nlohmann::json &json_landmark_line);

                     /**
                      * Decode JSON and register essential graph information
                      * (NOTE: keyframe database must be completely constructed before calling this function)
                      * @param id
                      * @param json_keyfrm
                      */
                     void register_graph(const unsigned int id, const nlohmann::json &json_keyfrm);

                     /**
                      * Decode JSON and register keyframe-landmark associations
                      * (NOTE: keyframe and landmark database must be completely constructed before calling this function)
                      * @param keyfrm_id
                      * @param json_keyfrm
                      */
                     void register_association(const unsigned int keyfrm_id, const nlohmann::json &json_keyfrm);

                     //! mutex for mutual exclusion controll between class methods
                     mutable std::mutex mtx_map_access_;

                     //-----------------------------------------
                     // keyframe and landmark database

                     //! IDs and keyframes
                     std::unordered_map<unsigned int, keyframe *> keyframes_;
                     //! IDs and landmarks
                     std::unordered_map<unsigned int, landmark *> landmarks_;

                     // FW:
                     std::unordered_map<unsigned int, Plane *> _landmarks_plane;

                     // FW:
                     std::unordered_map<unsigned int, Line *> _landmarks_line;

                     //! local landmarks
                     std::vector<landmark *> local_landmarks_;

                     // FW:
                     std::vector<Line *> _local_landmarks_line;

                     //! max keyframe ID
                     unsigned int max_keyfrm_id_ = 0;

                     //-----------------------------------------
                     // frame statistics for odometry evaluation

                     //! frame statistics
                     frame_statistics frm_stats_;
              };

       } // namespace data
} // namespace PLPSLAM

#endif // PLPSLAM_DATA_MAP_DATABASE_H
