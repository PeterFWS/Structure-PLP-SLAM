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

#ifndef PLPSLAM_MODULE_LOCAL_MAP_CLEANER_H
#define PLPSLAM_MODULE_LOCAL_MAP_CLEANER_H

#include <list>

namespace PLPSLAM
{

       namespace data
       {
              class keyframe;
              class landmark;

              // FW:
              class Line;
       } // namespace data

       namespace module
       {
              class local_map_cleaner
              {
              public:
                     /**
                      * Constructor
                      */
                     explicit local_map_cleaner(const bool is_monocular);

                     /**
                      * Destructor
                      */
                     ~local_map_cleaner() = default;

                     /**
                      * Set the origin keyframe ID
                      */
                     void set_origin_keyframe_id(const unsigned int id)
                     {
                            origin_keyfrm_id_ = id;
                     }

                     /**
                      * Add fresh landmark to check their redundancy
                      */
                     void add_fresh_landmark(data::landmark *lm)
                     {
                            fresh_landmarks_.push_back(lm);
                     }

                     // FW:
                     void add_fresh_landmark_line(data::Line *lm_line)
                     {
                            _fresh_landmarks_line.push_back(lm_line);
                     }

                     /**
                      * Reset the buffer
                      */
                     void reset();

                     /**
                      * Remove redundant landmarks
                      */
                     unsigned int remove_redundant_landmarks(const unsigned int cur_keyfrm_id);

                     // FW: same strategy for removing 3D line
                     unsigned int remove_redundant_landmarks_line(const unsigned int cur_keyfrm_id);

                     /**
                      * Remove redundant keyframes
                      */
                     unsigned int remove_redundant_keyframes(data::keyframe *cur_keyfrm) const;

                     /**
                      * Count the valid and the redundant observations in the specified keyframe
                      */
                     void count_redundant_observations(data::keyframe *keyfrm, unsigned int &num_valid_obs, unsigned int &num_redundant_obs) const;

                     // FW:
                     bool _b_use_line_tracking = false;

              private:
                     //! origin keyframe ID
                     unsigned int origin_keyfrm_id_ = 0;

                     //! flag which indicates the tracking camera is monocular or not
                     const bool is_monocular_;

                     //! fresh landmarks to check their redundancy
                     std::list<data::landmark *> fresh_landmarks_;

                     // FW:
                     std::list<data::Line *> _fresh_landmarks_line;
              };

       } // namespace module
} // namespace PLPSLAM

#endif // PLPSLAM_MODULE_LOCAL_MAP_CLEANER_H
