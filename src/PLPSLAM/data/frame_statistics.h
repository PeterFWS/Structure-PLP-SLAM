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

#ifndef PLPSLAM_DATA_FRAME_STATISTICS_H
#define PLPSLAM_DATA_FRAME_STATISTICS_H

#include "PLPSLAM/type.h"

#include <vector>
#include <unordered_map>

namespace PLPSLAM
{
       namespace data
       {

              class frame;
              class keyframe;

              class frame_statistics
              {
              public:
                     /**
                      * Constructor
                      */
                     frame_statistics() = default;

                     /**
                      * Destructor
                      */
                     virtual ~frame_statistics() = default;

                     /**
                      * Update frame statistics
                      * @param frm
                      * @param is_lost
                      */
                     void update_frame_statistics(const data::frame &frm, const bool is_lost);

                     /**
                      * Replace a keyframe which will be erased in frame statistics
                      * @param old_keyfrm
                      * @param new_keyfrm
                      */
                     void replace_reference_keyframe(data::keyframe *old_keyfrm, data::keyframe *new_keyfrm);

                     /**
                      * Get frame IDs of each of the reference keyframes
                      * @return
                      */
                     std::unordered_map<data::keyframe *, std::vector<unsigned int>> get_frame_id_of_reference_keyframes() const;

                     /**
                      * Get the number of the contained valid frames
                      * @return
                      */
                     unsigned int get_num_valid_frames() const;

                     /**
                      * Get reference keyframes of each of the frames
                      * @return
                      */
                     std::map<unsigned int, data::keyframe *> get_reference_keyframes() const;

                     /**
                      * Get relative camera poses from the corresponding reference keyframes
                      * @return
                      */
                     eigen_alloc_map<unsigned int, Mat44_t> get_relative_cam_poses() const;

                     /**
                      * Get timestamps
                      * @return
                      */
                     std::map<unsigned int, double> get_timestamps() const;

                     /**
                      * Get lost frame flags
                      * @return
                      */
                     std::map<unsigned int, bool> get_lost_frames() const;

                     /**
                      * Clear frame statistics
                      */
                     void clear();

              private:
                     //! Reference keyframe, frame ID associated with the keyframe
                     std::unordered_map<data::keyframe *, std::vector<unsigned int>> frm_ids_of_ref_keyfrms_;

                     //! Number of valid frames
                     unsigned int num_valid_frms_ = 0;

                     // Size of all the following variables is the number of frames
                     //! Reference keyframes for each frame
                     std::unordered_map<unsigned int, data::keyframe *> ref_keyfrms_;
                     //! Relative pose against reference keyframe for each frame
                     eigen_alloc_unord_map<unsigned int, Mat44_t> rel_cam_poses_from_ref_keyfrms_;
                     //! Timestamp for each frame
                     std::unordered_map<unsigned int, double> timestamps_;
                     //! Flag whether each frame is lost or not
                     std::unordered_map<unsigned int, bool> is_lost_frms_;
              };

       } // namespace data
} // namespace PLPSLAM

#endif // PLPSLAM_DATA_FRAME_STATISTICS_H
