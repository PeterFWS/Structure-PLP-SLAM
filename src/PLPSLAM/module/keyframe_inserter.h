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

#ifndef PLPSLAM_MODULE_KEYFRAME_INSERTER_H
#define PLPSLAM_MODULE_KEYFRAME_INSERTER_H

#include "PLPSLAM/camera/base.h"
#include "PLPSLAM/data/frame.h"
#include "PLPSLAM/data/keyframe.h"

#include <memory>

namespace PLPSLAM
{

    class mapping_module;

    namespace data
    {
        class map_database;
    } // namespace data

    namespace module
    {

        class keyframe_inserter
        {
        public:
            keyframe_inserter(const camera::setup_type_t setup_type, const float true_depth_thr,
                              data::map_database *map_db, data::bow_database *bow_db,
                              const unsigned int min_num_frms, const unsigned int max_num_frms);

            virtual ~keyframe_inserter() = default;

            void set_mapping_module(mapping_module *mapper);

            void reset();

            /**
             * Check the new keyframe is needed or not
             */
            bool new_keyframe_is_needed(const data::frame &curr_frm, const unsigned int num_tracked_lms,
                                        const data::keyframe &ref_keyfrm) const;

            /**
             * Insert the new keyframe derived from the current frame
             */
            data::keyframe *insert_new_keyframe(data::frame &curr_frm);

        private:
            /**
             * Queue the new keyframe to the mapping module
             */
            void queue_keyframe(data::keyframe *keyfrm);

            //! setup type of the tracking camera
            const camera::setup_type_t setup_type_;
            //! depth threshold in metric scale
            const float true_depth_thr_;

            //! map database
            data::map_database *map_db_ = nullptr;
            //! BoW database
            data::bow_database *bow_db_ = nullptr;

            //! mapping module
            mapping_module *mapper_ = nullptr;

            //! min number of frames to insert keyframe
            const unsigned int min_num_frms_;
            //! max number of frames to insert keyframe
            const unsigned int max_num_frms_;

            //! frame ID of the last keyframe
            unsigned int frm_id_of_last_keyfrm_ = 0;
        };

    } // namespace module
} // namespace PLPSLAM

#endif // PLPSLAM_MODULE_KEYFRAME_INSERTER_H
