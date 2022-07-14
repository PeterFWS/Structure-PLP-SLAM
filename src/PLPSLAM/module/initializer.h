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

#ifndef PLPSLAM_MODULE_INITIALIZER_H
#define PLPSLAM_MODULE_INITIALIZER_H

#include "PLPSLAM/data/frame.h"
#include "PLPSLAM/initialize/base.h"
#include <memory>

// FW:
#include "PLPSLAM/planar_mapping_module.h"
#include "PLPSLAM/data/landmark_line.h"

namespace PLPSLAM
{

    class config;
    class Planar_Mapping_module; // FW:

    namespace data
    {
        class frame;
        class map_database;
        class bow_database;
        class Line; // FW:
    }

    namespace module
    {

        // initializer state
        enum class initializer_state_t
        {
            NotReady,
            Initializing,
            Wrong,
            Succeeded
        };

        class initializer
        {
        public:
            initializer() = delete;

            //! Constructor
            initializer(const camera::setup_type_t setup_type,
                        data::map_database *map_db, data::bow_database *bow_db,
                        const YAML::Node &yaml_node);

            //! Destructor
            ~initializer();

            //! Reset initializer
            void reset();

            //! Get initialization state
            initializer_state_t get_state() const;

            //! Get keypoints of the initial frame
            std::vector<cv::KeyPoint> get_initial_keypoints() const;

            //! Get initial matches between the initial and current frames
            std::vector<int> get_initial_matches() const;

            //! Get the initial frame ID which succeeded in initialization
            unsigned int get_initial_frame_id() const;

            //! Initialize with the current frame
            bool initialize(data::frame &curr_frm);

        private:
            //! camera setup type
            const camera::setup_type_t setup_type_;
            //! map database
            data::map_database *map_db_ = nullptr;
            //! BoW database
            data::bow_database *bow_db_ = nullptr;
            //! initializer status
            initializer_state_t state_ = initializer_state_t::NotReady;

            //! frame ID used for initialization (will be set after succeeded)
            unsigned int init_frm_id_ = 0;

            //-----------------------------------------
            // parameters

            //! max number of iterations of RANSAC (only for monocular initializer)
            const unsigned int num_ransac_iters_;
            //! min number of triangulated pts
            const unsigned int min_num_triangulated_;
            //! min parallax (only for monocular initializer)
            const float parallax_deg_thr_;
            //! reprojection error threshold (only for monocular initializer)
            const float reproj_err_thr_;
            //! max number of iterations of BA (only for monocular initializer)
            const unsigned int num_ba_iters_;
            //! initial scaling factor (only for monocular initializer)
            const float scaling_factor_;

            //-----------------------------------------
            // for monocular camera model

            //! Create initializer for monocular
            void create_initializer(data::frame &curr_frm);

            //! Try to initialize a map with monocular camera setup
            bool try_initialize_for_monocular(data::frame &curr_frm);

            //! Create an initial map with monocular camera setup
            bool create_map_for_monocular(data::frame &curr_frm);

            //! Scaling up or down a initial map
            void scale_map(data::keyframe *init_keyfrm, data::keyframe *curr_keyfrm, const double scale);

            //! initializer for monocular
            std::unique_ptr<initialize::base> initializer_base_ = nullptr;
            //! initial frame
            data::frame init_frm_;
            //! coordinates of previously matched points to perform area-based matching
            std::vector<cv::Point2f> prev_matched_coords_;
            //! initial matching indices (index: idx of initial frame, value: idx of current frame)
            std::vector<int> init_matches_;

            // FW:
            void triangulate_line_with_two_keyframes(data::keyframe *cur_keyfrm, data::keyframe *ngh_keyfrm, data::frame &curr_frm);

            //-----------------------------------------
            // for stereo or RGBD camera model

            //! Try to initialize a map with stereo or RGBD camera setup
            bool try_initialize_for_stereo(data::frame &curr_frm);

            //! Create an initial map with stereo or RGBD camera setup
            bool create_map_for_stereo(data::frame &curr_frm);
        };

    } // namespace module
} // namespace PLPSLAM

#endif // PLPSLAM_MODULE_INITIALIZER_H
