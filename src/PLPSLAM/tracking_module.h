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

#ifndef PLPSLAM_TRACKING_MODULE_H
#define PLPSLAM_TRACKING_MODULE_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/data/frame.h"
#include "PLPSLAM/module/initializer.h"
#include "PLPSLAM/module/relocalizer.h"
#include "PLPSLAM/module/keyframe_inserter.h"
#include "PLPSLAM/module/frame_tracker.h"

#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace PLPSLAM
{

    class system;
    class mapping_module;
    class global_optimization_module;
    class Planar_Mapping_module; // FW: initialized in system.h and connected to tracker

    namespace data
    {
        class map_database;
        class bow_database;
    }

    namespace feature
    {
        class orb_extractor;
        class LineFeatureTracker; // FW: we add here a line tracker
    }

    // tracker state
    enum class tracker_state_t
    {
        NotInitialized,
        Initializing,
        Tracking,
        Lost
    };

    class tracking_module
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        //! Constructor
        tracking_module(const std::shared_ptr<config> &cfg, system *system, data::map_database *map_db,
                        data::bow_vocabulary *bow_vocab, data::bow_database *bow_db);

        //! Destructor
        ~tracking_module();

        //-----------------------------------------
        //! Set the mapping module
        void set_mapping_module(mapping_module *mapper);

        //! Set the global optimization module
        void set_global_optimization_module(global_optimization_module *global_optimizer);

        // FW: Set the planar mapping module
        void set_planar_mapping_module(Planar_Mapping_module *planar_mapper);

        //-----------------------------------------
        // interfaces

        //! Set mapping module status
        void set_mapping_module_status(const bool mapping_is_enabled);

        //! Get mapping module status
        bool get_mapping_module_status() const;

        //! Get the keypoints of the initial frame
        std::vector<cv::KeyPoint> get_initial_keypoints() const;

        //! Get the keypoint matches between the initial frame and the current frame
        std::vector<int> get_initial_matches() const;

        //-----------------------------------------
        //! Track a monocular frame
        //! (NOTE: distorted images are acceptable if calibrated)
        // FW: + line tracker if activated
        Mat44_t track_monocular_image(const cv::Mat &img, const double timestamp, const cv::Mat &mask = cv::Mat{});

        // FW: same but + planar segmentation mask
        // + line tracker if activated
        Mat44_t track_monocular_image(const cv::Mat &img, const cv::Mat &seg_mask_img, const double timestamp, const cv::Mat &mask = cv::Mat{});

        //-----------------------------------------
        //! Track a stereo frame
        //! (Note: Left and Right images must be stereo-rectified)
        // FW: + line tracker if activated
        Mat44_t track_stereo_image(const cv::Mat &left_img_rect, const cv::Mat &right_img_rect, const double timestamp, const cv::Mat &mask = cv::Mat{});

        // FW: same but +  planar segmentation mask
        // + line tracker if activated
        Mat44_t track_stereo_image(const cv::Mat &left_img_rect, const cv::Mat &right_img_rect, const cv::Mat &seg_mask_img, const double timestamp, const cv::Mat &mask = cv::Mat{});

        //-----------------------------------------
        //! Track an RGBD frame
        //! (Note: RGB and Depth images must be aligned)
        // FW: + line tracker if activated
        Mat44_t track_RGBD_image(const cv::Mat &img, const cv::Mat &depthmap, const double timestamp, const cv::Mat &mask = cv::Mat{});

        // FW: same but + planar segmentation mask
        // + line tracker if activated
        Mat44_t track_RGBD_image(const cv::Mat &img, const cv::Mat &depthmap, const cv::Mat &seg_mask_img, const double timestamp, const cv::Mat &mask = cv::Mat{});

        //-----------------------------------------
        // management for reset process

        //! Reset the databases
        void reset();

        //-----------------------------------------
        // management for pause process

        //! Request to pause the tracking module
        void request_pause();

        //! Check if the pause of the tracking module is requested or not
        bool pause_is_requested() const;

        //! Check if the tracking module is paused or not
        bool is_paused() const;

        //! Resume the tracking module
        void resume();

        //-----------------------------------------
        // variables

        //! config
        const std::shared_ptr<config> cfg_;

        //! camera model (equals to cfg_->camera_)
        camera::base *camera_;

        //! latest tracking state
        tracker_state_t tracking_state_ = tracker_state_t::NotInitialized;
        //! last tracking state
        tracker_state_t last_tracking_state_ = tracker_state_t::NotInitialized;

        //! current frame and its image
        data::frame curr_frm_;
        //! image of the current frame
        cv::Mat img_gray_;

        //! elapsed microseconds for each tracking
        double elapsed_ms_ = 0.0;

        //-----------------------------------------
        // FW: segmentation mask of current frame
        cv::Mat _seg_mask_img;
        unsigned int _num_planes = 0;

    protected:
        //-----------------------------------------
        // tracking processes

        //! Main stream of the tracking module
        void track();

        //! Try to initialize with the current frame
        bool initialize();

        //! Track the current frame
        bool track_current_frame();

        //! Update the motion model using the current and last frames
        void update_motion_model();

        //! Replace the landmarks if the `replaced` member has the valid pointer
        void apply_landmark_replace();

        // FW:
        void apply_landmark_replace_line();

        //! Update the camera pose of the last frame
        void update_last_frame();

        //! Optimize the camera pose of the current frame
        bool optimize_current_frame_with_local_map();

        //! Update the local map
        void update_local_map();

        //! Acquire more 2D-3D matches using initial camera pose estimation
        void search_local_landmarks();

        // FW:
        void search_local_landmarks_line();

        //! Check the new keyframe is needed or not
        bool new_keyframe_is_needed() const;

        //! Insert the new keyframe derived from the current frame
        void insert_new_keyframe();

        //! system
        system *system_ = nullptr;
        //! mapping module
        mapping_module *mapper_ = nullptr;
        //! global optimization module
        global_optimization_module *global_optimizer_ = nullptr;

        // FW: planar mapping module
        Planar_Mapping_module *_planar_mapper = nullptr;

        // ORB extractors
        //! ORB extractor for left/monocular image
        feature::orb_extractor *extractor_left_ = nullptr;
        //! ORB extractor for right image
        feature::orb_extractor *extractor_right_ = nullptr;
        //! ORB extractor only when used in initializing
        feature::orb_extractor *ini_extractor_left_ = nullptr;

        // FW: LSD/LBD extractors
        feature::LineFeatureTracker *_line_extractor = nullptr;
        feature::LineFeatureTracker *_line_extractor_right = nullptr;

        //! map_database
        data::map_database *map_db_ = nullptr;

        // Bag of Words
        //! BoW vocabulary
        data::bow_vocabulary *bow_vocab_ = nullptr;
        //! BoW database
        data::bow_database *bow_db_ = nullptr;

        //! initializer
        module::initializer initializer_;

        //! frame tracker for current frame
        module::frame_tracker frame_tracker_;

        //! relocalizer
        module::relocalizer relocalizer_;

        //! pose optimizer
        const optimize::pose_optimizer pose_optimizer_;

        // FW: pose optimizer with point and line reprojection error
        const optimize::pose_optimizer_extended_line _pose_optimizer_extended_line;

        //! keyframe inserter
        module::keyframe_inserter keyfrm_inserter_;

        //! reference keyframe
        data::keyframe *ref_keyfrm_ = nullptr;
        //! local keyframes
        std::vector<data::keyframe *> local_keyfrms_;
        //! local landmarks
        std::vector<data::landmark *> local_landmarks_;

        //! the number of tracked landmarks in the current frame
        unsigned int num_tracked_lms_ = 0;

        // FW: local landmarks for 3D line
        std::vector<data::Line *> _local_landmarks_line;

        // FW: the number of tracked 3D lines in the current frame
        unsigned int _num_tracked_lms_line = 0;

        //! last frame
        data::frame last_frm_;

        //! latest frame ID which succeeded in relocalization
        unsigned int last_reloc_frm_id_ = 0;

        //! motion model
        Mat44_t velocity_;
        //! motion model is valid or not
        bool velocity_is_valid_ = false;

        //! current camera pose from reference keyframe
        //! (to update last camera pose at the beginning of each tracking)
        Mat44_t last_cam_pose_from_ref_keyfrm_;

        //-----------------------------------------
        // mapping module status

        //! mutex for mapping module status
        mutable std::mutex mtx_mapping_;

        //! mapping module is enabled or not
        bool mapping_is_enabled_ = true;

        //-----------------------------------------
        // management for pause process

        //! mutex for pause process
        mutable std::mutex mtx_pause_;

        //! Check the request frame and pause the tracking module
        bool check_and_execute_pause();

        //! the tracking module is paused or not
        bool is_paused_ = false;

        //! Pause of the tracking module is requested or not
        bool pause_is_requested_ = false;

        // FW: print Debug info in the terminal (Planar Mapping Module)
        const bool _setVerbose = false;
    };

} // namespace PLPSLAM

#endif // PLPSLAM_TRACKING_MODULE_H
