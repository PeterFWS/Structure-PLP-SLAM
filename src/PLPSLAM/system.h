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

#ifndef PLPSLAM_SYSTEM_H
#define PLPSLAM_SYSTEM_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/data/bow_vocabulary.h"

#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <opencv2/core/core.hpp>

#include <yaml-cpp/yaml.h>

namespace PLPSLAM
{

    class config;
    class tracking_module;
    class mapping_module;
    class global_optimization_module;
    class Planar_Mapping_module; // FW: we add a new module called planar_mapping module

    namespace camera
    {
        class base;
    }

    namespace data
    {
        class camera_database;
        class map_database;
        class bow_database;
        class Plane; // FW:
    }

    namespace publish
    {
        class map_publisher;
        class frame_publisher;
    }

    class system
    {
    public:
        //! Constructor
        system(const std::shared_ptr<config> &cfg, const std::string &vocab_file_path,
               const bool b_seg_or_not = false,
               const bool b_use_line_tracking = false);

        //! Destructor
        ~system();

        //-----------------------------------------
        // system startup and shutdown

        //! Startup the SLAM system
        void startup(const bool need_initialize = true);

        //! Shutdown the SLAM system
        void shutdown();

        //-----------------------------------------
        // data I/O

        //! Save the frame trajectory in the specified format
        void save_frame_trajectory(const std::string &path, const std::string &format) const;

        //! Save the keyframe trajectory in the specified format
        void save_keyframe_trajectory(const std::string &path, const std::string &format) const;

        //! Load the map database from the MessagePack file
        void load_map_database(const std::string &path) const;

        //! Save the map database to the MessagePack file
        void save_map_database(const std::string &path) const;

        //! Get the map publisher
        const std::shared_ptr<publish::map_publisher> get_map_publisher() const;

        //! Get the frame publisher
        const std::shared_ptr<publish::frame_publisher> get_frame_publisher() const;

        //-----------------------------------------
        // module management

        //! Enable the mapping module
        void enable_mapping_module();

        //! Disable the mapping module
        void disable_mapping_module();

        //! The mapping module is enabled or not
        bool mapping_module_is_enabled() const;

        //! Enable the loop detector
        void enable_loop_detector();

        //! Disable the loop detector
        void disable_loop_detector();

        //! The loop detector is enabled or not
        bool loop_detector_is_enabled() const;

        //! Loop BA is running or not
        bool loop_BA_is_running() const;

        //! Abort the loop BA externally
        void abort_loop_BA();

        //-----------------------------------------
        // data feeding methods (Monocular)

        //! Feed a monocular frame to SLAM system
        //! (NOTE: distorted images are acceptable if calibrated)
        Mat44_t feed_monocular_frame(const cv::Mat &img, const double timestamp, const cv::Mat &mask = cv::Mat{});

        // FW: same monocular function but with segmentation mask as auxiliary input
        Mat44_t feed_monocular_frame(const cv::Mat &img, const cv::Mat &seg_mask_img, const double timestamp, const cv::Mat &mask = cv::Mat{});

        //-----------------------------------------
        // data feeding methods (Stereo)

        //! Feed a stereo frame to SLAM system
        //! (Note: Left and Right images must be stereo-rectified)
        Mat44_t feed_stereo_frame(const cv::Mat &left_img, const cv::Mat &right_img, const double timestamp, const cv::Mat &mask = cv::Mat{});

        Mat44_t feed_stereo_frame(const cv::Mat &left_img, const cv::Mat &right_img, const cv::Mat &seg_mask_img, const double timestamp, const cv::Mat &mask = cv::Mat{});

        //-----------------------------------------
        // data feeding methods (RGB-D)

        //! Feed an RGBD frame to SLAM system
        //! (Note: RGB and Depth images must be aligned)
        Mat44_t feed_RGBD_frame(const cv::Mat &rgb_img, const cv::Mat &depthmap, const double timestamp, const cv::Mat &mask = cv::Mat{});

        // FW: same rgbd function but with segmentation mask as auxiliary input
        Mat44_t feed_RGBD_frame(const cv::Mat &rgb_img, const cv::Mat &depthmap, const cv::Mat &seg_mask_img, const double timestamp, const cv::Mat &mask = cv::Mat{});

        //-----------------------------------------
        // management for pause

        //! Pause the tracking module
        void pause_tracker();

        //! The tracking module is paused or not
        bool tracker_is_paused() const;

        //! Resume the tracking module
        void resume_tracker();

        //-----------------------------------------
        // management for reset

        //! Request to reset the system
        void request_reset();

        //! Reset of the system is requested or not
        bool reset_is_requested() const;

        //-----------------------------------------
        // management for terminate

        //! Request to terminate the system
        void request_terminate();

        //!! Termination of the system is requested or not
        bool terminate_is_requested() const;

        //-----------------------------------------
        // FW: (Planar Mapping Module)
        // some parameters need to be loaded from yaml
        const std::string _cfg_path = "./src/PLPSLAM/planar_mapping_parameters.yaml";
        void load_configuration(const std::string path);

        void final_refinement_plane(); // this function does not bring too much improvement on ATE

    private:
        //! Check reset request of the system
        void check_reset_request();

        //! Pause the mapping module and the global optimization module
        void pause_other_threads() const;

        //! Resume the mapping module and the global optimization module
        void resume_other_threads() const;

        //! config
        const std::shared_ptr<config> cfg_;
        //! camera model
        camera::base *camera_ = nullptr;

        //! camera database
        data::camera_database *cam_db_ = nullptr;

        //! map database
        data::map_database *map_db_ = nullptr;

        //! BoW vocabulary
        data::bow_vocabulary *bow_vocab_ = nullptr;

        //! BoW database
        data::bow_database *bow_db_ = nullptr;

        //! tracker
        tracking_module *tracker_ = nullptr;

        //! mapping module
        mapping_module *mapper_ = nullptr;
        //! mapping thread
        std::unique_ptr<std::thread> mapping_thread_ = nullptr;

        //! global optimization module
        global_optimization_module *global_optimizer_ = nullptr;
        //! global optimization thread
        std::unique_ptr<std::thread> global_optimization_thread_ = nullptr;

        // FW: planar mapping module
        Planar_Mapping_module *_planar_mapper = nullptr;

        //! frame publisher
        std::shared_ptr<publish::frame_publisher> frame_publisher_ = nullptr;
        //! map publisher
        std::shared_ptr<publish::map_publisher> map_publisher_ = nullptr;

        //! system running status flag
        std::atomic<bool> system_is_running_{false};

        //! mutex for reset flag
        mutable std::mutex mtx_reset_;
        //! reset flag
        bool reset_is_requested_ = false;

        //! mutex for terminate flag
        mutable std::mutex mtx_terminate_;
        //! terminate flag
        bool terminate_is_requested_ = false;

        //! mutex for flags of enable/disable mapping module
        mutable std::mutex mtx_mapping_;

        //! mutex for flags of enable/disable loop detector
        mutable std::mutex mtx_loop_detector_;

        //-----------------------------------------
        // FW: a boolean variable indicates the system has segmentation mask as input, with corresponding RGB image
        bool _b_seg_or_not = false;

        // FW: a boolean variable indicates the system will use LSD/LBD
        bool _b_use_line_tracking = false;
    };

} // namespace PLPSLAM

#endif // PLPSLAM_SYSTEM_H
