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

#include "PLPSLAM/system.h"
#include "PLPSLAM/config.h"
#include "PLPSLAM/tracking_module.h"
#include "PLPSLAM/mapping_module.h"
#include "PLPSLAM/global_optimization_module.h"
#include "PLPSLAM/camera/base.h"
#include "PLPSLAM/data/camera_database.h"
#include "PLPSLAM/data/map_database.h"
#include "PLPSLAM/data/bow_database.h"
#include "PLPSLAM/io/trajectory_io.h"
#include "PLPSLAM/io/map_database_io.h"
#include "PLPSLAM/publish/map_publisher.h"
#include "PLPSLAM/publish/frame_publisher.h"
#include "PLPSLAM/optimize/global_bundle_adjuster.h"

#include <thread>

#include <spdlog/spdlog.h>

// FW:
#include "PLPSLAM/data/landmark_plane.h"
#include "PLPSLAM/planar_mapping_module.h"

namespace PLPSLAM
{

    system::system(const std::shared_ptr<config> &cfg,
                   const std::string &vocab_file_path,
                   const bool b_seg_or_not,
                   const bool b_use_line_tracking)
        : cfg_(cfg),
          camera_(cfg->camera_),
          _b_seg_or_not(b_seg_or_not),
          _b_use_line_tracking(b_use_line_tracking)
    {
        spdlog::debug("CONSTRUCT: system");

        std::cout << std::endl;
        std::cout << "----------------------------------------------------------------" << std::endl;
        std::cout << "Structure PLP-SLAM:" << std::endl;
        std::cout << "Copyright (C) 2022, Department Augmented Vision, DFKI, Germany." << std::endl;
        std::cout << "All rights reserved." << std::endl;
        std::cout << std::endl;
        std::cout << "This is free software," << std::endl;
        std::cout << "and you are welcome to redistribute it under certain conditions." << std::endl;
        std::cout << "See the LICENSE file." << std::endl;
        std::cout << "----------------------------------------------------------------" << std::endl;
        std::cout << std::endl;

        // show configuration
        std::cout << *cfg_ << std::endl;

        // load ORB vocabulary
        spdlog::info("loading ORB vocabulary: {}", vocab_file_path);
#ifdef USE_DBOW2
        bow_vocab_ = new data::bow_vocabulary();
        try
        {
            bow_vocab_->loadFromBinaryFile(vocab_file_path);
        }
        catch (const std::exception &e)
        {
            spdlog::critical("wrong path to vocabulary");
            delete bow_vocab_;
            bow_vocab_ = nullptr;
            exit(EXIT_FAILURE);
        }
#else
        bow_vocab_ = new fbow::Vocabulary();
        bow_vocab_->readFromFile(vocab_file_path);
        if (!bow_vocab_->isValid())
        {
            spdlog::critical("wrong path to vocabulary");
            delete bow_vocab_;
            bow_vocab_ = nullptr;
            exit(EXIT_FAILURE);
        }
#endif

        // database
        cam_db_ = new data::camera_database(camera_);
        map_db_ = new data::map_database();
        bow_db_ = new data::bow_database(bow_vocab_);

        // FW: activate line tracker after map database is initialized
        if (_b_use_line_tracking)
        {
            map_db_->_b_use_line_tracking = true;
        }

        // FW: load some configurations for planar mapping module
        if (_b_seg_or_not)
        {
            spdlog::info(">system< connect other module with PlanarMapping module");

            // load planar mapping parameters
            load_configuration(_cfg_path);

            // in case of planar SLAM, we load this boolean variable from the config file
            if (_b_use_line_tracking)
            {
                map_db_->_b_use_line_tracking = true;
            }
        }

        // frame and map publisher
        frame_publisher_ = std::shared_ptr<publish::frame_publisher>(new publish::frame_publisher(cfg_, map_db_));
        map_publisher_ = std::shared_ptr<publish::map_publisher>(new publish::map_publisher(cfg_, map_db_));

        // tracking module
        tracker_ = new tracking_module(cfg_, this, map_db_, bow_vocab_, bow_db_);

        // mapping module
        mapper_ = new mapping_module(map_db_, camera_->setup_type_ == camera::setup_type_t::Monocular);

        // global optimization module
        global_optimizer_ = new global_optimization_module(map_db_, bow_db_, bow_vocab_, camera_->setup_type_ != camera::setup_type_t::Monocular);

        // connect modules each other
        tracker_->set_mapping_module(mapper_);
        tracker_->set_global_optimization_module(global_optimizer_);

        mapper_->set_tracking_module(tracker_);
        mapper_->set_global_optimization_module(global_optimizer_);

        global_optimizer_->set_tracking_module(tracker_);
        global_optimizer_->set_mapping_module(mapper_);

        // FW: activate and connect Planar Mapping Module
        if (_b_seg_or_not)
        {
            // initialize the planar_mapping module and link the map database, set camera configuration
            _planar_mapper = new Planar_Mapping_module(map_db_, camera_->setup_type_ == camera::setup_type_t::Monocular);

            // connect modules
            tracker_->set_planar_mapping_module(_planar_mapper);
            mapper_->set_planar_mapping_module(_planar_mapper);
        }
    }

    system::~system()
    {
        global_optimization_thread_.reset(nullptr);
        delete global_optimizer_;
        global_optimizer_ = nullptr;

        mapping_thread_.reset(nullptr);
        delete mapper_;
        mapper_ = nullptr;

        delete tracker_;
        tracker_ = nullptr;

        // FW: also delete planar mapper
        if (_b_seg_or_not)
        {
            delete _planar_mapper;
            _planar_mapper = nullptr;
        }

        delete bow_db_;
        bow_db_ = nullptr;
        delete map_db_;
        map_db_ = nullptr;
        delete cam_db_;
        cam_db_ = nullptr;
        delete bow_vocab_;
        bow_vocab_ = nullptr;

        spdlog::debug("DESTRUCT: system");
    }

    void system::startup(const bool need_initialize)
    {
        spdlog::info("startup SLAM system");
        system_is_running_ = true;

        if (!need_initialize)
        {
            tracker_->tracking_state_ = tracker_state_t::Lost;
        }

        mapping_thread_ = std::unique_ptr<std::thread>(new std::thread(&PLPSLAM::mapping_module::run, mapper_));
        global_optimization_thread_ = std::unique_ptr<std::thread>(new std::thread(&PLPSLAM::global_optimization_module::run, global_optimizer_));
    }

    void system::shutdown()
    {
        // terminate the other threads
        mapper_->request_terminate();
        global_optimizer_->request_terminate();

        // wait until they stop
        while (!mapper_->is_terminated() || !global_optimizer_->is_terminated() || global_optimizer_->loop_BA_is_running())
        {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }

        // wait until the threads stop
        mapping_thread_->join();
        global_optimization_thread_->join();

        spdlog::info("shutdown SLAM system");
        system_is_running_ = false;
    }

    void system::save_frame_trajectory(const std::string &path, const std::string &format) const
    {
        pause_other_threads();
        io::trajectory_io trajectory_io(map_db_);
        trajectory_io.save_frame_trajectory(path, format);
        resume_other_threads();
    }

    void system::save_keyframe_trajectory(const std::string &path, const std::string &format) const
    {
        pause_other_threads();
        io::trajectory_io trajectory_io(map_db_);
        trajectory_io.save_keyframe_trajectory(path, format);
        resume_other_threads();
    }

    void system::load_map_database(const std::string &path) const
    {
        pause_other_threads();
        io::map_database_io map_db_io(cam_db_, map_db_, bow_db_, bow_vocab_);
        map_db_io.load_message_pack(path);
        resume_other_threads();
    }

    void system::save_map_database(const std::string &path) const
    {
        pause_other_threads();
        io::map_database_io map_db_io(cam_db_, map_db_, bow_db_, bow_vocab_);
        map_db_io.save_message_pack(path);
        resume_other_threads();
    }

    const std::shared_ptr<publish::map_publisher> system::get_map_publisher() const
    {
        return map_publisher_;
    }

    const std::shared_ptr<publish::frame_publisher> system::get_frame_publisher() const
    {
        return frame_publisher_;
    }

    void system::enable_mapping_module()
    {
        std::lock_guard<std::mutex> lock(mtx_mapping_);
        if (!system_is_running_)
        {
            spdlog::critical("please call system::enable_mapping_module() after system::startup()");
        }
        // resume the mapping module
        mapper_->resume();
        // inform to the tracking module
        tracker_->set_mapping_module_status(true);
    }

    void system::disable_mapping_module()
    {
        std::lock_guard<std::mutex> lock(mtx_mapping_);
        if (!system_is_running_)
        {
            spdlog::critical("please call system::disable_mapping_module() after system::startup()");
        }
        // pause the mapping module
        mapper_->request_pause();
        // wait until it stops
        while (!mapper_->is_paused())
        {
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
        // inform to the tracking module
        tracker_->set_mapping_module_status(false);
    }

    bool system::mapping_module_is_enabled() const
    {
        return !mapper_->is_paused();
    }

    void system::enable_loop_detector()
    {
        std::lock_guard<std::mutex> lock(mtx_loop_detector_);
        global_optimizer_->enable_loop_detector();
    }

    void system::disable_loop_detector()
    {
        std::lock_guard<std::mutex> lock(mtx_loop_detector_);
        global_optimizer_->disable_loop_detector();
    }

    bool system::loop_detector_is_enabled() const
    {
        return global_optimizer_->loop_detector_is_enabled();
    }

    bool system::loop_BA_is_running() const
    {
        return global_optimizer_->loop_BA_is_running();
    }

    void system::abort_loop_BA()
    {
        global_optimizer_->abort_loop_BA();
    }

    // (default) monocular
    Mat44_t system::feed_monocular_frame(const cv::Mat &img, const double timestamp, const cv::Mat &mask)
    {
        assert(camera_->setup_type_ == camera::setup_type_t::Monocular);

        check_reset_request();

        const Mat44_t cam_pose_cw = tracker_->track_monocular_image(img, timestamp, mask);

        frame_publisher_->update(tracker_);
        if (tracker_->tracking_state_ == tracker_state_t::Tracking)
        {
            map_publisher_->set_current_cam_pose(cam_pose_cw);
        }

        return cam_pose_cw;
    }

    // FW: monocular + planar segmentation
    Mat44_t system::feed_monocular_frame(const cv::Mat &img, const cv::Mat &seg_mask_img, const double timestamp, const cv::Mat &mask)
    {
        assert(camera_->setup_type_ == camera::setup_type_t::Monocular);

        check_reset_request();

        const Mat44_t cam_pose_cw = tracker_->track_monocular_image(img, seg_mask_img, timestamp, mask);

        frame_publisher_->update(tracker_);
        if (tracker_->tracking_state_ == tracker_state_t::Tracking)
        {
            map_publisher_->set_current_cam_pose(cam_pose_cw);
        }

        return cam_pose_cw;
    }

    // (default) stereo
    Mat44_t system::feed_stereo_frame(const cv::Mat &left_img, const cv::Mat &right_img, const double timestamp, const cv::Mat &mask)
    {
        assert(camera_->setup_type_ == camera::setup_type_t::Stereo);

        check_reset_request();

        const Mat44_t cam_pose_cw = tracker_->track_stereo_image(left_img, right_img, timestamp, mask);

        frame_publisher_->update(tracker_);
        if (tracker_->tracking_state_ == tracker_state_t::Tracking)
        {
            map_publisher_->set_current_cam_pose(cam_pose_cw);
        }

        return cam_pose_cw;
    }

    // FW: stereo + planar segmentation
    Mat44_t system::feed_stereo_frame(const cv::Mat &left_img, const cv::Mat &right_img, const cv::Mat &seg_mask_img, const double timestamp, const cv::Mat &mask)
    {
        assert(camera_->setup_type_ == camera::setup_type_t::Stereo);

        check_reset_request();

        const Mat44_t cam_pose_cw = tracker_->track_stereo_image(left_img, right_img, seg_mask_img, timestamp, mask);

        frame_publisher_->update(tracker_);
        if (tracker_->tracking_state_ == tracker_state_t::Tracking)
        {
            map_publisher_->set_current_cam_pose(cam_pose_cw);
        }

        return cam_pose_cw;
    }

    // (default) RGBD
    Mat44_t system::feed_RGBD_frame(const cv::Mat &rgb_img, const cv::Mat &depthmap, const double timestamp, const cv::Mat &mask)
    {
        assert(camera_->setup_type_ == camera::setup_type_t::RGBD);

        check_reset_request();

        const Mat44_t cam_pose_cw = tracker_->track_RGBD_image(rgb_img, depthmap, timestamp, mask);

        frame_publisher_->update(tracker_);
        if (tracker_->tracking_state_ == tracker_state_t::Tracking)
        {
            map_publisher_->set_current_cam_pose(cam_pose_cw);
        }

        return cam_pose_cw;
    }

    // FW: RGB-D + planar segmentation
    Mat44_t system::feed_RGBD_frame(const cv::Mat &rgb_img, const cv::Mat &depthmap, const cv::Mat &seg_mask_img, const double timestamp, const cv::Mat &mask)
    {
        assert(camera_->setup_type_ == camera::setup_type_t::RGBD);

        check_reset_request();

        const Mat44_t cam_pose_cw = tracker_->track_RGBD_image(rgb_img, depthmap, seg_mask_img, timestamp, mask);

        frame_publisher_->update(tracker_);
        if (tracker_->tracking_state_ == tracker_state_t::Tracking)
        {
            map_publisher_->set_current_cam_pose(cam_pose_cw);
        }

        return cam_pose_cw;
    }

    void system::pause_tracker()
    {
        tracker_->request_pause();
    }

    bool system::tracker_is_paused() const
    {
        return tracker_->is_paused();
    }

    void system::resume_tracker()
    {
        tracker_->resume();
    }

    void system::request_reset()
    {
        std::lock_guard<std::mutex> lock(mtx_reset_);
        reset_is_requested_ = true;
    }

    bool system::reset_is_requested() const
    {
        std::lock_guard<std::mutex> lock(mtx_reset_);
        return reset_is_requested_;
    }

    void system::request_terminate()
    {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        terminate_is_requested_ = true;
    }

    bool system::terminate_is_requested() const
    {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        return terminate_is_requested_;
    }

    void system::check_reset_request()
    {
        std::lock_guard<std::mutex> lock(mtx_reset_);
        if (reset_is_requested_)
        {
            tracker_->reset();
            reset_is_requested_ = false;
        }
    }

    void system::pause_other_threads() const
    {
        // pause the mapping module
        if (mapper_ && !mapper_->is_terminated())
        {
            mapper_->request_pause();
            while (!mapper_->is_paused() && !mapper_->is_terminated())
            {
                std::this_thread::sleep_for(std::chrono::microseconds(5000));
            }
        }

        // pause the global optimization module
        if (global_optimizer_ && !global_optimizer_->is_terminated())
        {
            global_optimizer_->request_pause();
            while (!global_optimizer_->is_paused() && !global_optimizer_->is_terminated())
            {
                std::this_thread::sleep_for(std::chrono::microseconds(5000));
            }
        }
    }

    void system::resume_other_threads() const
    {
        // resume the global optimization module
        if (global_optimizer_)
        {
            global_optimizer_->resume();
        }

        // resume the mapping module
        if (mapper_)
        {
            mapper_->resume();
        }
    }

    // FW: this function does not bring too much improvement on ATE
    void system::final_refinement_plane()
    {
        std::cout << std::endl;
        spdlog::info("-- Final merge and refinement of planes");

        // try to merge/refine whatever planes existing in the map
        // this should be helpful for the visualization
        _planar_mapper->refinement();

        auto planes = map_db_->get_all_landmark_planes();
        spdlog::info("\t | Total number of planes: {}", planes.size());
        for (auto pl : planes)
        {
            if (pl->is_valid())
                spdlog::info("\t \t | plane id: {}, size: {}", pl->_id, pl->get_num_landmarks());
        }
    }

    // FW:
    void system::load_configuration(const std::string path)
    {
        YAML::Node yaml_node = YAML::LoadFile(path);

        // in case using line tracking
        _b_use_line_tracking = yaml_node["Threshold.use_line_tracking"].as<bool>();
    }

} // namespace PLPSLAM
