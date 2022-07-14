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

#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/data/map_database.h"
#include "PLPSLAM/publish/frame_publisher.h"

#include <spdlog/spdlog.h>
#include <opencv2/imgproc.hpp>

namespace PLPSLAM
{
    namespace publish
    {

        frame_publisher::frame_publisher(const std::shared_ptr<config> &cfg,
                                         data::map_database *map_db, const unsigned int img_width)
            : cfg_(cfg), map_db_(map_db), img_width_(img_width),
              img_(cv::Mat(480, img_width_, CV_8UC3, cv::Scalar(0, 0, 0))),
              // FW: add segmentation img for visualization
              _seg_mask_img(cv::Mat(480, img_width_, CV_8UC3, cv::Scalar(0, 0, 0)))
        {
            spdlog::debug("CONSTRUCT: publish::frame_publisher");
        }

        frame_publisher::~frame_publisher()
        {
            spdlog::debug("DESTRUCT: publish::frame_publisher");
        }

        cv::Mat frame_publisher::draw_frame(const bool draw_text)
        {
            cv::Mat img;
            tracker_state_t tracking_state;
            std::vector<cv::KeyPoint> init_keypts;
            std::vector<int> init_matches;
            std::vector<cv::KeyPoint> curr_keypts;
            double elapsed_ms;
            bool mapping_is_enabled;
            std::vector<bool> is_tracked;

            // FW:
            std::vector<cv::line_descriptor::KeyLine> curr_keylines;
            std::vector<bool> is_tracked_line;

            // copy to avoid memory access conflict
            {
                std::lock_guard<std::mutex> lock(mtx_);

                img_.copyTo(img);
                tracking_state = tracking_state_;

                // copy tracking information
                if (tracking_state == tracker_state_t::Initializing)
                {
                    init_keypts = init_keypts_;
                    init_matches = init_matches_;
                }

                curr_keypts = curr_keypts_;

                elapsed_ms = elapsed_ms_;

                mapping_is_enabled = mapping_is_enabled_;

                is_tracked = is_tracked_;

                if (map_db_->_b_use_line_tracking)
                {
                    curr_keylines = _curr_keylines;
                    is_tracked_line = _is_tracked_line;
                }
            }

            // resize image
            const float mag = (img_width_ < img_.cols) ? static_cast<float>(img_width_) / img.cols : 1.0;
            if (mag != 1.0)
            {
                cv::resize(img, img, cv::Size(), mag, mag, cv::INTER_NEAREST);
            }

            // to draw COLOR information
            if (img.channels() < 3)
            {
                cvtColor(img, img, cv::COLOR_GRAY2BGR);
            }

            // draw keypoints
            unsigned int num_tracked = 0;
            unsigned int num_tracked_line = 0;
            switch (tracking_state)
            {
            case tracker_state_t::Initializing:
            {
                num_tracked = draw_initial_points(img, init_keypts, init_matches, curr_keypts, mag);
                break;
            }
            case tracker_state_t::Tracking:
            {
                num_tracked = draw_tracked_points(img, curr_keypts, is_tracked, mapping_is_enabled, mag);

                // FW: also draw lines
                if (map_db_->_b_use_line_tracking)
                {
                    if (!_curr_keylines.empty() || !is_tracked_line.empty())
                    {
                        num_tracked_line = draw_tracked_lines(img, curr_keylines, is_tracked_line, mapping_is_enabled, mag);
                    }
                }

                break;
            }
            default:
            {
                break;
            }
            }

            if (draw_text)
            {
                // draw tracking info
                draw_info_text(img, tracking_state, num_tracked, num_tracked_line, elapsed_ms, mapping_is_enabled);
            }

            return img;
        }

        unsigned int frame_publisher::draw_initial_points(cv::Mat &img, const std::vector<cv::KeyPoint> &init_keypts,
                                                          const std::vector<int> &init_matches, const std::vector<cv::KeyPoint> &curr_keypts,
                                                          const float mag) const
        {
            unsigned int num_tracked = 0;

            for (unsigned int i = 0; i < init_matches.size(); ++i)
            {
                if (init_matches.at(i) < 0)
                {
                    continue;
                }

                cv::circle(img, init_keypts.at(i).pt * mag, 2, mapping_color_, -1);
                cv::circle(img, curr_keypts.at(init_matches.at(i)).pt * mag, 2, mapping_color_, -1);
                cv::line(img, init_keypts.at(i).pt * mag, curr_keypts.at(init_matches.at(i)).pt * mag, mapping_color_);

                ++num_tracked;
            }

            return num_tracked;
        }

        unsigned int frame_publisher::draw_tracked_points(cv::Mat &img, const std::vector<cv::KeyPoint> &curr_keypts,
                                                          const std::vector<bool> &is_tracked, const bool mapping_is_enabled,
                                                          const float mag) const
        {
            constexpr float radius = 5;

            unsigned int num_tracked = 0;

            for (unsigned int i = 0; i < curr_keypts.size(); ++i)
            {
                if (!is_tracked.at(i))
                {
                    continue;
                }

                const cv::Point2f pt_begin{curr_keypts.at(i).pt.x * mag - radius, curr_keypts.at(i).pt.y * mag - radius};
                const cv::Point2f pt_end{curr_keypts.at(i).pt.x * mag + radius, curr_keypts.at(i).pt.y * mag + radius};

                if (mapping_is_enabled)
                {
                    cv::rectangle(img, pt_begin, pt_end, mapping_color_);
                    cv::circle(img, curr_keypts.at(i).pt * mag, 2, mapping_color_, -1);
                }
                else
                {
                    cv::rectangle(img, pt_begin, pt_end, localization_color_);
                    cv::circle(img, curr_keypts.at(i).pt * mag, 2, localization_color_, -1);
                }

                ++num_tracked;
            }

            return num_tracked;
        }

        // FW:
        unsigned int frame_publisher::draw_tracked_lines(cv::Mat &img, const std::vector<cv::line_descriptor::KeyLine> &curr_keylines,
                                                         const std::vector<bool> &is_tracked_line, const bool mapping_is_enabled,
                                                         const float mag) const
        {
            unsigned int num_tracked = 0;

            if (curr_keylines.empty() || is_tracked_line.empty())
                return num_tracked;

            for (unsigned int i = 0; i < is_tracked_line.size(); i++)
            {
                cv::line_descriptor::KeyLine line = curr_keylines.at(i);
                cv::Point startPoint = cv::Point(int(line.startPointX * mag), int(line.startPointY * mag));
                cv::Point endPoint = cv::Point(int(line.endPointX * mag), int(line.endPointY * mag));

                if (!is_tracked_line.at(i))
                {
                    // lines not tracked (red color)
                    if (mapping_is_enabled)
                    {
                        cv::line(img, startPoint, endPoint, cv::Scalar(0.4 * 255, 0.35 * 255, 0.8 * 255), 1, 8);
                    }
                    else
                    {
                        // image re-localization mode
                        cv::line(img, startPoint, endPoint, cv::Scalar(0.4 * 255, 0.35 * 255, 0.8 * 255), 1, 8);
                    }
                }
                else
                {
                    // lines are tracked (bold, purple color)
                    if (mapping_is_enabled)
                    {
                        cv::line(img, startPoint, endPoint, cv::Scalar(0.8 * 255, 0.35 * 255, 0.4 * 255), 3, 8);
                    }
                    else
                    {
                        // image re-localization mode
                        cv::line(img, startPoint, endPoint, cv::Scalar(0.8 * 255, 0.35 * 255, 0.4 * 255), 3, 8);
                    }

                    ++num_tracked;
                }
            }

            return num_tracked;
        }

        void frame_publisher::draw_info_text(cv::Mat &img, const tracker_state_t tracking_state,
                                             const unsigned int num_tracked, const unsigned int num_tracked_line,
                                             const double elapsed_ms, const bool mapping_is_enabled) const
        {
            // create text info
            std::stringstream ss;
            switch (tracking_state)
            {
            case tracker_state_t::NotInitialized:
            {
                ss << "WAITING FOR IMAGES ";
                break;
            }
            case tracker_state_t::Initializing:
            {
                ss << "INITIALIZE | ";
                ss << "KP: " << num_tracked << ", ";
                ss << "track time: " << std::fixed << std::setprecision(0) << elapsed_ms << "ms";
                break;
            }
            case tracker_state_t::Tracking:
            {
                ss << (mapping_is_enabled ? "MAPPING | " : "LOCALIZATION | ");
                ss << "KF: " << map_db_->get_num_keyframes() << ", ";
                ss << "LM: " << map_db_->get_num_landmarks() << ", ";
                ss << "L3d: " << map_db_->get_num_landmarks_line() << ", "; // FW:
                ss << "KP: " << num_tracked << ", ";
                ss << "KL: " << num_tracked_line << ", "; // FW:
                ss << "track time: " << std::fixed << std::setprecision(0) << elapsed_ms << "ms";
                break;
            }
            case tracker_state_t::Lost:
            {
                ss << "LOST | ";
                ss << "track time: " << std::fixed << std::setprecision(0) << elapsed_ms << "ms";
                break;
            }
            }

            int baseline = 0;
            const cv::Size text_size = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

            // create text area
            constexpr float alpha = 0.6;
            cv::Mat overlay(img.rows, img.cols, img.type());
            img.copyTo(overlay);
            cv::rectangle(overlay, cv::Point2i{0, img.rows - text_size.height - 10}, cv::Point2i{img.cols, img.rows}, cv::Scalar{0, 0, 0}, -1);
            cv::addWeighted(overlay, alpha, img, 1 - alpha, 0, img);
            // put text
            cv::putText(img, ss.str(), cv::Point(5, img.rows - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar{255, 255, 255}, 1, 8);
        }

        void frame_publisher::update(tracking_module *tracker)
        {
            std::lock_guard<std::mutex> lock(mtx_);

            tracker->img_gray_.copyTo(img_);

            // FW: copy the segmentation mask to frame_publisher
            if (map_db_->_b_seg_or_not)
            {
                tracker->_seg_mask_img.copyTo(_seg_mask_img);
            }

            const auto num_curr_keypts = tracker->curr_frm_.num_keypts_;
            curr_keypts_ = tracker->curr_frm_.keypts_;
            elapsed_ms_ = tracker->elapsed_ms_;
            mapping_is_enabled_ = tracker->get_mapping_module_status();
            tracking_state_ = tracker->last_tracking_state_;

            is_tracked_ = std::vector<bool>(num_curr_keypts, false);

            // FW: also get current keylines
            unsigned int num_curr_keylines = 0;
            if (map_db_->_b_use_line_tracking)
            {
                num_curr_keylines = tracker->curr_frm_._keylsd.size();
                _curr_keylines = tracker->curr_frm_._keylsd;
                _is_tracked_line = std::vector<bool>(num_curr_keylines, false);
            }

            switch (tracking_state_)
            {
            case tracker_state_t::Initializing:
            {
                init_keypts_ = tracker->get_initial_keypoints();
                init_matches_ = tracker->get_initial_matches();
                break;
            }
            case tracker_state_t::Tracking:
            {
                for (unsigned int i = 0; i < num_curr_keypts; ++i)
                {
                    auto lm = tracker->curr_frm_.landmarks_.at(i);
                    if (!lm)
                    {
                        continue;
                    }
                    if (tracker->curr_frm_.outlier_flags_.at(i))
                    {
                        continue;
                    }

                    if (0 < lm->num_observations())
                    {
                        is_tracked_.at(i) = true;
                    }
                }

                // FW:
                if (map_db_->_b_use_line_tracking)
                {
                    if (!(num_curr_keylines == 0))
                    {

                        for (unsigned int k = 0; k < num_curr_keylines; k++)
                        {
                            auto lm_line = tracker->curr_frm_._landmarks_line.at(k);
                            if (!lm_line)
                            {
                                continue;
                            }
                            if (tracker->curr_frm_._outlier_flags_line.at(k))
                            {
                                continue;
                            }

                            if (lm_line->num_observations() > 0)
                            {
                                _is_tracked_line.at(k) = true;
                            }
                        }
                    }
                }

                break;
            }
            default:
            {
                break;
            }
            }
        }

        cv::Mat frame_publisher::draw_seg_mask()
        {
            return _seg_mask_img;
        }
    } // namespace publish
} // namespace PLPSLAM
