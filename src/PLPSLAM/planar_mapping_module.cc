/**
 * This file is part of Structure PLP-SLAM.
 *
 * Copyright 2022 DFKI (German Research Center for Artificial Intelligence)
 * Developed by Fangwen Shu <Fangwen.Shu@dfki.de>
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

#include "PLPSLAM/planar_mapping_module.h"
#include "PLPSLAM/data/frame.h"
#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/data/map_database.h"
#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/data/landmark_plane.h"

#include <spdlog/spdlog.h>
#include <random>

#include <opencv2/core/core.hpp>

// added libs for Graph-cut RANSAC
#include <opencv2/calib3d.hpp>
#include "PLPSLAM/solve/GCRANSAC/GCRANSAC.h"
#include "PLPSLAM/solve/GCRANSAC/flann_neighborhood_graph.h"
#include "PLPSLAM/solve/GCRANSAC/grid_neighborhood_graph.h"
#include "PLPSLAM/solve/GCRANSAC/uniform_sampler.h"
#include "PLPSLAM/solve/GCRANSAC/prosac_sampler.h"
#include "PLPSLAM/solve/GCRANSAC/progressive_napsac_sampler.h"
#include "PLPSLAM/solve/GCRANSAC/preemption_sprt.h"
#include "PLPSLAM/solve/GCRANSAC/types.h"
#include "PLPSLAM/solve/GCRANSAC/statistics.h"
#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

namespace PLPSLAM
{
    Planar_Mapping_module::Planar_Mapping_module(data::map_database *map_db, const bool is_monocular)
        : _map_db(map_db), _is_monocular(is_monocular)
    {
        // FW: this is a global flag indicates using instance planar segmentation
        // used in initialization, tracking, mapping, map_publisher and visualization
        _map_db->_b_seg_or_not = true;

        // see "./planar_mapping_parameters.yaml"
        load_configuration(_cfg_path);
    }

    Planar_Mapping_module::~Planar_Mapping_module()
    {
    }

    bool Planar_Mapping_module::initialize_map_with_plane(data::keyframe *keyfrm)
    {
        if (keyfrm->will_be_erased())
        {
            return false;
        }

        return process_new_kf(keyfrm);
    }

    bool Planar_Mapping_module::process_new_kf(data::keyframe *keyfrm)
    {
        if (keyfrm->will_be_erased())
        {
            return false;
        }

        std::lock_guard<std::mutex> lock(_mtxPlaneMutex);

        if (_setVerbose)
        {
            spdlog::info("-- PlanarMapping -- processing keyframe: id({})", keyfrm->id_);
        }

        // [1] estimate map scale, and calculate the geometric thresholds for RANSAC
        if (_is_monocular)
        {
            // monocular mode, the scale need to be estimated dynamicly
            estimate_map_scale(keyfrm); // estimate map scale as inverse of median depth of this keyframe
        }
        else
        {
            // RGB-D mode, the scale is (more or less) fixed
            if (_map_db->get_all_keyframes().size() < 3)
            {
                estimate_map_scale(); // estimate map scale as average of sum of the norm of all the map points coordinates
            }
        }

        // [2] create an unordered_map between the segmentation label and the potential 3D plane
        // the map points which corresponding to the 2D keypoints are linked to the potential plane for later RANSAC fitting
        auto before = std::chrono::high_resolution_clock::now();

        eigen_alloc_unord_map<long, data::Plane *> colorToPlanes;
        if (create_ColorToPlane(keyfrm, colorToPlanes))
        {
            auto after = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count();

            if (_setVerbose)
            {
                spdlog::info("\t \t | process time (create ColorPlane): {}ms", duration);
            }

            // [3] try to reconstruct 3D plane
            return create_new_plane(colorToPlanes);
        }

        return false;
    }

    // for Monocular
    void Planar_Mapping_module::estimate_map_scale(data::keyframe *keyfrm)
    {
        // the map scale is calculated as median depth of the current keyframe
        auto before = std::chrono::high_resolution_clock::now();

        const auto median_depth = keyfrm->compute_median_depth(true);
        _map_scale = 1 / median_depth;
        _planar_distance_thresh = PLANAR_DISTANCE_CORRECTION * _map_scale;
        _final_error_thresh = FINAL_ERROR_CORRECTION * _map_scale;

        auto after = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count();

        if (_setVerbose)
        {
            spdlog::info("\t \t | _map_scale: {}", _map_scale);
            spdlog::info("\t \t | _planar_distance_thresh: {}", _planar_distance_thresh);
            spdlog::info("\t \t | _final_error_thr: {}", _final_error_thresh);
            spdlog::info("\t \t | process time (scale): {}ms", duration);
        }
    }

    // for RGB-D
    void Planar_Mapping_module::estimate_map_scale()
    {
        // The map scale is calculated as the average of sum of world postion of all the map points
        auto before = std::chrono::high_resolution_clock::now();

        std::vector<data::landmark *> lms = _map_db->get_all_landmarks();
        double map_points_added = 0.0;
        for (auto const &lm : lms)
        {
            if (!lm->will_be_erased())
            {
                Vec3_t pos_w = lm->get_pos_in_world();
                map_points_added += pos_w.norm();
            }
        }

        _map_scale = map_points_added / lms.size();
        _planar_distance_thresh = PLANAR_DISTANCE_CORRECTION * _map_scale;
        _final_error_thresh = FINAL_ERROR_CORRECTION * _map_scale;

        auto after = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count();

        if (_setVerbose)
        {
            spdlog::info("\t \t | _map_scale: {}", _map_scale);
            spdlog::info("\t \t | _planar_distance_thresh: {}", _planar_distance_thresh);
            spdlog::info("\t \t | _final_error_thr: {}", _final_error_thresh);
            spdlog::info("\t \t | process time (scale): {}ms", duration);
        }
    }

    bool Planar_Mapping_module::create_ColorToPlane(data::keyframe *keyfrm,
                                                    eigen_alloc_unord_map<long, data::Plane *> &colorToPlanes)
    {
        if (keyfrm->will_be_erased())
        {
            return false;
        }

        cv::Mat segmentation_mask = keyfrm->get_segmentation_mask();
        std::set<data::landmark *> lms = keyfrm->get_valid_landmarks();

        if (segmentation_mask.empty() || lms.empty())
        {
            return false;
        }

        if (_setVerbose)
        {
            spdlog::info("\t \t | lms linked to this keyframe: {}", lms.size());
        }

        for (auto &lm : lms)
        {
            if (lm->will_be_erased() || lm->get_Owning_Plane())
            {
                continue;
            }

            // get corresponding keypoint
            int kpt_id = lm->get_index_in_keyframe(keyfrm);
            auto kpt = keyfrm->undist_keypts_[kpt_id];

            // this fixes the *Segmentation fault*
            // when running monocular SLAM on EuROC MAV dataset
            if (kpt.pt.y < 0 || kpt.pt.y > segmentation_mask.rows ||
                kpt.pt.x < 0 || kpt.pt.x > segmentation_mask.cols)
            {
                spdlog::debug("keypoint out of image range!");
                continue;
            }

            // get the color (the semantic label from segmentation) of this pixel
            auto center_color = segmentation_mask.at<cv::Vec3b>((int)kpt.pt.y, (int)kpt.pt.x);
            long pseudo_hash_center = center_color.val[0] + (center_color.val[1] << 8) + (center_color.val[2] << 16);

            // check if black color (non-plane)
            if (pseudo_hash_center == 0)
            {
                continue;
            }

            // check 3x3 window, or not
            if (!_check_3x3_window)
            {
                // check duplication
                if (colorToPlanes.empty())
                {
                    auto plane = new data::Plane(keyfrm, _map_db);
                    colorToPlanes[pseudo_hash_center] = plane;
                }
                else
                {
                    // avoid duplicate
                    if (!colorToPlanes.count(pseudo_hash_center))
                    {
                        auto plane = new data::Plane(keyfrm, _map_db);
                        colorToPlanes[pseudo_hash_center] = plane;
                    }
                }

                // link map point to the plane, and set ownership
                colorToPlanes[pseudo_hash_center]->add_landmark(lm);
            }
            else
            {
                //  check a small 3x3 window if all the pixel belongs to the same semantic class
                std::vector<cv::Vec3b> color_list;

                if ((int)kpt.pt.y + 1 > 0 && (int)kpt.pt.y + 1 < segmentation_mask.rows &&
                    (int)kpt.pt.x + 1 > 0 && (int)kpt.pt.x + 1 < segmentation_mask.cols)
                    color_list.push_back(segmentation_mask.at<cv::Vec3b>((int)kpt.pt.y + 1, (int)kpt.pt.x + 1)); // bottom right

                if ((int)kpt.pt.y - 1 > 0 && (int)kpt.pt.y - 1 < segmentation_mask.rows &&
                    (int)kpt.pt.x - 1 > 0 && (int)kpt.pt.x - 1 < segmentation_mask.cols)
                    color_list.push_back(segmentation_mask.at<cv::Vec3b>((int)kpt.pt.y - 1, (int)kpt.pt.x - 1)); // top left

                if ((int)kpt.pt.y + 1 > 0 && (int)kpt.pt.y + 1 < segmentation_mask.rows &&
                    (int)kpt.pt.x - 1 > 0 && (int)kpt.pt.x - 1 < segmentation_mask.cols)
                    color_list.push_back(segmentation_mask.at<cv::Vec3b>((int)kpt.pt.y + 1, (int)kpt.pt.x - 1)); // bottom left

                if ((int)kpt.pt.y - 1 > 0 && (int)kpt.pt.y - 1 < segmentation_mask.rows &&
                    (int)kpt.pt.x + 1 > 0 && (int)kpt.pt.x + 1 < segmentation_mask.cols)
                    color_list.push_back(segmentation_mask.at<cv::Vec3b>((int)kpt.pt.y - 1, (int)kpt.pt.x + 1)); // top right

                if ((int)kpt.pt.y + 1 > 0 && (int)kpt.pt.y + 1 < segmentation_mask.rows &&
                    (int)kpt.pt.x > 0 && (int)kpt.pt.x < segmentation_mask.cols)
                    color_list.push_back(segmentation_mask.at<cv::Vec3b>((int)kpt.pt.y + 1, (int)kpt.pt.x)); // bottom

                if ((int)kpt.pt.y - 1 > 0 && (int)kpt.pt.y - 1 < segmentation_mask.rows &&
                    (int)kpt.pt.x > 0 && (int)kpt.pt.x < segmentation_mask.cols)
                    color_list.push_back(segmentation_mask.at<cv::Vec3b>((int)kpt.pt.y - 1, (int)kpt.pt.x)); // top

                if ((int)kpt.pt.y > 0 && (int)kpt.pt.y < segmentation_mask.rows &&
                    (int)kpt.pt.x - 1 > 0 && (int)kpt.pt.x - 1 < segmentation_mask.cols)
                    color_list.push_back(segmentation_mask.at<cv::Vec3b>((int)kpt.pt.y, (int)kpt.pt.x - 1)); // left

                if ((int)kpt.pt.y > 0 && (int)kpt.pt.y < segmentation_mask.rows &&
                    (int)kpt.pt.x + 1 > 0 && (int)kpt.pt.x + 1 < segmentation_mask.cols)
                    color_list.push_back(segmentation_mask.at<cv::Vec3b>((int)kpt.pt.y, (int)kpt.pt.x + 1)); // right

                bool if_consistant_color = true;
                if (!color_list.empty())
                {
                    for (auto &color : color_list)
                    {
                        long pseudo_hash = color.val[0] + (color.val[1] << 8) + (color.val[2] << 16);
                        if (pseudo_hash == 0 || pseudo_hash_center != pseudo_hash)
                        {
                            // non-planar surfaces are black, all the pixel inside 3x3 window should have same label
                            if_consistant_color = false;
                            break;
                        }
                    }
                }

                if (if_consistant_color)
                {
                    // create a new plane if a new color (semantic label) found
                    if (colorToPlanes.empty())
                    {
                        auto plane = new data::Plane(keyfrm, _map_db);
                        colorToPlanes[pseudo_hash_center] = plane;
                    }
                    else
                    {
                        // avoid duplicate
                        if (!colorToPlanes.count(pseudo_hash_center))
                        {
                            auto plane = new data::Plane(keyfrm, _map_db);
                            colorToPlanes[pseudo_hash_center] = plane;
                        }
                    }

                    // link map point to the plane, and set ownership
                    colorToPlanes[pseudo_hash_center]->add_landmark(lm);
                }
            }
        }

        if (_setVerbose)
        {
            spdlog::info("\t \t | number of planes (colors) initialized by this keyframe: {}", colorToPlanes.size());
        }

        if (colorToPlanes.empty())
        {
            return false;
        }

        return true;
    }

    bool Planar_Mapping_module::create_new_plane(eigen_alloc_unord_map<long, data::Plane *> &colorToPlanes)
    {
        if (_setVerbose)
        {
            spdlog::info("-- PlanarMapping -- trying to create plane");
        }

        auto before = std::chrono::high_resolution_clock::now();

        bool hasNewPlanes = false;
        for (auto &color_plane_pair : colorToPlanes)
        {
            if (color_plane_pair.second->get_num_landmarks() < MIN_NUMBER_POINTS_BEFORE_RANSAC) // 12
            {
                delete color_plane_pair.second;
                color_plane_pair.second = nullptr;
                continue;
            }

            // Conditional ternary operator
            if (_use_graph_cut
                    ? estimate_plane_sequential_Graph_cut_RANSAC(color_plane_pair.second)
                    : estimate_plane_sequential_RANSAC(color_plane_pair.second))
            {
                hasNewPlanes = true;
                color_plane_pair.second->set_valid(); // set valid, but plane need to be refined

                if (_setVerbose)
                {
                    double a, b, c, d;
                    color_plane_pair.second->get_equation(a, b, c, d);
                    spdlog::info("\t \t | new plane detected: ({}, {}, {}, {})", a, b, c, d);
                }

                // save to map database
                _map_db->add_landmark_plane(color_plane_pair.second);
            }
            else
            {
                color_plane_pair.second->remove_landmarks_ownership();

                // delete pointer first, then set as null
                delete color_plane_pair.second;
                color_plane_pair.second = nullptr;

                if (_setVerbose)
                {
                    spdlog::info("\t \t | estimate plane through RANSAC failed");
                }
            }
        }

        colorToPlanes.clear();

        auto after = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count();

        if (_setVerbose)
        {
            spdlog::info("\t \t | process time (RANSAC+SVD): {}ms", duration);
        }

        return hasNewPlanes;
    }

    bool Planar_Mapping_module::estimate_plane_sequential_RANSAC(data::Plane *plane)
    {
        std::vector<data::landmark *> lms = plane->get_landmarks();

        if (_setVerbose)
        {
            spdlog::info("\t \t | ----");
            spdlog::info("\t \t | estimate plane RANSAC");
            spdlog::info("\t \t | number of map points linked to the plane: {} ", lms.size());
        }

        if (lms.empty())
        {
            return false;
        }

        if (lms.size() < POINTS_PER_RANSAC)
        {
            if (_setVerbose)
            {
                spdlog::info("\t \t | Not enough points!");
            }

            return false;
        }

        // randomly choose points for RANSAC, save as index
        double best_error = std::numeric_limits<double>::max();
        std::vector<int> inliers_list;
        std::vector<int> best_inliers_list;
        bool best_found = false;
        double a, b, c, d;
        double a_best, b_best, c_best, d_best;
        std::function<int()> rnd = std::bind(std::uniform_int_distribution<>(0, lms.size() - 1), std::mt19937(std::random_device()()));
        for (unsigned int i = 0; i < _iterationsCount; i++)
        {
            std::vector<int> indexes;
            while (indexes.size() < POINTS_PER_RANSAC)
            {
                int index = rnd(); // random index
                if (!lms[index]->will_be_erased())
                {
                    indexes.push_back(index);
                }
            }

            if (indexes.empty())
            {
                break;
            }

            // [1] estimate initial plane parameter from random selected lms
            double residual = estimate_plane_SVD(lms, indexes, a, b, c, d);
            if (residual < best_error)
            {
                best_error = residual;
            }
            a_best = a;
            b_best = b;
            c_best = c;
            d_best = d;
            plane->set_equation(a_best, b_best, c_best, d_best);
            plane->set_best_error(residual);

            // [2] find inlier 3D points by calculating the distance between lms to the plane
            inliers_list.clear();
            for (unsigned int j = 0; j < lms.size(); j++)
            {
                if (lms[j]->will_be_erased())
                {
                    continue;
                }

                // calculate point-plane distance
                double point2plane_error = plane->calculate_distance(lms[j]->get_pos_in_world());
                if (point2plane_error < _planar_distance_thresh)
                {
                    inliers_list.push_back(j);
                }
            }

            // [3] try to update best parameters which minimize the distance between all plane-linked (3D) points to the estimated plane
            if (_ransacMode == RANSAC_SMALLEST_DISTANCE_ERROR)
            {
                const double inlier_ratio = double(inliers_list.size()) / double(lms.size());

                if (inlier_ratio > _inliers_ratio_thr &&
                    inliers_list.size() >= POINTS_PER_RANSAC)
                {
                    // estimate plane parameters again using all the inliers found before
                    auto error = estimate_plane_SVD(lms, inliers_list, a, b, c, d);

                    // if new error is better than before, then update parameters
                    if (error < best_error)
                    {
                        best_error = error;
                        a_best = a;
                        b_best = b;
                        c_best = c;
                        d_best = d;
                        plane->set_equation(a_best, b_best, c_best, d_best);
                        plane->set_best_error(best_error);

                        // find best inlier 3D points
                        best_inliers_list.clear();
                        for (unsigned int z = 0; z < inliers_list.size(); z++)
                        {
                            int index = inliers_list[z];
                            best_inliers_list.push_back(std::move(index));
                        }

                        best_found = true;

                        // early break if the error is small enough
                        if (error < _final_error_thresh)
                        {
                            if (_setVerbose)
                            {
                                spdlog::info("\t \t | early break in RANSAC when error < _final_error_thresh");
                            }

                            break;
                        }
                    }
                }
            }
            else if (_ransacMode == RANSAC_HIGHEST_INLIER_RATIO)
            {
                // FW: for now, we try to find plane which minimize the geometric threshold (point2plane distance)
                //  this mode indicates try to find more 3D lms to fit the plane (high inlier ratio), which is not used so far
            }
        } // end of ransac iteration

        if (!best_found)
        {
            if (_setVerbose)
            {
                spdlog::info("\t \t | plane not founded");
            }

            return false;
        }
        else if (best_error > _final_error_thresh)
        {
            if (_setVerbose)
            {
                spdlog::info("\t \t | error is higher than threshold");
            }

            return false;
        }

        // [4] assign the best lms to the plane
        std::vector<data::landmark *> plane_best_inlier_map_points;
        for (unsigned int i = 0; i < best_inliers_list.size(); i++)
        {
            auto lm = lms[best_inliers_list[i]];
            if (lm->will_be_erased())
            {
                continue;
            }

            double dis = plane->calculate_distance(lm->get_pos_in_world());
            if (dis < _planar_distance_thresh)
            {
                plane_best_inlier_map_points.push_back(lm);
            }
        }

        if (_setVerbose)
        {
            spdlog::info("\t \t | best inlier points: {}", plane_best_inlier_map_points.size());
        }

        plane->remove_landmarks_ownership();
        plane->set_landmarks(plane_best_inlier_map_points);
        plane->set_landmarks_ownership();

        return true;
    }

    bool Planar_Mapping_module::update_plane_via_RANSAC(data::Plane *plane)
    {
        std::vector<data::landmark *> lms = plane->get_landmarks();

        if (lms.empty())
        {
            return false;
        }

        if (lms.size() < POINTS_PER_RANSAC)
        {
            plane->set_invalid();
            return false;
        }

        // randomly choose points for RANSAC, save as index
        double best_error = plane->get_best_error();
        std::vector<int> inliers_list;
        std::vector<int> best_inliers_list;
        bool best_found = false;
        double a, b, c, d;
        double a_best, b_best, c_best, d_best;
        std::function<int()> rnd = std::bind(std::uniform_int_distribution<>(0, lms.size() - 1), std::mt19937(std::random_device()()));
        for (unsigned int i = 0; i < _iterationsCount; i++)
        {
            std::vector<int> indexes;
            // we use more than half of the points to update plane parameters i.e. after merge
            while (indexes.size() < lms.size() * 0.8)
            {
                int index = rnd(); // random index
                if (!lms[index]->get_Owning_Plane())
                {
                    continue;
                }

                if (!lms[index]->will_be_erased())
                {
                    indexes.push_back(index);
                }
            }

            if (indexes.empty())
            {
                break;
            }

            // [1] estimate initial plane parameter from random selected lms
            double residual = estimate_plane_SVD(lms, indexes, a, b, c, d);
            if (residual < best_error)
            {
                best_error = residual;
            }
            a_best = a;
            b_best = b;
            c_best = c;
            d_best = d;
            plane->set_equation(a_best, b_best, c_best, d_best);
            plane->set_best_error(residual);

            // [2] find inlier 3D points by calculating the distance between lms to the plane
            inliers_list.clear();
            for (unsigned int j = 0; j < lms.size(); j++)
            {
                if (lms[j]->will_be_erased())
                {
                    continue;
                }

                // calculate point-plane distance
                double point2plane_error = plane->calculate_distance(lms[j]->get_pos_in_world());
                if (point2plane_error < _planar_distance_thresh)
                {
                    inliers_list.push_back(j);
                }
            }

            // [3] try to update best parameters which minimize the distance between all plane-linked (3D) points to the estimated plane
            if (inliers_list.size() >= POINTS_PER_RANSAC)
            {
                auto error = estimate_plane_SVD(lms, inliers_list, a, b, c, d);

                // early break if the error is smaller than before
                if (error < best_error)
                {
                    best_error = error;
                    a_best = a;
                    b_best = b;
                    c_best = c;
                    d_best = d;
                    plane->set_equation(a_best, b_best, c_best, d_best);
                    plane->set_best_error(best_error);

                    // find best inlier 3D points
                    best_inliers_list.clear();
                    for (unsigned int z = 0; z < inliers_list.size(); z++)
                    {
                        int index = inliers_list[z];
                        best_inliers_list.push_back(std::move(index));
                    }

                    best_found = true;
                }
            }
        } // end of ransac iteration

        if (!best_found || best_error > _final_error_thresh)
        {
            plane->set_need_refinement();
            return false;
        }

        // [4] assign the best lms to the plane
        std::vector<data::landmark *> plane_best_inlier_map_points;
        for (unsigned int i = 0; i < best_inliers_list.size(); i++)
        {
            auto lm = lms[best_inliers_list[i]];

            if (lm->will_be_erased())
            {
                continue;
            }

            double dis = plane->calculate_distance(lm->get_pos_in_world());
            if (dis < _planar_distance_thresh)
            {
                plane_best_inlier_map_points.push_back(lm);
            }
        }

        if (plane_best_inlier_map_points.size() < POINTS_PER_RANSAC)
        {
            plane->set_invalid();
            return false;
        }

        plane->remove_landmarks_ownership();
        plane->set_landmarks(plane_best_inlier_map_points);
        plane->set_landmarks_ownership();

        return true;
    }

    double Planar_Mapping_module::estimate_plane_SVD(std::vector<data::landmark *> const &all_plane_points, std::vector<int> indexes, double &a, double &b, double &c, double &d)
    {
        std::vector<data::landmark *> candidate_points;
        candidate_points.reserve(indexes.size());
        for (unsigned int i = 0; i < indexes.size(); i++)
        {
            candidate_points.push_back(all_plane_points.at(indexes[i]));
        }

        // build a linear system for plane
        Eigen::Matrix<double, 3, Eigen::Dynamic> X(3, candidate_points.size());

        for (unsigned int i = 0; i < candidate_points.size(); i++)
        {
            X.col(i) = candidate_points[i]->get_pos_in_world();
        }

        // FW: why we need to centralize the coordinates?
        Vec3_t centroid = X.rowwise().mean();
        Eigen::Matrix<double, 3, Eigen::Dynamic> centered_X = X.colwise() - centroid;

        // Singular Value Decomposition
        Eigen::JacobiSVD<Eigen::Matrix<double, 3, Eigen::Dynamic>> svd(centered_X, Eigen::ComputeFullU);
        Eigen::Matrix3d const U = svd.matrixU();
        Eigen::Vector3d normal = U.col(2).normalized();
        d = -normal.dot(centroid);
        a = normal(0);
        b = normal(1);
        c = normal(2);

        // not really a residual because we average it according to the total n# of points
        // as having a bigger residual with more points is expected...
        Eigen::Matrix<double, Eigen::Dynamic, 1> const b_vector = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(X.cols(), 1) * d;
        double const residual = ((X.transpose() * normal) + b_vector).norm() / candidate_points.size();

        return std::fabs(residual);
    }

    void Planar_Mapping_module::refinement()
    {
        std::lock_guard<std::mutex> lock(_mtxPlaneMutex);

        if (_map_db->get_num_landmark_planes() < 2)
        {
            return;
        }

        // merge if two plane have similar normal vector or very close to each other, etc.
        if (merge_planes())
        {
            if (_setVerbose)
            {
                spdlog::info("\t \t | merge succuss!");
            }
        }

        refine_planes(); // re-estimate plane parameters for the plane which need to be refined, e.g. after merge
        refine_points(); // TODO: this should be only down in local BA?
    }

    bool Planar_Mapping_module::merge_planes()
    {
        auto all_planes = _map_db->get_all_landmark_planes();

        if (all_planes.empty())
        {
            return false;
        }

        // at least two planes should exist
        if (all_planes.size() < 2)
        {
            return false;
        }

        // Two constants are used to decide whether or not two planes are equal
        // DOT_PRODUCT_THRESHOLD is used when comparing the angle between two planes
        // OFFSET_DELTA_THRESHOLD is used when comparing the offset between two planes
        OFFSET_DELTA_THRESHOLD = _offset_delta_factor * _planar_distance_thresh;

        std::vector<data::Plane *> planes_will_be_removed; // a container used to aggregate merged planes or outlier planes
        bool was_merged = false;
        for (unsigned int i = 0; i < all_planes.size(); i++)
        {
            // merge_parent: all_planes[i]
            if (!all_planes[i]->is_valid())
            {
                continue;
            }

            // merge_candidate: all_planes[j]
            for (unsigned int j = i + 1; j < all_planes.size(); j++)
            {
                if (!all_planes[j]->is_valid())
                {
                    j++;
                    if (j >= all_planes.size())
                    { // avoid out of range issue
                        break;
                    }
                }

                // get plane normal
                Vec3_t parent_normal = all_planes[i]->get_normal();
                Vec3_t candidate_normal = all_planes[j]->get_normal();

                // get plane normal's norm
                double parent_normalizer = 1.0 / all_planes[i]->get_normal_norm();
                double candidate_normalizer = 1.0 / all_planes[j]->get_normal_norm();

                // get plane distance to the origin
                double parent_offset = all_planes[i]->get_offset();
                double candidate_offset = all_planes[j]->get_offset();

                // calculate geometric thresholds
                double offset_delta = parent_offset * parent_normalizer - candidate_offset * candidate_normalizer;
                double normalized_dot_product = parent_normal.dot(candidate_normal) * (parent_normalizer * candidate_normalizer);

                if (std::fabs(offset_delta) > OFFSET_DELTA_THRESHOLD &&
                    std::fabs(normalized_dot_product) < DOT_PRODUCT_THRESHOLD)
                {
                    // if two planes are not close to each other, or vary a lot in terms of normal
                    continue;
                }
                else if (std::fabs(offset_delta) < OFFSET_DELTA_THRESHOLD &&
                         std::fabs(normalized_dot_product) > DOT_PRODUCT_THRESHOLD)
                {
                    // if two planes are close to each other, and normals are nearly parallel (dot product ~= 1)
                    // merge
                    if (all_planes[i]->get_num_landmarks() > all_planes[j]->get_num_landmarks())
                    {
                        all_planes[i]->merge(all_planes[j]);             // transfer the point' ownership to the merge_parent
                        planes_will_be_removed.push_back(all_planes[j]); // aggregate planes which were merged

                        // update information of merged plane
                        update_plane_via_RANSAC(all_planes[i]);
                        all_planes[i]->set_need_refinement();
                    }
                    else
                    {
                        all_planes[j]->merge(all_planes[i]);
                        planes_will_be_removed.push_back(all_planes[i]);

                        // update information of merged plane
                        update_plane_via_RANSAC(all_planes[j]);
                        all_planes[j]->set_need_refinement();
                    }

                    was_merged = true;
                }
            }
        }

        // remove merged planes (which labeled as outlier before) from map_database
        if (was_merged)
        {
            for (auto &pl : planes_will_be_removed)
            {
                _map_db->erase_landmark_plane(pl);
            }
        }

        return was_merged;
    }

    bool Planar_Mapping_module::refine_planes()
    {
        bool was_changed = false;
        std::vector<data::Plane *> planes_will_be_erased;

        for (auto &plane : _map_db->get_all_landmark_planes())
        {
            if (!plane->is_valid())
            {
                planes_will_be_erased.push_back(plane);
                continue;
            }

            if (plane->need_refinement())
            {
                // a merged plane will be labeled as need refinement
                // update the plane equation using RANSAC
                if (update_plane_via_RANSAC(plane))
                {
                    plane->set_refinement_is_done();
                    was_changed = true;
                }
            }
            else
            {
                // check if the plane is very small
                if (plane->get_num_landmarks() < 2 * POINTS_PER_RANSAC)
                {
                    planes_will_be_erased.push_back(plane);
                }
            }
        }

        // erase outlier plane from map_database
        if (!planes_will_be_erased.empty())
        {
            for (auto &pl : planes_will_be_erased)
            {
                pl->remove_landmarks_ownership();
                _map_db->erase_landmark_plane(pl);
            }
        }

        if (_setVerbose)
        {
            if (was_changed)
            {
                spdlog::info("\t \t | plane refined!");
            }
        }

        return was_changed;
    }

    bool Planar_Mapping_module::refine_points()
    {
        bool was_changed = false;

        for (auto &plane : _map_db->get_all_landmark_planes())
        {
            if (!plane->is_valid() ||
                plane->need_refinement() ||
                plane->get_num_landmarks() < POINTS_PER_RANSAC)
            {
                continue;
            }

            auto lms = plane->get_landmarks();
            for (auto &lm : lms)
            {
                if (lm->will_be_erased() || !lm->get_Owning_Plane())
                {
                    continue;
                }

                Vec3_t pos_w = lm->get_pos_in_world();
                double dist_old = plane->calculate_distance(pos_w);
                if (std::fabs(dist_old) > 0.0)
                {
                    // if a point is assumed on the plane, then it should move along the normal direction
                    // and minimize the distance between plane
                    // update 3D coordinates along the normal direction
                    Vec3_t n = plane->get_normal().normalized();
                    pos_w -= n * dist_old;
                    double dist_new = plane->calculate_distance(pos_w);
                    if (dist_new < _planar_distance_thresh &&
                        dist_new < dist_old)
                    {
                        lm->set_pos_in_world(pos_w);
                        was_changed = true;
                    }
                }
            }
        }

        if (_setVerbose)
        {
            if (was_changed)
            {
                spdlog::info("\t \t | points refined!");
            }
        }

        return was_changed;
    }

    bool Planar_Mapping_module::estimate_plane_sequential_Graph_cut_RANSAC(data::Plane *plane)
    {
        std::vector<data::landmark *> lms = plane->get_landmarks();

        if (_setVerbose)
        {
            spdlog::info("\t \t | ----");
            spdlog::info("\t \t | estimate plane GC-RANSAC");
            spdlog::info("\t \t | number of map points linked to the plane: {} ", lms.size());
        }

        if (lms.empty())
        {
            return false;
        }

        if (lms.size() < POINTS_PER_RANSAC)
        {
            if (_setVerbose)
            {
                spdlog::info("\t \t | Not enough points!");
            }

            return false;
        }

        // 3D points in cv::Mat
        cv::Mat points(0, 3, CV_64F), point(1, 3, CV_64F);
        for (unsigned int i = 0; i < lms.size(); i++)
        {
            auto lm = lms[i];
            if (lm->will_be_erased())
            {
                continue;
            }

            auto pos_w = lm->get_pos_in_world();
            point.at<double>(0) = pos_w(0);
            point.at<double>(1) = pos_w(1);
            point.at<double>(2) = pos_w(2);
            points.push_back(point);
        }

        // adaptive parameters
        if (adaptive_number_ != 0)
        {
            _inlier_outlier_threshold = _planar_distance_thresh;
            _sphere_radius = adaptive_number_ * _final_error_thresh; // used to construct neighborhood graph
        }

        // [1] Apply Graph-cut RANSAC
        gcransac::utils::Default3DPlaneEstimator estimator; // The estimator used for the pose fitting
        gcransac::Plane3D model;                            // The estimated model parameters

        // Initialize the neighborhood used in Graph-cut RANSAC
        // FW: TODO: the graph may also be constructed by 3D grid (GridNeighborhoodGraph<3>)?
        gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points,         // The data points
                                                                    _sphere_radius); // The radius of the neighborhood sphere used for determining the neighborhood structure

        // Initialize the samplers
        // FW: TODO: should the sampler of NAPSAC or Progressive-NAPSAC used?
        gcransac::sampler::UniformSampler main_sampler(&points);               // The main sampler is used inside the local optimization
        gcransac::sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

        // Checking if the samplers are initialized successfully.
        if (!main_sampler.isInitialized() ||
            !local_optimization_sampler.isInitialized())
        {
            fprintf(stderr, "One of the samplers is not initialized successfully.\n");
            return false;
        }

        // Initializing SPRT test
        gcransac::preemption::SPRTPreemptiveVerfication<gcransac::utils::Default3DPlaneEstimator> preemptive_verification(
            points,                          // The set of 3D points
            estimator,                       // The linear estimator object (2D line or 3D plane)
            _minimum_inlier_ratio_for_sprt); // The minimum acceptable inlier ratio. Models with fewer inliers will not be accepted.

        gcransac::GCRANSAC<gcransac::utils::Default3DPlaneEstimator,
                           gcransac::neighborhood::FlannNeighborhoodGraph,
                           gcransac::MSACScoringFunction<gcransac::utils::Default3DPlaneEstimator>,
                           gcransac::preemption::SPRTPreemptiveVerfication<gcransac::utils::Default3DPlaneEstimator>>
            gcransac;
        gcransac.setFPS(_fps);
        gcransac.settings.threshold = _inlier_outlier_threshold;                // The inlier-outlier threshold
        gcransac.settings.spatial_coherence_weight = _spatial_coherence_weight; // The weight of the spatial coherence term
        gcransac.settings.confidence = _confidence;                             // The required confidence in the results
        gcransac.settings.max_iteration_number = 5000;                          // The maximum number of iterations
        gcransac.settings.min_iteration_number = 20;                            // The minimum number of iterations

        // Start GC-RANSAC
        gcransac.run(points,                      // The normalized points
                     estimator,                   // The estimator
                     &main_sampler,               // The sample used for selecting minimal samples in the main iteration
                     &local_optimization_sampler, // The sampler used for selecting a minimal sample when doing the local optimization
                     &neighborhood,               // The neighborhood-graph
                     model,                       // The obtained model parameters
                     preemptive_verification);

        // Get the statistics of the results
        const gcransac::utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();

        // Debug
        if (_setVerbose)
        {
            spdlog::info("\t \t | Elapsed time = {} secs", statistics.processing_time);
            spdlog::info("\t \t | Inlier number = {}", static_cast<int>(statistics.inliers.size()));
            spdlog::info("\t \t | Total number = {}", points.rows);
            spdlog::info("\t \t | Applied number of local optimizations = {}", static_cast<int>(statistics.local_optimization_number));
            spdlog::info("\t \t | Applied number of graph-cuts = {}", static_cast<int>(statistics.graph_cut_number));
            spdlog::info("\t \t | Number of iterations = {}", static_cast<int>(statistics.iteration_number));
        }

        plane->set_equation(model.descriptor(0, 0), model.descriptor(0, 1), model.descriptor(0, 2), model.descriptor(0, 3));

        // reject model if inlier ratio is lower than i.e. 70%
        const double inlier_ratio = double(statistics.inliers.size()) / double(lms.size());
        if (inlier_ratio < _inliers_ratio_thr && (statistics.inliers.size() < POINTS_PER_RANSAC))
        {
            return false;
        }

        // [2] find inlier 3D points by calculating the distance between lms to the plane
        double best_error = std::numeric_limits<double>::max();
        std::vector<int> inliers_list;
        std::vector<data::landmark *> plane_best_inlier_map_points;
        for (unsigned int i = 0; i < statistics.inliers.size(); i++)
        {
            inliers_list.push_back(statistics.inliers[i]); // do this just for convenience because our code was implemented with "int" instead of "unsigned int"
            auto lm = lms[statistics.inliers[i]];
            plane_best_inlier_map_points.push_back(lm);
        }

        // [3] try to update best parameters which minimize the distance between all plane-linked (3D) points to the estimated plane
        // estimate plane parameters again using all the inliers found before
        double a, b, c, d;
        auto error = estimate_plane_SVD(lms, inliers_list, a, b, c, d);
        if (error < best_error)
        {
            best_error = error;
            plane->set_equation(a, b, c, d);
            plane->set_best_error(best_error);
        }

        // [4] assign the best lms to the plane
        plane->remove_landmarks_ownership();
        plane->set_landmarks(plane_best_inlier_map_points);
        plane->set_landmarks_ownership();

        return true;
    }

    void Planar_Mapping_module::load_configuration(const std::string path)
    {
        YAML::Node yaml_node = YAML::LoadFile(path);

        _use_graph_cut = yaml_node["Threshold.use_graph_cut"].as<bool>();
        _setVerbose = yaml_node["Threshold.setVerbose"].as<bool>();

        _iterationsCount = yaml_node["Threshold.iterationsCount"].as<unsigned int>();
        _inliers_ratio_thr = yaml_node["Threshold.inliers_ratio_thr"].as<double>();
        MIN_NUMBER_POINTS_BEFORE_RANSAC = yaml_node["Threshold.min_number_points_before_ransac"].as<unsigned int>();
        POINTS_PER_RANSAC = yaml_node["Threshold.point_per_ransac"].as<unsigned int>();

        _check_3x3_window = yaml_node["Threshold.check_3x3_window"].as<bool>();

        DOT_PRODUCT_THRESHOLD = yaml_node["Threshold.dot_product_threshold"].as<double>();
        _offset_delta_factor = yaml_node["Threshold.offset_delta_factor"].as<int>();

        PLANAR_DISTANCE_CORRECTION = yaml_node["Threshold.plane_distance_correction"].as<double>();
        FINAL_ERROR_CORRECTION = yaml_node["Threshold.final_error_correction"].as<double>();

        _confidence = yaml_node["Threshold.confidence"].as<double>();
        _fps = yaml_node["Threshold.fps_limit"].as<int>();
        _inlier_outlier_threshold = yaml_node["Threshold.inlier_outlier_threshold"].as<double>();
        _spatial_coherence_weight = yaml_node["Threshold.spatial_coherence_weight"].as<double>();
        _sphere_radius = yaml_node["Threshold.sphere_radius"].as<double>();
        _minimum_inlier_ratio_for_sprt = yaml_node["Threshold.minimum_inlier_ratio_for_sprt"].as<double>();
        adaptive_number_ = yaml_node["Threshold.adaptive_number"].as<int>();
    }
} // namespace PLPSLAM