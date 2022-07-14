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

#ifndef PLPSLAM_DATA_LANDMARK_LINE_H
#define PLPSLAM_DATA_LANDMARK_LINE_H

#include <opencv2/features2d.hpp>
#include "PLPSLAM/feature/line_descriptor/line_descriptor_custom.hpp"
#include "PLPSLAM/camera/base.h"
#include "PLPSLAM/camera/perspective.h"
#include <eigen3/Eigen/Dense>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "PLPSLAM/type.h"

#include <map>
#include <mutex>
#include <atomic>

#include <nlohmann/json_fwd.hpp>

namespace PLPSLAM
{

    namespace data
    {

        class frame;
        class keyframe;
        class map_database;

        // FW: this class is similar to the class for 3D point -> landmark.h
        class Line
        {
        public:
            // Constructor for 3D line triangulated from keyframes
            Line(const Vec6_t &pos_w, keyframe *ref_keyfrm, map_database *map_db);

            // Constructor for map loading
            Line(const unsigned int id, const unsigned int first_keyfrm_id,
                 const Vec6_t &pos_w, keyframe *ref_keyfrm,
                 const unsigned int num_visible, const unsigned int num_found,
                 map_database *map_db);

            void set_pos_in_world(const Vec6_t &pos_w);                         // set endpoints of this line
            void set_pos_in_world_without_update_pluecker(const Vec6_t &pos_w); // set endpoints of this line (estimated by endpoints trimming), without updating Plücker coordinates
            Vec6_t get_pos_in_world() const;                                    // get endpoints of this line

            void toPlueckerCoord();                                            // set Plücker coordinates after line is triangulated
            void set_PlueckerCoord_without_update_endpoints(Vec6_t &pluecker); // update Plücker coordinates during BA, without updating endpoints (which should be re-estimated by endpoints trimming)
            Vec6_t get_PlueckerCoord() const;                                  // get Plücker coordinates

            keyframe *get_ref_keyframe() const; // get reference keyframe, a keyframe at the creation of a given 3D line

            void add_observation(keyframe *keyfrm, unsigned int idx); // add observation
            void erase_observation(keyframe *keyfrm);                 // erase observation

            std::map<keyframe *, unsigned int> get_observations() const; // get observations (keyframe and keyline idx)
            unsigned int num_observations() const;                       // get number of observations
            bool has_observation() const;                                // whether this landmark is observed from more than zero keyframes

            int get_index_in_keyframe(keyframe *keyfrm) const;    // get index of associated keyline in the specified keyframe
            bool is_observed_in_keyframe(keyframe *keyfrm) const; // whether this landmark is observed in the specified keyframe

            cv::Mat get_descriptor() const; // get representative descriptor
            void compute_descriptor();      // compute representative descriptor

            void prepare_for_erasing(); // erase this landmark from database
            bool will_be_erased();      // whether this landmark will be erased shortly or not

            void update_information(); // calculate maximum and minimum valid distance

            float get_min_valid_distance() const; // get max valid distance between landmark and camera
            float get_max_valid_distance() const; // get min valid distance between landmark and camera

            /**
             * @brief predict scale level assuming this landmark is observed in the specified frame/keyframe
             *
             * @param current_dist cam_to_lm_dist
             * @param log_scale_factor the log(scale_factor) of image pyramid where feature extracted, passed from keyframe or frame
             * @param num_scale_levels the number of levels of image pyramid, passed from keyframe or frame
             * @return unsigned int, the predicted level of image pyramid
             */
            unsigned int predict_scale_level(const float &current_dist, const float &log_scale_factor, const unsigned int &num_scale_levels);

            void replace(Line *line);   // replace this with specified landmark
            Line *get_replaced() const; // get replace landmark

            void increase_num_observable(unsigned int num_observable = 1); // 3D line is observable by a frame (isInFrustum), but not necessarily have a 2D match
            void increase_num_observed(unsigned int num_observed = 1);     // 3D line is observed with a 2D match
            float get_observed_ratio() const;                              // used in local_map_cleaner::remove_redundant_landmarks_line()

            // check the distance between landmark and camera is in scale variance
            inline bool is_inside_in_feature_scale(const float cam_to_lm_dist) const
            {
                const float max_dist = this->get_max_valid_distance();
                const float min_dist = this->get_min_valid_distance();
                return (min_dist <= cam_to_lm_dist && cam_to_lm_dist <= max_dist);
            }

            // encode landmark information as JSON
            nlohmann::json to_json() const;

        public:
            unsigned int _id;                          // initialized in the constructor
            static std::atomic<unsigned int> _next_id; // initialized in the constructor
            unsigned int _first_keyfrm_id = 0;         // initialized in the constructor
            unsigned int _first_frame_id = 0;          // initialized in the constructor

            unsigned int _num_observations = 0;

            // Variables for frame tracking.
            Vec2_t _reproj_in_tracking_sp;                    // used in tracking_module::search_local_landmarks_line()
            Vec2_t _reproj_in_tracking_ep;                    // used in tracking_module::search_local_landmarks_line()
            bool _is_observable_in_tracking;                  // used in tracking_module::search_local_landmarks_line()
            int _scale_level_in_tracking;                     // used in tracking_module::search_local_landmarks_line()
            unsigned int _identifier_in_local_map_update = 0; // used in local_map_updater::find_local_landmarks_line()
            unsigned int _identifier_in_local_lm_search = 0;  // used in tracking_module::search_local_landmarks_line()

            // Variables used by loop closing, Essential graph optimization, and global BA
            unsigned int _loop_fusion_identifier = 0;
            unsigned int _ref_keyfrm_id_in_loop_fusion = 0;
            Vec6_t _pos_w_after_global_BA; // pluecker coordinates
            unsigned int _loop_BA_identifier = 0;

        private:
            Vec6_t _pos_w;                // initialized in the constructor
            Vec6_t _pluecker_coordinates; // initialized in the constructor

            // reference keyframe;
            keyframe *_ref_keyfrm; // initialized in the constructor

            // map database
            map_database *_map_db; // initialized in the constructor

            // observations (keyframe and corresponding keyline index in this keyframe)
            std::map<keyframe *, unsigned int> _observations;

            // representative descriptor
            cv::Mat _descriptor;

            // track counter
            unsigned int _num_observable = 1;
            unsigned int _num_observed = 1;

            // this landmark will be erased shortly or not
            bool _will_be_erased = false;

            // replace this landmark with below
            Line *_replaced = nullptr;

            // max valid distance between landmark and camera
            float _min_valid_dist = 0;
            // min valid distance between landmark and camera
            float _max_valid_dist = 0;

            mutable std::mutex _mtx_position;
            mutable std::mutex _mtx_observations;
        };
    }
}

#endif