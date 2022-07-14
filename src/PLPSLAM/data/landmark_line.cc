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

#include "PLPSLAM/data/landmark_line.h"

#include "PLPSLAM/data/frame.h"
#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/data/map_database.h"
#include "PLPSLAM/match/base.h"

#include <nlohmann/json.hpp>

namespace PLPSLAM
{
    namespace data
    {
        std::atomic<unsigned int> Line::_next_id{0};

        Line::Line(const Vec6_t &pos_w, keyframe *ref_keyfrm, map_database *map_db)
            : _id(_next_id++),
              _first_keyfrm_id(ref_keyfrm->id_),
              _pos_w(pos_w),
              _ref_keyfrm(ref_keyfrm),
              _map_db(map_db)
        {

            toPlueckerCoord();
        }

        Line::Line(const unsigned int id, const unsigned int first_keyfrm_id,
                   const Vec6_t &pos_w, keyframe *ref_keyfrm,
                   const unsigned int num_visible, const unsigned int num_found,
                   map_database *map_db)
            : _id(id), _first_keyfrm_id(first_keyfrm_id), _pos_w(pos_w), _ref_keyfrm(ref_keyfrm),
              _num_observable(num_visible), _num_observed(num_found), _map_db(map_db)
        {
            toPlueckerCoord();
        }

        void Line::set_pos_in_world(const Vec6_t &pos_w)
        {
            std::lock_guard<std::mutex> lock(_mtx_position);

            // set two endpoints poses
            _pos_w = pos_w;

            // calculate also the Plücker coordinates
            double px(_pos_w(0)), py(_pos_w(1)), pz(_pos_w(2)), qx(_pos_w(3)), qy(_pos_w(4)), qz(_pos_w(5));
            Vec6_t pluecker_coordinates;
            pluecker_coordinates << qy * pz - py * qz,
                qz * px - pz * qx,
                qx * py - px * qy,
                px - qx,
                py - qy,
                pz - qz;
            _pluecker_coordinates = pluecker_coordinates;
        }

        void Line::set_pos_in_world_without_update_pluecker(const Vec6_t &pos_w)
        {
            std::lock_guard<std::mutex> lock(_mtx_position);
            _pos_w = pos_w;
        }

        Vec6_t Line::get_pos_in_world() const
        {
            std::lock_guard<std::mutex> lock(_mtx_position);
            return _pos_w;
        }

        void Line::toPlueckerCoord()
        {
            std::lock_guard<std::mutex> lock(_mtx_position);

            // convert the two endpoints representation -> Plücker coordinates
            // see: https://www.euclideanspace.com/maths/geometry/elements/line/plucker/index.htm
            double px(_pos_w(0)), py(_pos_w(1)), pz(_pos_w(2)), qx(_pos_w(3)), qy(_pos_w(4)), qz(_pos_w(5));

            Vec6_t pluecker_coordinates;
            pluecker_coordinates << qy * pz - py * qz,
                qz * px - pz * qx,
                qx * py - px * qy,
                px - qx,
                py - qy,
                pz - qz;
            _pluecker_coordinates = pluecker_coordinates;
        }

        void Line::set_PlueckerCoord_without_update_endpoints(Vec6_t &pluecker)
        {
            std::lock_guard<std::mutex> lock(_mtx_position);
            _pluecker_coordinates = pluecker;
        }

        Vec6_t Line::get_PlueckerCoord() const
        {
            std::lock_guard<std::mutex> lock(_mtx_position);
            return _pluecker_coordinates;
        }

        keyframe *Line::get_ref_keyframe() const
        {
            std::lock_guard<std::mutex> lock(_mtx_observations);
            return _ref_keyfrm;
        }

        void Line::add_observation(keyframe *keyfrm, unsigned int idx)
        {
            std::lock_guard<std::mutex> lock(_mtx_observations);
            if (_observations.count(keyfrm))
            {
                return;
            }
            _observations[keyfrm] = idx;

            // monocular/RGB-D, not for stereo settings for now
            _num_observations += 1;
        }

        void Line::erase_observation(keyframe *keyfrm)
        {
            bool discard = false;
            {
                std::lock_guard<std::mutex> lock(_mtx_observations);
                if (_observations.count(keyfrm))
                {
                    // int idx = _observations.at(keyfrm);

                    // monocular/RGB-D, not for stereo settings for now
                    _num_observable -= 1;

                    _observations.erase(keyfrm);

                    if (_ref_keyfrm == keyfrm)
                    {
                        _ref_keyfrm = _observations.begin()->first;
                    }

                    if (_num_observations <= 2)
                    {
                        discard = true;
                    }
                }
            }

            if (discard)
            {
                prepare_for_erasing();
            }
        }

        std::map<keyframe *, unsigned int> Line::get_observations() const
        {
            std::lock_guard<std::mutex> lock(_mtx_observations);
            return _observations;
        }

        unsigned int Line::num_observations() const
        {
            std::lock_guard<std::mutex> lock(_mtx_observations);
            return _num_observations;
        }

        bool Line::has_observation() const
        {
            std::lock_guard<std::mutex> lock(_mtx_observations);
            return 0 < _num_observations;
        }

        int Line::get_index_in_keyframe(keyframe *keyfrm) const
        {
            std::lock_guard<std::mutex> lock(_mtx_observations);
            if (_observations.count(keyfrm))
            {
                return _observations.at(keyfrm);
            }
            else
            {
                return -1;
            }
        }

        bool Line::is_observed_in_keyframe(keyframe *keyfrm) const
        {
            std::lock_guard<std::mutex> lock(_mtx_observations);
            return static_cast<bool>(_observations.count(keyfrm));
        }

        cv::Mat Line::get_descriptor() const
        {
            std::lock_guard<std::mutex> lock(_mtx_observations);
            return _descriptor.clone();
        }

        void Line::compute_descriptor()
        {
            // retrieve all observed descriptors
            std::map<keyframe *, unsigned int> observations;
            {
                std::lock_guard<std::mutex> lock1(_mtx_observations);
                if (_will_be_erased)
                {
                    return;
                }
                observations = _observations;
            }

            if (observations.empty())
            {
                return;
            }

            // Append features of corresponding lines
            std::vector<cv::Mat> descriptors;
            descriptors.reserve(observations.size());
            for (const auto &observation : observations)
            {
                auto keyfrm = observation.first;
                const auto idx = observation.second;

                if (!keyfrm->will_be_erased())
                {
                    descriptors.push_back(keyfrm->_lbd_descr.row(idx));
                }
            }

            // Get median of Hamming distance
            // Calculate all the Hamming distances between every pair of the features
            const auto num_descs = descriptors.size();
            std::vector<std::vector<unsigned int>> hamm_dists(num_descs, std::vector<unsigned int>(num_descs));
            for (unsigned int i = 0; i < num_descs; ++i)
            {
                hamm_dists.at(i).at(i) = 0;
                for (unsigned int j = i + 1; j < num_descs; ++j)
                {
                    const auto dist = match::compute_descriptor_distance_32(descriptors.at(i), descriptors.at(j));
                    hamm_dists.at(i).at(j) = dist;
                    hamm_dists.at(j).at(i) = dist;
                }
            }

            // Get the nearest value to median
            unsigned int best_median_dist = match::MAX_HAMMING_DIST;
            unsigned int best_idx = 0;
            for (unsigned idx = 0; idx < num_descs; ++idx)
            {
                std::vector<unsigned int> partial_hamm_dists(hamm_dists.at(idx).begin(), hamm_dists.at(idx).begin() + num_descs);
                std::sort(partial_hamm_dists.begin(), partial_hamm_dists.end());
                const auto median_dist = partial_hamm_dists.at(static_cast<unsigned int>(0.5 * (num_descs - 1)));

                if (median_dist < best_median_dist)
                {
                    best_median_dist = median_dist;
                    best_idx = idx;
                }
            }

            {
                std::lock_guard<std::mutex> lock(_mtx_observations);
                _descriptor = descriptors.at(best_idx).clone();
            }
        }

        void Line::prepare_for_erasing()
        {
            std::map<keyframe *, unsigned int> observations;

            {
                std::lock_guard<std::mutex> lock1(_mtx_observations);
                std::lock_guard<std::mutex> lock2(_mtx_position);
                observations = _observations;
                _observations.clear();
                _will_be_erased = true;
            }

            for (const auto &keyfrm_and_idx : observations)
            {
                keyfrm_and_idx.first->erase_landmark_line_with_index(keyfrm_and_idx.second);
            }

            _map_db->erase_landmark_line(this);
        }

        bool Line::will_be_erased()
        {
            std::lock_guard<std::mutex> lock1(_mtx_observations);
            std::lock_guard<std::mutex> lock2(_mtx_position);
            return _will_be_erased;
        }

        void Line::update_information()
        {
            // FW: for 3D line, we do not calculate average viewing direction, which I think it is not useful
            // the reason would be, e.g. during triangulation, two 3D planes alway intersect to give one 3D line
            std::map<keyframe *, unsigned int> observations;
            keyframe *ref_kf;
            Vec6_t pose;
            Vec3_t ref_kf_camera_center;

            {
                std::lock_guard<std::mutex> lock1(_mtx_observations);
                std::lock_guard<std::mutex> lock2(_mtx_position);

                if (_will_be_erased)
                    return;

                observations = _observations;
                ref_kf = _ref_keyfrm;
                pose = _pos_w;
                ref_kf_camera_center = _ref_keyfrm->get_cam_center();
            }

            if (observations.empty())
                return;

            Vec3_t sp, ep, mp;
            sp << pose(0), pose(1), pose(2);
            ep << pose(3), pose(4), pose(5);
            mp = 0.5 * (sp + ep);

            // we use middle point to calculate
            double distance = (mp - ref_kf_camera_center).norm();
            int level = ref_kf->_keylsd[observations[ref_kf]].octave;     // octave = 0
            float level_scale_factor = ref_kf->_scale_factors_lsd[level]; // this scale factor should be the factor of LSD
            int nlevels = ref_kf->_num_scale_levels_lsd;                  // = 1

            {
                std::lock_guard<std::mutex> lock3(_mtx_position);
                _max_valid_dist = distance * level_scale_factor;
                _min_valid_dist = _max_valid_dist / ref_kf->scale_factors_[nlevels - 1];
            }
        }

        float Line::get_min_valid_distance() const
        {
            std::lock_guard<std::mutex> lock(_mtx_position);
            return 0.8 * _min_valid_dist;
        }

        float Line::get_max_valid_distance() const
        {
            std::lock_guard<std::mutex> lock(_mtx_position);
            return 1.2 * _max_valid_dist;
        }

        unsigned int Line::predict_scale_level(const float &current_dist, const float &log_scale_factor, const unsigned int &num_scale_levels)
        {
            float ratio;
            {
                std::lock_guard<std::mutex> lock(_mtx_position);
                ratio = _max_valid_dist / current_dist;
            }

            const auto pred_scale_level = static_cast<int>(std::ceil(std::log(ratio) / log_scale_factor));
            if (pred_scale_level < 0)
            {
                return 0;
            }
            else if (num_scale_levels <= static_cast<unsigned int>(pred_scale_level))
            {
                return num_scale_levels - 1;
            }
            else
            {
                return static_cast<unsigned int>(pred_scale_level);
            }
        }

        void Line::replace(Line *line)
        {
            if (line->_id == this->_id)
                return;

            unsigned int num_observable, num_observed;
            std::map<keyframe *, unsigned> observations;
            {
                std::lock_guard<std::mutex> lock1(_mtx_observations);
                std::lock_guard<std::mutex> lock2(_mtx_position);
                observations = _observations;
                _observations.clear();
                _will_be_erased = true;
                num_observable = _num_observable;
                num_observed = _num_observed;
                _replaced = line;
            }

            for (const auto &keyfrm_and_idx : observations)
            {
                keyframe *keyfrm = keyfrm_and_idx.first;

                // check all the keyframes which could observe this landmark
                if (!line->is_observed_in_keyframe(keyfrm))
                {
                    keyfrm->replace_landmark_line(line, keyfrm_and_idx.second);
                    line->add_observation(keyfrm, keyfrm_and_idx.second);
                }
                else
                {
                    keyfrm->erase_landmark_line_with_index(keyfrm_and_idx.second);
                }
            }

            line->increase_num_observed(num_observed);
            line->increase_num_observable(num_observable);
            line->compute_descriptor();

            _map_db->erase_landmark_line(this);
        }

        Line *Line::get_replaced() const
        {
            std::lock_guard<std::mutex> lock1(_mtx_observations);
            std::lock_guard<std::mutex> lock2(_mtx_position);
            return _replaced;
        }

        void Line::increase_num_observable(unsigned int num_observable)
        {
            std::lock_guard<std::mutex> lock(_mtx_observations);
            _num_observable += num_observable;
        }

        void Line::increase_num_observed(unsigned int num_observed)
        {
            std::lock_guard<std::mutex> lock(_mtx_observations);
            _num_observed += num_observed;
        }

        float Line::get_observed_ratio() const
        {
            std::lock_guard<std::mutex> lock(_mtx_observations);
            return static_cast<float>(_num_observed) / _num_observable;
        }

        nlohmann::json Line::to_json() const
        {
            return {{"1st_keyfrm", _first_keyfrm_id},
                    {"pos_w", {_pos_w(0), _pos_w(1), _pos_w(2), _pos_w(3), _pos_w(4), _pos_w(5)}},
                    {"ref_keyfrm", _ref_keyfrm->id_},
                    {"n_vis", _num_observable},
                    {"n_fnd", _num_observed}};
        }

    }
}