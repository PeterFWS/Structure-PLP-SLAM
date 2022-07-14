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

#ifndef PLPSLAM_DATA_BOW_DATABASE_H
#define PLPSLAM_DATA_BOW_DATABASE_H

#include "PLPSLAM/data/bow_vocabulary.h"

#include <mutex>
#include <list>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace PLPSLAM
{
       namespace data
       {

              class frame;
              class keyframe;

              class bow_database
              {
              public:
                     /**
                      * Constructor
                      * @param bow_vocab
                      */
                     explicit bow_database(bow_vocabulary *bow_vocab);

                     /**
                      * Destructor
                      */
                     ~bow_database();

                     /**
                      * Add a keyframe to the database
                      * @param keyfrm
                      */
                     void add_keyframe(keyframe *keyfrm);

                     /**
                      * Erase the keyframe from the database
                      * @param keyfrm
                      */
                     void erase_keyframe(keyframe *keyfrm);

                     /**
                      * Clear the database
                      */
                     void clear();

                     /**
                      * Acquire loop-closing candidates over the specified score
                      * @param qry_keyfrm
                      * @param min_score
                      * @return
                      */
                     std::vector<keyframe *> acquire_loop_candidates(keyframe *qry_keyfrm, const float min_score);

                     /**
                      * Acquire relocalization candidates
                      * @param qry_frm
                      * @return
                      */
                     std::vector<keyframe *> acquire_relocalization_candidates(frame *qry_frm);

              protected:
                     /**
                      * Initialize temporary variables
                      */
                     void initialize();

                     /**
                      * Compute the number of shared words and set candidates (init_candidates_ and num_common_words_)
                      * @tparam T
                      * @param qry_shot
                      * @param keyfrms_to_reject
                      * @return whether candidates are found or not
                      */
                     template <typename T>
                     bool set_candidates_sharing_words(const T *const qry_shot, const std::set<keyframe *> &keyfrms_to_reject = {});

                     /**
                      * Compute scores (scores_) between the query and the each of keyframes contained in the database
                      * @tparam T
                      * @param qry_shot
                      * @param min_num_common_words_thr
                      * @return whether candidates are found or not
                      */
                     template <typename T>
                     bool compute_scores(const T *const qry_shot, const unsigned int min_num_common_words_thr);

                     /**
                      * Align scores and keyframes only which have greater scores than the minimum one
                      * @param min_num_common_words_thr
                      * @param min_score
                      * @return whether candidates are found or not
                      */
                     bool align_scores_and_keyframes(const unsigned int min_num_common_words_thr, const float min_score);

                     /**
                      * Compute and align total scores and keyframes
                      * @param min_num_common_words_thr
                      * @param min_score
                      * @return
                      */
                     float align_total_scores_and_keyframes(const unsigned int min_num_common_words_thr, const float min_score);

                     //-----------------------------------------
                     // BoW feature vectors

                     //! mutex to access BoW database
                     mutable std::mutex mtx_;
                     //! BoW database
                     std::unordered_map<unsigned int, std::list<keyframe *>> keyfrms_in_node_;

                     //-----------------------------------------
                     // BoW vocabulary

                     //! BoW vocabulary
                     bow_vocabulary *bow_vocab_;

                     //-----------------------------------------
                     // temporary variables

                     //! mutex to access temporary variables
                     mutable std::mutex tmp_mtx_;

                     //! initial candidates for loop or relocalization
                     std::unordered_set<keyframe *> init_candidates_;

                     // key: keyframe sharing word with query, value: number of shared words
                     //! number of shared words between the query and the each of keyframes contained in the database
                     std::unordered_map<keyframe *, unsigned int> num_common_words_;

                     // key: keyframe sharing word with query, value: score
                     //! similarity scores between the query and the each of keyframes contained in the database
                     std::unordered_map<keyframe *, float> scores_;

                     // Save more than min_score vector (score, keyframe)
                     //! pairs of score and keyframe which has the larger score than the minimum one
                     std::vector<std::pair<float, keyframe *>> score_keyfrm_pairs_;

                     // Save more than min_score vector (score, keyframe)
                     //! pairs of total score and keyframe which has the larger score than the minimum one
                     std::vector<std::pair<float, keyframe *>> total_score_keyfrm_pairs_;
              };

       } // namespace data
} // namespace PLPSLAM

#endif // PLPSLAM_DATA_BOW_DATABASE_H
