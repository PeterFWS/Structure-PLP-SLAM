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

#ifndef PLPSLAM_OPTIMIZE_G2O_LANDMARK_VERTEX_CONTAINER_H
#define PLPSLAM_OPTIMIZE_G2O_LANDMARK_VERTEX_CONTAINER_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex.h"

#include <unordered_map>

namespace PLPSLAM
{

    namespace data
    {
        class landmark;
    } // namespace data

    namespace optimize
    {
        namespace g2o
        {

            class landmark_vertex_container
            {
            public:
                /**
                 * Constructor
                 * @param offset
                 * @param num_reserve
                 */
                explicit landmark_vertex_container(const unsigned int offset, const unsigned int num_reserve = 200);

                /**
                 * Destructor
                 */
                virtual ~landmark_vertex_container() = default;

                /**
                 * Create and return the g2o vertex created from the specified landmark
                 * @param lm
                 * @param is_constant
                 * @return
                 */
                landmark_vertex *create_vertex(data::landmark *lm, const bool is_constant)
                {
                    return create_vertex(lm->id_, lm->get_pos_in_world(), is_constant);
                }

                /**
                 * Create and return the g2o vertex created from the specified landmark
                 * @param id
                 * @param pos_w
                 * @param is_constant
                 * @return
                 */
                landmark_vertex *create_vertex(const unsigned int id, const Vec3_t &pos_w, const bool is_constant);

                /**
                 * Get vertex corresponding with the specified landmark
                 * @param lm
                 * @return
                 */
                inline landmark_vertex *get_vertex(data::landmark *lm) const
                {
                    return get_vertex(lm->id_);
                }

                /**
                 * Get vertex corresponding with the specified landmark ID
                 * @param id
                 * @return
                 */
                inline landmark_vertex *get_vertex(const unsigned int id) const
                {
                    return vtx_container_.at(id);
                }

                /**
                 * Convert landmark to vertex ID
                 * @param lm
                 * @return
                 */
                inline unsigned int get_vertex_id(data::landmark *lm) const
                {
                    return get_vertex_id(lm->id_);
                }

                /**
                 * Convert landmark ID to vertex ID
                 * @param id
                 * @return
                 */
                inline unsigned int get_vertex_id(const unsigned int id) const
                {
                    return offset_ + id;
                }

                /**
                 * Convert vertex to landmark ID
                 * @param vtx
                 * @return
                 */
                inline unsigned int get_id(landmark_vertex *vtx) const
                {
                    return vtx->id() - offset_;
                }

                /**
                 * Convert vertex ID to landmark ID
                 * @param vtx_id
                 * @return
                 */
                inline unsigned int get_id(const unsigned int vtx_id) const
                {
                    return vtx_id - offset_;
                }

                /**
                 * Contains the specified landmark or not
                 */
                inline bool contain(data::landmark *lm) const
                {
                    return 0 != vtx_container_.count(lm->id_);
                }

                /**
                 * Get maximum vertex ID
                 * @return
                 */
                unsigned int get_max_vertex_id() const
                {
                    return max_vtx_id_;
                }

                typedef std::unordered_map<unsigned int, landmark_vertex *>::iterator iterator;
                typedef std::unordered_map<unsigned int, landmark_vertex *>::const_iterator const_iterator;

                iterator begin()
                {
                    return vtx_container_.begin();
                }

                const_iterator begin() const
                {
                    return vtx_container_.begin();
                }

                iterator end()
                {
                    return vtx_container_.end();
                }

                const_iterator end() const
                {
                    return vtx_container_.end();
                }

            private:
                //! vertex ID = offset + landmark ID
                const unsigned int offset_ = 0;

                //! key: landmark ID, value: vertex
                std::unordered_map<unsigned int, landmark_vertex *> vtx_container_;

                //! max vertex ID
                unsigned int max_vtx_id_ = 0;
            };

        } // namespace g2o
    }     // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_OPTIMIZE_G2O_LANDMARK_VERTEX_CONTAINER_H
