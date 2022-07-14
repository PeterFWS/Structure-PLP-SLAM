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

#ifndef PLPSLAM_OPTIMIZE_G2O_LANDMARK_VERTEX_PLANE_CONTAINER_H
#define PLPSLAM_OPTIMIZE_G2O_LANDMARK_VERTEX_PLANE_CONTAINER_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/data/landmark_plane.h"
#include "PLPSLAM/optimize/g2o/Plane3D.h"
#include "PLPSLAM/optimize/g2o/landmark_vertex_plane.h"
#include "PLPSLAM/util/converter.h"

namespace PLPSLAM
{
    namespace data
    {
        class Plane;
    }

    namespace optimize
    {
        namespace g2o
        {
            class landmark_vertex_plane_container
            {
            public:
                // Constructor
                explicit landmark_vertex_plane_container(const unsigned int offset, const unsigned int num_reserve = 50);

                // Destructor
                virtual ~landmark_vertex_plane_container() = default;

                // Create and return the g2o vertex created from the specific plane landmark
                landmark_vertex_plane *create_vertex(data::Plane *pl, const bool is_constant);

                // Converter
                Plane3D toPlane3D(data::Plane *pl);

                // Get vertex of the specific plane landmark
                inline landmark_vertex_plane *get_vertex(data::Plane *pl) const
                {
                    return get_vertex(pl->_id);
                }

                inline landmark_vertex_plane *get_vertex(const unsigned int id) const
                {
                    return _vtx_container.at(id);
                }

                // Convert plane landmark to vertex ID
                inline unsigned int get_vertex_id(data::Plane *pl) const
                {
                    return get_vertex_id(pl->_id);
                }

                inline unsigned int get_vertex_id(const unsigned int id) const
                {
                    return _offset + id;
                }

                // Convert vertex to plane landmark ID
                inline unsigned int get_id(landmark_vertex_plane *vtx) const
                {
                    return vtx->id() - _offset;
                }

                inline unsigned int get_id(const unsigned int vtx_id) const
                {
                    return vtx_id - _offset;
                }

                // Contains the specified plane landmark or not
                inline bool contain(data::Plane *pl) const
                {
                    return 0 != _vtx_container.count(pl->_id);
                }

                // Get maximum vertex ID
                unsigned int get_max_vertex_id() const
                {
                    return _max_vtx_id;
                }

                typedef std::unordered_map<unsigned int, landmark_vertex_plane *>::iterator iterator;
                typedef std::unordered_map<unsigned int, landmark_vertex_plane *>::const_iterator const_iterator;

                iterator begin()
                {
                    return _vtx_container.begin();
                }

                const_iterator begin() const
                {
                    return _vtx_container.begin();
                }

                iterator end()
                {
                    return _vtx_container.end();
                }

                const_iterator end() const
                {
                    return _vtx_container.end();
                }

            private:
                // vertex id = offset + landmark_plane id
                const unsigned int _offset = 0;

                // key: plane id; value: plane vertex
                std::unordered_map<unsigned int, landmark_vertex_plane *> _vtx_container;

                // maximum vertex id
                unsigned int _max_vtx_id = 0;
            };

        } // namespace g2o

    } // namespace optimize

} // namespace PLPSLAM

#endif