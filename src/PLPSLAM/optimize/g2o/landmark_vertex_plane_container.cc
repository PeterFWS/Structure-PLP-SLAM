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

#include "PLPSLAM/optimize/g2o/landmark_vertex_plane_container.h"

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            landmark_vertex_plane_container::landmark_vertex_plane_container(const unsigned int offset,
                                                                             const unsigned int num_reserve)
                : _offset(offset)
            {
                _vtx_container.reserve(num_reserve);
            }

            landmark_vertex_plane *landmark_vertex_plane_container::create_vertex(data::Plane *pl, const bool is_constant)
            {
                // create a vertex
                const auto vtx_id = _offset + pl->_id;
                auto vtx = new landmark_vertex_plane();
                vtx->setId(vtx_id);
                auto plane = toPlane3D(pl);
                vtx->setEstimate(std::move(plane));
                vtx->setFixed(is_constant);

                // FW: TODO: debug this
                vtx->setMarginalized(true);

                // register in the database
                _vtx_container[pl->_id] = vtx;

                // update maximum id
                if (_max_vtx_id < vtx_id)
                {
                    _max_vtx_id = vtx_id;
                }

                return vtx;
            }

            Plane3D landmark_vertex_plane_container::toPlane3D(data::Plane *pl)
            {
                Vec3_t normal = pl->get_normal();
                double offset = pl->get_offset();

                ::g2o::Vector4 v;
                v << normal(0), normal(1), normal(2), offset;

                // FW: why?
                if (offset < 0)
                    v = -v;

                return Plane3D(v);
            }

        } // namespace g2o
    }     // namespace optimize
} // namespace PLPSLAM