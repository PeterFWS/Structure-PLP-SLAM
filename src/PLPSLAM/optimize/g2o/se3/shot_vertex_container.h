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

#ifndef PLPSLAM_OPTIMIZE_G2O_SE3_SHOT_VERTEX_CONTAINER_H
#define PLPSLAM_OPTIMIZE_G2O_SE3_SHOT_VERTEX_CONTAINER_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/data/frame.h"
#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/optimize/g2o/se3/shot_vertex.h"

#include <unordered_map>

namespace PLPSLAM
{

    namespace data
    {
        class frame;
        class keyframe;
    } // namespace data

    namespace optimize
    {
        namespace g2o
        {
            namespace se3
            {

                class shot_vertex_container
                {
                public:
                    /**
                     * Constructor
                     * @param offset
                     * @param num_reserve
                     */
                    explicit shot_vertex_container(const unsigned int offset = 0, const unsigned int num_reserve = 50);

                    /**
                     * Destructor
                     */
                    virtual ~shot_vertex_container() = default;

                    /**
                     * Create and return the g2o vertex created from the specified frame
                     * @param frm
                     * @param is_constant
                     * @return
                     */
                    shot_vertex *create_vertex(data::frame *frm, const bool is_constant)
                    {
                        return create_vertex(frm->id_, frm->cam_pose_cw_, is_constant);
                    }

                    /**
                     * Create and return the g2o vertex created from the specified keyframe
                     * @param keyfrm
                     * @param is_constant
                     * @return
                     */
                    shot_vertex *create_vertex(data::keyframe *keyfrm, const bool is_constant)
                    {
                        return create_vertex(keyfrm->id_, keyfrm->get_cam_pose(), is_constant);
                    }

                    /**
                     * Create and return the g2o vertex created from shot ID and camera pose
                     * @param id
                     * @param cam_pose_cw
                     * @param is_constant
                     * @return
                     */
                    shot_vertex *create_vertex(const unsigned int id, const Mat44_t &cam_pose_cw, const bool is_constant);

                    /**
                     * Get vertex corresponding with the specified frame
                     * @param frm
                     * @return
                     */
                    inline shot_vertex *get_vertex(data::frame *frm) const
                    {
                        return get_vertex(frm->id_);
                    }

                    /**
                     * Get vertex corresponding with the specified keyframe
                     * @param keyfrm
                     * @return
                     */
                    inline shot_vertex *get_vertex(data::keyframe *keyfrm) const
                    {
                        return get_vertex(keyfrm->id_);
                    }

                    /**
                     * Get vertex corresponding with the specified shot (frame/keyframe) ID
                     * @param id
                     * @return
                     */
                    inline shot_vertex *get_vertex(const unsigned int id) const
                    {
                        return vtx_container_.at(id);
                    }

                    /**
                     * Convert frame ID to vertex ID
                     * @param frm
                     * @return
                     */
                    inline unsigned int get_vertex_id(data::frame *frm) const
                    {
                        return get_vertex_id(frm->id_);
                    }

                    /**
                     * Convert keyframe ID to vertex ID
                     * @param keyfrm
                     * @return
                     */
                    inline unsigned int get_vertex_id(data::keyframe *keyfrm) const
                    {
                        return get_vertex_id(keyfrm->id_);
                    }

                    /**
                     * Convert shot (frame/keyframe) ID to vertex ID
                     * @param id
                     * @return
                     */
                    inline unsigned int get_vertex_id(unsigned int id) const
                    {
                        return offset_ + id;
                    }

                    /**
                     * Convert vertex ID to shot (frame/keyframe) ID
                     * @param vtx
                     * @return
                     */
                    inline unsigned int get_id(shot_vertex *vtx)
                    {
                        return vtx->id() - offset_;
                    }

                    /**
                     * Convert vertex ID to shot (frame/keyframe) ID
                     * @param vtx_id
                     * @return
                     */
                    inline unsigned int get_id(unsigned int vtx_id) const
                    {
                        return vtx_id - offset_;
                    }

                    /**
                     * Get maximum vertex ID
                     * @return
                     */
                    inline unsigned int get_max_vertex_id() const
                    {
                        return max_vtx_id_;
                    }

                    /**
                     * Contains the specified keyframe or not
                     */
                    inline bool contain(data::keyframe *keyfrm) const
                    {
                        return 0 != vtx_container_.count(keyfrm->id_);
                    }

                    typedef std::unordered_map<unsigned int, shot_vertex *>::iterator iterator;
                    typedef std::unordered_map<unsigned int, shot_vertex *>::const_iterator const_iterator;

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
                    //! vertex ID = offset + shot (frame/keyframe) ID
                    const unsigned int offset_ = 0;

                    //! key: shot (frame/keyframe) ID, value: vertex
                    std::unordered_map<unsigned int, shot_vertex *> vtx_container_;

                    //! max vertex ID
                    unsigned int max_vtx_id_ = 0;
                };

            } // namespace se3
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_OPTIMIZE_G2O_SE3_SHOT_VERTEX_CONTAINER_H
