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

#ifndef PLPSLAM_OPTIMIZER_G2O_EXTENDED_PLANE_POINT_EDGE_WRAPPER_H
#define PLPSLAM_OPTIMIZER_G2O_EXTENDED_PLANE_POINT_EDGE_WRAPPER_H

#include "PLPSLAM/optimize/g2o/extended/plane_point_distance_edge.h"
#include <g2o/core/robust_kernel_impl.h>

namespace PLPSLAM
{
    namespace data
    {
        class landmark;
        class Plane;
    } // namespace data

    namespace optimize
    {
        namespace g2o
        {
            namespace extended
            {
                class point2plane_edge_wrapper
                {
                public:
                    point2plane_edge_wrapper() = delete;

                    point2plane_edge_wrapper(data::landmark *lm, landmark_vertex *lm_vtx,
                                             const Vec4_t pl_function,
                                             const bool use_huber_loss = true);

                    virtual ~point2plane_edge_wrapper() = default;

                    inline bool is_inlier() const
                    {
                        return edge_->level() == 0;
                    }

                    inline bool is_outlier() const
                    {
                        return edge_->level() != 0;
                    }

                    inline void set_as_inlier() const
                    {
                        edge_->setLevel(0);
                    }

                    inline void set_as_outlier() const
                    {
                        edge_->setLevel(1);
                    }

                    // members
                    ::g2o::OptimizableGraph::Edge *edge_;
                    data::landmark *_lm;
                };

                point2plane_edge_wrapper::point2plane_edge_wrapper(data::landmark *lm, landmark_vertex *lm_vtx,
                                                                   const Vec4_t pl_function,
                                                                   const bool use_huber_loss)
                    : _lm(lm)
                {
                    auto edge = new point_plane_distance_edge();
                    Vec4_t obs = pl_function;
                    edge->setMeasurement(obs);
                    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity()); // information matrix

                    edge->setVertex(0, lm_vtx);

                    edge_ = edge;

                    // loss function
                    if (use_huber_loss)
                    {
                        auto huber_kernel = new ::g2o::RobustKernelHuber();
                        huber_kernel->setDelta(1.0);
                        edge_->setRobustKernel(huber_kernel);
                    }
                }

            } // namespace extended
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM

#endif