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

#ifndef PLPSLAM_OPTIMIZER_G2O_SIM3_GRAPH_OPT_EDGE_H
#define PLPSLAM_OPTIMIZER_G2O_SIM3_GRAPH_OPT_EDGE_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/optimize/g2o/sim3/shot_vertex.h"

#include <g2o/core/base_binary_edge.h>

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            namespace sim3
            {

                class graph_opt_edge final : public ::g2o::BaseBinaryEdge<7, ::g2o::Sim3, shot_vertex, shot_vertex>
                {
                public:
                    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                    graph_opt_edge();

                    bool read(std::istream &is) override;

                    bool write(std::ostream &os) const override;

                    void computeError() override
                    {
                        const auto v1 = static_cast<const shot_vertex *>(_vertices.at(0));
                        const auto v2 = static_cast<const shot_vertex *>(_vertices.at(1));

                        const ::g2o::Sim3 C(_measurement);
                        const ::g2o::Sim3 error_ = C * v1->estimate() * v2->estimate().inverse();
                        _error = error_.log();
                    }

                    double initialEstimatePossible(const ::g2o::OptimizableGraph::VertexSet &, ::g2o::OptimizableGraph::Vertex *) override
                    {
                        return 1.0;
                    }

                    void initialEstimate(const ::g2o::OptimizableGraph::VertexSet &from, ::g2o::OptimizableGraph::Vertex *) override
                    {
                        auto v1 = static_cast<shot_vertex *>(_vertices[0]);
                        auto v2 = static_cast<shot_vertex *>(_vertices[1]);
                        if (0 < from.count(v1))
                        {
                            v2->setEstimate(measurement() * v1->estimate());
                        }
                        else
                        {
                            v1->setEstimate(measurement().inverse() * v2->estimate());
                        }
                    }
                };

            } // namespace sim3
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_OPTIMIZER_G2O_SIM3_GRAPH_OPT_EDGE_H
