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

#ifndef PLPSLAM_OPTIMIZE_G2O_SE3_POSE_OPT_EDGE_WRAPPER_H
#define PLPSLAM_OPTIMIZE_G2O_SE3_POSE_OPT_EDGE_WRAPPER_H

#include "PLPSLAM/camera/perspective.h"
#include "PLPSLAM/camera/fisheye.h"
#include "PLPSLAM/camera/equirectangular.h"
#include "PLPSLAM/optimize/g2o/se3/perspective_pose_opt_edge.h"
#include "PLPSLAM/optimize/g2o/se3/equirectangular_pose_opt_edge.h"

#include <g2o/core/robust_kernel_impl.h>

// FW:
#include "PLPSLAM/optimize/g2o/se3/pose_opt_edge_line3d_orthonormal.h" // orthonormal representation

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
            namespace se3
            {

                template <typename T>
                class pose_opt_edge_wrapper
                {
                public:
                    pose_opt_edge_wrapper() = delete;

                    /**
                     * @brief Construct a new pose optimization edge (Unary) wrapper object, for motion-only BA (pose optimizer)
                     *
                     * @param shot frame
                     * @param shot_vtx frame vertex
                     * @param pos_w 3D coordinates (pose) of the point landmark
                     * @param idx id of keypoint
                     * @param obs_x pixel position of the key point
                     * @param obs_y pixel position of the key point
                     * @param obs_x_right < 0, if monocular
                     * @param inv_sigma_sq inverse of covariance matrix
                     * @param sqrt_chi_sq
                     */
                    pose_opt_edge_wrapper(T *shot, shot_vertex *shot_vtx, const Vec3_t &pos_w,
                                          const unsigned int idx, const float obs_x, const float obs_y, const float obs_x_right,
                                          const float inv_sigma_sq, const float sqrt_chi_sq);

                    // FW:
                    // for line (Plücker coordinates, and orthonormal representation)
                    pose_opt_edge_wrapper(T *shot, shot_vertex *shot_vtx,
                                          const Vec6_t &pos_w, const unsigned int idx,
                                          const cv::Point2f sp, const cv::Point2f ep,
                                          const float obs_x_right,
                                          const float inv_sigma_sq, const float sqrt_chi_sq);

                    virtual ~pose_opt_edge_wrapper() = default;

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

                    inline bool depth_is_positive() const;

                    ::g2o::OptimizableGraph::Edge *edge_;

                    camera::base *camera_;
                    T *shot_;
                    const unsigned int idx_;
                    const bool is_monocular_;
                }; // End: class definition

                // Constructor
                template <typename T>
                pose_opt_edge_wrapper<T>::pose_opt_edge_wrapper(T *shot, shot_vertex *shot_vtx, const Vec3_t &pos_w,
                                                                const unsigned int idx, const float obs_x, const float obs_y, const float obs_x_right,
                                                                const float inv_sigma_sq, const float sqrt_chi_sq)
                    : camera_(shot->camera_),
                      shot_(shot),
                      idx_(idx),
                      is_monocular_(obs_x_right < 0)
                {
                    // set constraints
                    switch (camera_->model_type_)
                    {
                    case camera::model_type_t::Perspective:
                    {
                        auto c = static_cast<camera::perspective *>(camera_);
                        if (is_monocular_)
                        { // Monocular
                            // define a reprojection edge (Unary)
                            auto edge = new mono_perspective_pose_opt_edge(); // create an edge

                            const Vec2_t obs{obs_x, obs_y};
                            edge->setMeasurement(obs);                                // y-real: the measurement->2D pixel position in the image
                            edge->setInformation(Mat22_t::Identity() * inv_sigma_sq); // Information matrix: the inverse of the covariance matrix

                            edge->fx_ = c->fx_;
                            edge->fy_ = c->fy_;
                            edge->cx_ = c->cx_;
                            edge->cy_ = c->cy_;

                            edge->pos_w_ = pos_w;

                            edge->setVertex(0, shot_vtx); // set vertex which is connected to this edge, Unary edge->just one vertex, id always=0

                            edge_ = edge;
                        }
                        else
                        { // Stereo
                            auto edge = new stereo_perspective_pose_opt_edge();

                            const Vec3_t obs{obs_x, obs_y, obs_x_right};
                            edge->setMeasurement(obs);
                            edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                            edge->fx_ = c->fx_;
                            edge->fy_ = c->fy_;
                            edge->cx_ = c->cx_;
                            edge->cy_ = c->cy_;
                            edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                            edge->pos_w_ = pos_w;

                            edge->setVertex(0, shot_vtx);

                            edge_ = edge;
                        }
                        break;
                    }
                    case camera::model_type_t::Fisheye:
                    {
                        auto c = static_cast<camera::fisheye *>(camera_);
                        if (is_monocular_)
                        {
                            auto edge = new mono_perspective_pose_opt_edge();

                            const Vec2_t obs{obs_x, obs_y};
                            edge->setMeasurement(obs);
                            edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                            edge->fx_ = c->fx_;
                            edge->fy_ = c->fy_;
                            edge->cx_ = c->cx_;
                            edge->cy_ = c->cy_;

                            edge->pos_w_ = pos_w;

                            edge->setVertex(0, shot_vtx);

                            edge_ = edge;
                        }
                        else
                        {
                            auto edge = new stereo_perspective_pose_opt_edge();

                            const Vec3_t obs{obs_x, obs_y, obs_x_right};
                            edge->setMeasurement(obs);
                            edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                            edge->fx_ = c->fx_;
                            edge->fy_ = c->fy_;
                            edge->cx_ = c->cx_;
                            edge->cy_ = c->cy_;
                            edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                            edge->pos_w_ = pos_w;

                            edge->setVertex(0, shot_vtx);

                            edge_ = edge;
                        }
                        break;
                    }
                    case camera::model_type_t::Equirectangular:
                    {
                        assert(is_monocular_);

                        auto c = static_cast<camera::equirectangular *>(camera_);

                        auto edge = new equirectangular_pose_opt_edge();

                        const Vec2_t obs{obs_x, obs_y};
                        edge->setMeasurement(obs);
                        edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                        edge->cols_ = c->cols_;
                        edge->rows_ = c->rows_;

                        edge->pos_w_ = pos_w;

                        edge->setVertex(0, shot_vtx);

                        edge_ = edge;

                        break;
                    }
                    }

                    // Set loss function
                    auto huber_kernel = new ::g2o::RobustKernelHuber();
                    huber_kernel->setDelta(sqrt_chi_sq);
                    edge_->setRobustKernel(huber_kernel);
                }

                // FW: for line (Plücker coordinates, and orthonormal representation)
                template <typename T>
                pose_opt_edge_wrapper<T>::pose_opt_edge_wrapper(T *shot, shot_vertex *shot_vtx,
                                                                const Vec6_t &pos_w, const unsigned int idx,
                                                                const cv::Point2f sp, const cv::Point2f ep,
                                                                const float obs_x_right,
                                                                const float inv_sigma_sq, const float sqrt_chi_sq)
                    : camera_(shot->camera_),
                      shot_(shot),
                      idx_(idx),
                      is_monocular_(obs_x_right < 0)
                {
                    auto c = static_cast<camera::perspective *>(camera_);

                    // define a reprojection edge (Unary)
                    auto edge = new PLPSLAM::optimize::g2o::se3::pose_opt_edge_line3d();

                    Vec4_t obs{sp.x, sp.y, ep.x, ep.y};
                    edge->setMeasurement(obs);                                // y-real: the measurement->the line parameters (ax+by+c=0)
                    edge->setInformation(Mat22_t::Identity() * inv_sigma_sq); // Information matrix * inv_sigma_sq

                    edge->_fx = c->fx_;
                    edge->_fy = c->fy_;
                    edge->_cx = c->cx_;
                    edge->_cy = c->cy_;

                    edge->_K << c->fy_, 0.0, 0.0,
                        0.0, c->fx_, 0.0,
                        -c->fy_ * c->cx_, -c->fx_ * c->cy_, c->fx_ * c->fy_;

                    edge->_pos_w = pos_w; // Plücker coordinates

                    edge->setVertex(0, shot_vtx); // set vertex which is connected to this edge, Unary edge->just one vertex, id always=0

                    edge_ = edge;

                    // Set loss function
                    auto huber_kernel = new ::g2o::RobustKernelHuber();
                    huber_kernel->setDelta(sqrt_chi_sq);
                    edge_->setRobustKernel(huber_kernel);
                }

                template <typename T>
                bool pose_opt_edge_wrapper<T>::depth_is_positive() const
                {
                    switch (camera_->model_type_)
                    {
                    case camera::model_type_t::Perspective:
                    {
                        if (is_monocular_)
                        {
                            return static_cast<mono_perspective_pose_opt_edge *>(edge_)->mono_perspective_pose_opt_edge::depth_is_positive();
                        }
                        else
                        {
                            return static_cast<stereo_perspective_pose_opt_edge *>(edge_)->stereo_perspective_pose_opt_edge::depth_is_positive();
                        }
                    }
                    case camera::model_type_t::Fisheye:
                    {
                        if (is_monocular_)
                        {
                            return static_cast<mono_perspective_pose_opt_edge *>(edge_)->mono_perspective_pose_opt_edge::depth_is_positive();
                        }
                        else
                        {
                            return static_cast<stereo_perspective_pose_opt_edge *>(edge_)->stereo_perspective_pose_opt_edge::depth_is_positive();
                        }
                    }
                    case camera::model_type_t::Equirectangular:
                    {
                        return true;
                    }
                    }

                    return true;
                }

            } // namespace se3
        }     // namespace g2o
    }         // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_OPTIMIZE_G2O_SE3_POSE_OPT_EDGE_WRAPPER_H
