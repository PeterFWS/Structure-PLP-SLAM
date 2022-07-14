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

#include "PLPSLAM/camera/perspective.h"
#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/module/two_view_triangulator_line.h"

namespace PLPSLAM
{
    namespace module
    {

        two_view_triangulator_line::two_view_triangulator_line(data::keyframe *keyfrm_1, data::keyframe *keyfrm_2,
                                                               const float rays_parallax_deg_thr)
            : keyfrm_1_(keyfrm_1), keyfrm_2_(keyfrm_2),
              rot_1w_(keyfrm_1->get_rotation()), rot_w1_(rot_1w_.transpose()), trans_1w_(keyfrm_1->get_translation()),
              cam_pose_1w_(keyfrm_1->get_cam_pose()), cam_center_1_(keyfrm_1->get_cam_center()), camera_1_(keyfrm_1->camera_),
              rot_2w_(keyfrm_2->get_rotation()), rot_w2_(rot_2w_.transpose()), trans_2w_(keyfrm_2->get_translation()),
              cam_pose_2w_(keyfrm_2->get_cam_pose()), cam_center_2_(keyfrm_2->get_cam_center()), camera_2_(keyfrm_2->camera_),
              ratio_factor_(2.0f * std::max(keyfrm_1->scale_factor_, keyfrm_2->scale_factor_)),
              cos_rays_parallax_thr_(std::cos(rays_parallax_deg_thr * M_PI / 180.0))
        {
            _camera = static_cast<camera::perspective *>(keyfrm_1->camera_);

            _K << _camera->fy_, 0.0, 0.0,
                0.0, _camera->fx_, 0.0,
                -_camera->fy_ * _camera->cx_, -_camera->fx_ * _camera->cy_, _camera->fx_ * _camera->fy_;

            _camera_type = keyfrm_1->camera_->setup_type_;
        }

        bool two_view_triangulator_line::triangulate(const unsigned idx_1, const unsigned int idx_2, Vec6_t &pos_w_line) const
        {
            // get 2D line segments and function (parameters)
            cv::line_descriptor::KeyLine keyline1 = keyfrm_1_->_keylsd[idx_1];
            cv::line_descriptor::KeyLine keyline2 = keyfrm_2_->_keylsd[idx_2];
            Vec3_t keyline1_function = keyfrm_1_->_keyline_functions[idx_1];
            Vec3_t keyline2_function = keyfrm_2_->_keyline_functions[idx_2];

            const float keyline_1_right = keyfrm_1_->_stereo_x_right_cooresponding_to_keylines[idx_1].first;
            const bool is_stereo_1 = (0 <= keyline_1_right);

            const float keyline_2_right = keyfrm_2_->_stereo_x_right_cooresponding_to_keylines[idx_2].first;
            const bool is_stereo_2 = (0 <= keyline_2_right);

            // convert keyline's middle point to bearings and compute the parallax between keyframe_1 and keyframe_2
            // bearing in keyframe_1
            const double x_normalized_1 = (keyline1.pt.x - _camera->cx_) / _camera->fx_;
            const double y_normalized_1 = (keyline1.pt.y - _camera->cy_) / _camera->fy_;
            const auto l2_norm_1 = std::sqrt(x_normalized_1 * x_normalized_1 + y_normalized_1 * y_normalized_1 + 1.0);
            Vec3_t ray_c_1 = Vec3_t{x_normalized_1 / l2_norm_1,
                                    y_normalized_1 / l2_norm_1,
                                    1.0 / l2_norm_1};

            // bearing in keyframe_2
            const double x_normalized_2 = (keyline2.pt.x - _camera->cx_) / _camera->fx_;
            const double y_normalized_2 = (keyline2.pt.y - _camera->cy_) / _camera->fy_;
            const auto l2_norm_2 = std::sqrt(x_normalized_2 * x_normalized_2 + y_normalized_2 * y_normalized_2 + 1.0);
            Vec3_t ray_c_2 = Vec3_t{x_normalized_2 / l2_norm_2,
                                    y_normalized_2 / l2_norm_2,
                                    1.0 / l2_norm_2};

            // rays with the world reference
            const Vec3_t ray_w_1 = rot_w1_ * ray_c_1;
            const Vec3_t ray_w_2 = rot_w2_ * ray_c_2;
            const auto cos_rays_parallax = ray_w_1.dot(ray_w_2);

            // compute the stereo parallax if the keypoint is observed as stereo
            const auto cos_stereo_parallax_1 = is_stereo_1
                                                   ? std::cos(2.0 * atan2(camera_1_->true_baseline_ / 2.0, keyfrm_1_->depths_.at(idx_1)))
                                                   : 2.0;
            const auto cos_stereo_parallax_2 = is_stereo_2
                                                   ? std::cos(2.0 * atan2(camera_2_->true_baseline_ / 2.0, keyfrm_2_->depths_.at(idx_2)))
                                                   : 2.0;
            const auto cos_stereo_parallax = std::min(cos_stereo_parallax_1, cos_stereo_parallax_2);

            // select to use "linear triangulation" or "stereo triangulation/depth triangulation"
            // threshold of minimum angle of the two rays
            const bool triangulate_with_two_cameras =
                // check if the sufficient parallax is provided
                ((!is_stereo_1 && !is_stereo_2) && 0.0 < cos_rays_parallax && cos_rays_parallax < cos_rays_parallax_thr_)
                // check if the parallax between the two cameras is larger than the stereo parallax
                || ((is_stereo_1 || is_stereo_2) && 0.0 < cos_rays_parallax && cos_rays_parallax < cos_stereo_parallax);

            // recover 3D position of starting/ending point
            Vec4_t sp_3D, ep_3D;
            if (triangulate_with_two_cameras)
            { // FW: Triangulation Method 2: find an infinite 3D line via intersection of two 3D planes, while endpoints estimated by endpoints trimming
                // references of endpoints trimming:
                // "Elaborate Monocular Point and Line SLAM with Robust Initialization", ICCV'19
                // "Building a 3-D line-based map using stereo SLAM", IEEE Transactions on Robotics'15
                //! Notice: endpoints trimming is also used in BA for re-estimating the endpoints

                // the projection matrix of two keyframes
                Mat34_t Tcw1 = keyfrm_1_->get_cam_pose().block<3, 4>(0, 0);
                Mat34_t Tcw2 = keyfrm_2_->get_cam_pose().block<3, 4>(0, 0);
                Mat34_t P1 = _camera->eigen_cam_matrix_ * Tcw1;
                Mat34_t P2 = _camera->eigen_cam_matrix_ * Tcw2;

                // construct two planes
                Vec3_t xs_1{keyline1.getStartPoint().x, keyline1.getStartPoint().y, 1.0};
                Vec3_t xe_1{keyline1.getEndPoint().x, keyline1.getEndPoint().y, 1.0};
                Vec3_t line_1 = xs_1.cross(xe_1);
                Vec4_t plane_1 = line_1.transpose() * P1;

                Vec3_t xs_2{keyline2.getStartPoint().x, keyline2.getStartPoint().y, 1.0};
                Vec3_t xe_2{keyline2.getEndPoint().x, keyline2.getEndPoint().y, 1.0};
                Vec3_t line_2 = xs_2.cross(xe_2);
                Vec4_t plane_2 = line_2.transpose() * P2;

                // calculate dual Pluecker matrix via two plane intersection
                Mat44_t L_star = plane_1 * plane_2.transpose() - plane_2 * plane_1.transpose();

                // extract Pluecker coordinates of the 3D line (infinite line representation)
                Mat33_t d_skew = L_star.block<3, 3>(0, 0);
                Vec3_t d;
                d << d_skew(2, 1), d_skew(0, 2), d_skew(1, 0); // the direction vector of the line
                Vec3_t m = L_star.block<3, 1>(0, 3);           // the moment vector of the line

                Vec6_t plucker_coord;
                plucker_coord << m(0), m(1), m(2), d(0), d(1), d(2);

                // endpoints trimming (using keyframe 1)
                Mat66_t transformation_line_cw = Eigen::Matrix<double, 6, 6>::Zero();
                transformation_line_cw.block<3, 3>(0, 0) = rot_1w_;
                transformation_line_cw.block<3, 3>(3, 3) = rot_1w_;
                transformation_line_cw.block<3, 3>(0, 3) = skew(trans_1w_) * rot_1w_;

                Vec3_t reproj_line_function;
                reproj_line_function = _K * (transformation_line_cw * plucker_coord).block<3, 1>(0, 0);

                double l1 = reproj_line_function(0);
                double l2 = reproj_line_function(1);
                double l3 = reproj_line_function(2);

                // calculate closet point on the re-projected line
                auto sp = keyline1.getStartPoint();
                auto ep = keyline1.getEndPoint();
                double x_sp_closet = -(sp.y - (l2 / l1) * sp.x + (l3 / l2)) * ((l1 * l2) / (l1 * l1 + l2 * l2));
                double y_sp_closet = -(l1 / l2) * x_sp_closet - (l3 / l2);

                double x_ep_closet = -(ep.y - (l2 / l1) * ep.x + (l3 / l2)) * ((l1 * l2) / (l1 * l1 + l2 * l2));
                double y_ep_closet = -(l1 / l2) * x_ep_closet - (l3 / l2);

                // calculate another point
                double x_0sp = 0;
                double y_0sp = sp.y - (l2 / l1) * sp.x;

                double x_0ep = 0;
                double y_0ep = ep.y - (l2 / l1) * ep.x;

                // calculate 3D plane (using keyframe 1)
                Vec3_t point2d_sp_closet{x_sp_closet, y_sp_closet, 1.0};
                Vec3_t point2d_0sp{x_0sp, y_0sp, 1.0};
                Vec3_t line_temp_sp = point2d_sp_closet.cross(point2d_0sp);
                Vec4_t plane3d_temp_sp = P1.transpose() * line_temp_sp;

                Vec3_t point2d_ep_closet{x_ep_closet, y_ep_closet, 1.0};
                Vec3_t point2d_0ep{x_0ep, y_0ep, 1.0};
                Vec3_t line_temp_ep = point2d_ep_closet.cross(point2d_0ep);
                Vec4_t plane3d_temp_ep = P1.transpose() * line_temp_ep;

                // calculate intersection of the 3D plane and 3d line
                Mat44_t line3d_pluecker_matrix = Eigen::Matrix<double, 4, 4>::Zero();
                line3d_pluecker_matrix.block<3, 3>(0, 0) = skew(m);
                line3d_pluecker_matrix.block<3, 1>(0, 3) = d;
                line3d_pluecker_matrix.block<1, 3>(3, 0) = -d.transpose();

                Vec4_t intersect_endpoint_sp, intersect_endpoint_ep;
                intersect_endpoint_sp = line3d_pluecker_matrix * plane3d_temp_sp;
                intersect_endpoint_ep = line3d_pluecker_matrix * plane3d_temp_ep;

                sp_3D << intersect_endpoint_sp(0) / intersect_endpoint_sp(3),
                    intersect_endpoint_sp(1) / intersect_endpoint_sp(3),
                    intersect_endpoint_sp(2) / intersect_endpoint_sp(3),
                    1.0;
                ep_3D << intersect_endpoint_ep(0) / intersect_endpoint_ep(3),
                    intersect_endpoint_ep(1) / intersect_endpoint_ep(3),
                    intersect_endpoint_ep(2) / intersect_endpoint_ep(3),
                    1.0;
            }
            else if (is_stereo_1 && cos_stereo_parallax_1 < cos_stereo_parallax_2)
            {
                if (_camera_type == camera::setup_type_t::RGBD)
                {
                    pos_w_line = keyfrm_1_->triangulate_stereo_for_line(idx_1);
                    sp_3D.block<3, 1>(0, 0) = pos_w_line.head(3);
                    sp_3D(3) = 1.0;
                    ep_3D.block<3, 1>(0, 0) = pos_w_line.tail(3);
                    ep_3D(3) = 1.0;
                }
                if (_camera_type == camera::setup_type_t::Stereo)
                {
                    pos_w_line = keyfrm_1_->triangulate_stereo_for_line(idx_1);
                    sp_3D.block<3, 1>(0, 0) = pos_w_line.head(3);
                    sp_3D(3) = 1.0;
                    ep_3D.block<3, 1>(0, 0) = pos_w_line.tail(3);
                    ep_3D(3) = 1.0;
                }
            }
            else if (is_stereo_2 && cos_stereo_parallax_2 < cos_stereo_parallax_1)
            {
                if (_camera_type == camera::setup_type_t::RGBD)
                {
                    pos_w_line = keyfrm_2_->triangulate_stereo_for_line(idx_2);
                    sp_3D.block<3, 1>(0, 0) = pos_w_line.head(3);
                    sp_3D(3) = 1.0;
                    ep_3D.block<3, 1>(0, 0) = pos_w_line.tail(3);
                    ep_3D(3) = 1.0;
                }
                if (_camera_type == camera::setup_type_t::Stereo)
                {
                    pos_w_line = keyfrm_2_->triangulate_stereo_for_line(idx_2);
                    sp_3D.block<3, 1>(0, 0) = pos_w_line.head(3);
                    sp_3D(3) = 1.0;
                    ep_3D.block<3, 1>(0, 0) = pos_w_line.tail(3);
                    ep_3D(3) = 1.0;
                }
            }
            else
            {
                return false;
            }

            // check if the ending points are too close to the camera_center
            if ((sp_3D.block<3, 1>(0, 0) - cam_center_1_).norm() / keyfrm_2_->compute_median_depth(true) < 0.3 ||
                (ep_3D.block<3, 1>(0, 0) - cam_center_2_).norm() / keyfrm_2_->compute_median_depth(true) < 0.3)
            {
                return false;
            }

            // check if the 3D line is too lang
            if ((ep_3D.block<3, 1>(0, 0) - sp_3D.block<3, 1>(0, 0)).norm() / keyfrm_2_->compute_median_depth(true) > 0.9)
            {
                return false;
            }

            // check if positive depth in both keyframes
            if (!check_depth_is_positive(sp_3D.block<3, 1>(0, 0), rot_1w_, trans_1w_) ||
                !check_depth_is_positive(sp_3D.block<3, 1>(0, 0), rot_2w_, trans_2w_) ||
                !check_depth_is_positive(ep_3D.block<3, 1>(0, 0), rot_1w_, trans_1w_) ||
                !check_depth_is_positive(ep_3D.block<3, 1>(0, 0), rot_2w_, trans_2w_))
            {
                return false;
            }

            // check reprojection error (midpoint, starting point, and ending point) in both keyframes
            Vec3_t midpoint = 0.5 * (sp_3D.block<3, 1>(0, 0) + ep_3D.block<3, 1>(0, 0));
            if (!check_reprojection_error(midpoint, rot_1w_, trans_1w_, keyline1, keyfrm_1_->level_sigma_sq_.at(keyline1.octave)) ||
                !check_reprojection_error(midpoint, rot_2w_, trans_2w_, keyline2, keyfrm_2_->level_sigma_sq_.at(keyline2.octave)) ||
                !check_reprojection_error(sp_3D.block<3, 1>(0, 0), rot_1w_, trans_1w_, keyline1_function, keyline_1_right,
                                          keyfrm_1_->level_sigma_sq_.at(keyline1.octave), is_stereo_1) ||
                !check_reprojection_error(ep_3D.block<3, 1>(0, 0), rot_1w_, trans_1w_, keyline1_function, keyline_1_right,
                                          keyfrm_1_->level_sigma_sq_.at(keyline1.octave), is_stereo_1) ||
                !check_reprojection_error(sp_3D.block<3, 1>(0, 0), rot_2w_, trans_2w_, keyline2_function, keyline_2_right,
                                          keyfrm_2_->level_sigma_sq_.at(keyline2.octave), is_stereo_2) ||
                !check_reprojection_error(ep_3D.block<3, 1>(0, 0), rot_2w_, trans_2w_, keyline2_function, keyline_2_right,
                                          keyfrm_2_->level_sigma_sq_.at(keyline2.octave), is_stereo_2))
            {
                return false;
            }

            // reject the line if the real scale factor and the predicted one are much different
            if (!check_scale_factors(sp_3D.block<3, 1>(0, 0),
                                     keyfrm_1_->scale_factors_.at(keyline1.octave),
                                     keyfrm_2_->scale_factors_.at(keyline2.octave)) ||
                !check_scale_factors(ep_3D.block<3, 1>(0, 0),
                                     keyfrm_1_->scale_factors_.at(keyline1.octave),
                                     keyfrm_2_->scale_factors_.at(keyline2.octave)))
            {
                return false;
            }

            pos_w_line << sp_3D(0), sp_3D(1), sp_3D(2), ep_3D(0), ep_3D(1), ep_3D(2);
            return true;
        }

        // used to check the ending point of the line
        bool two_view_triangulator_line::check_reprojection_error(const Vec3_t &pos_w, const Mat33_t &rot_cw, const Vec3_t &trans_cw,
                                                                  const Vec3_t &keyline_function, const float x_right,
                                                                  const float sigma_sq, const bool is_stereo) const
        {
            // chi-squared values for p=5%
            // (n=2)
            constexpr float chi_sq_2D = 5.99146;
            // (n=3)
            // constexpr float chi_sq_3D = 7.81473;

            Vec2_t reproj_in_cur;
            float x_right_in_cur;
            _camera->reproject_to_image(rot_cw, trans_cw, pos_w, reproj_in_cur, x_right_in_cur);

            const float reproj_err = (keyline_function(0) * reproj_in_cur(0) + keyline_function(1) * reproj_in_cur(1) + keyline_function(2)) /
                                     sqrt((keyline_function(0) * keyline_function(0) + keyline_function(1) * keyline_function(1)));

            if (chi_sq_2D * sigma_sq < abs(reproj_err))
            {
                return false;
            }

            return true;
        }

        // used to check the middle point of the line
        bool two_view_triangulator_line::check_reprojection_error(const Vec3_t &pos_w_middlepoint, const Mat33_t &rot_cw, const Vec3_t &trans_cw,
                                                                  const cv::line_descriptor::KeyLine &keyline, const float sigma_sq) const
        {
            // chi-squared values for p=5%
            // (n=2)
            constexpr float chi_sq_2D = 5.99146;
            // (n=3)
            // constexpr float chi_sq_3D = 7.81473;

            Vec2_t reproj_in_cur;
            float x_right_in_cur;
            _camera->reproject_to_image(rot_cw, trans_cw, pos_w_middlepoint, reproj_in_cur, x_right_in_cur);

            const Vec2_t reproj_err_curr = reproj_in_cur - keyline.pt;
            if (chi_sq_2D * sigma_sq < reproj_err_curr.squaredNorm())
            {
                return false;
            }

            return true;
        }

    } // namespace module
} // namespace PLPSLAM
