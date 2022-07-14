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

#ifndef PLPSLAM_MODULE_TWO_VIEW_TRIANGULATOR_LINE_H
#define PLPSLAM_MODULE_TWO_VIEW_TRIANGULATOR_LINE_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/camera/base.h"

// FW: for LSD ...
#include <opencv2/features2d.hpp>
#include "PLPSLAM/feature/line_descriptor/line_descriptor_custom.hpp"
#include "PLPSLAM/data/landmark_line.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/SVD>
#include <opencv2/core.hpp>

namespace PLPSLAM
{

    namespace camera
    {
        class base;
    } // namespace camera

    namespace data
    {
        class keyframe;
    } // namespace data

    namespace module
    {
        // FW: similar to two_view_triangulator.h
        class two_view_triangulator_line
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            explicit two_view_triangulator_line(data::keyframe *keyfrm_1, data::keyframe *keyfrm_2,
                                                const float rays_parallax_deg_thr = 1.0);

            ~two_view_triangulator_line() = default;

            // Triangulate a 3D line between the keyline idx_1 of keyfrm_1 and the keyline idx_2 of keyfrm_2
            bool triangulate(const unsigned idx_1, const unsigned int idx_2, Vec6_t &pos_w) const;

        private:
            // Check depth is positive or not
            bool check_depth_is_positive(const Vec3_t &pos_w, const Mat33_t &rot_cw, const Vec3_t &trans_cw) const;

            // Check reprojection error is within the acceptable threshold
            bool check_reprojection_error(const Vec3_t &pos_w, const Mat33_t &rot_cw, const Vec3_t &trans_cw,
                                          const Vec3_t &keyline_function, const float x_right,
                                          const float sigma_sq, const bool is_stereo) const;

            bool check_reprojection_error(const Vec3_t &pos_w_middlepoint, const Mat33_t &rot_cw, const Vec3_t &trans_cw,
                                          const cv::line_descriptor::KeyLine &keyline, const float sigma_sq) const;

            // Check estimated and actual scale factors are within the acceptable threshold
            bool check_scale_factors(const Vec3_t &pos_w, const float scale_factor_1, const float scale_factor_2) const;

            //! pointer to keyframe 1
            data::keyframe *const keyfrm_1_;
            //! pointer to keyframe 2
            data::keyframe *const keyfrm_2_;

            // camera poses of keyframe 1
            const Mat33_t rot_1w_;
            const Mat33_t rot_w1_;
            const Vec3_t trans_1w_;
            const Mat44_t cam_pose_1w_;
            const Vec3_t cam_center_1_;

            // camera model of keyframe 1
            camera::base *const camera_1_;

            // camera poses fo keyframe 2
            const Mat33_t rot_2w_;
            const Mat33_t rot_w2_;
            const Vec3_t trans_2w_;
            const Mat44_t cam_pose_2w_;
            const Vec3_t cam_center_2_;

            // camera model of keyframe 2
            camera::base *const camera_2_;

            const float ratio_factor_;

            const float cos_rays_parallax_thr_;

            // FW:
            camera::perspective *_camera;      // camera_1_ of camera::perspective
            Mat33_t _K;                        // used to project a 3D line to the image
            camera::setup_type_t _camera_type; // used for triangulating line from stereo image pair

            // FW:
            inline Mat33_t skew(const Vec3_t &t) const
            {
                Mat33_t S;
                S << 0, -t.z(), t.y(), t.z(), 0, -t.x(), -t.y(), t.x(), 0;
                return S;
            }
        };

        inline bool two_view_triangulator_line::check_depth_is_positive(const Vec3_t &pos_w, const Mat33_t &rot_cw, const Vec3_t &trans_cw) const
        {
            const auto pos_z = rot_cw.block<1, 3>(2, 0).dot(pos_w) + trans_cw(2); // R_cw.row(2) * pos_w + trans_cw(2)
            return 0 < pos_z;
        }

        inline bool two_view_triangulator_line::check_scale_factors(const Vec3_t &pos_w, const float scale_factor_1, const float scale_factor_2) const
        {
            const Vec3_t cam_1_to_lm_vec = pos_w - cam_center_1_;
            const auto cam_1_to_lm_dist = cam_1_to_lm_vec.norm();

            const Vec3_t cam_2_to_lm_vec = pos_w - cam_center_2_;
            const auto cam_2_to_lm_dist = cam_2_to_lm_vec.norm();

            if (cam_1_to_lm_dist == 0 || cam_2_to_lm_dist == 0)
            {
                return false;
            }

            const auto ratio_dists = cam_2_to_lm_dist / cam_1_to_lm_dist;
            const auto ratio_octave = scale_factor_1 / scale_factor_2;

            return ratio_octave / ratio_dists < ratio_factor_ && ratio_dists / ratio_octave < ratio_factor_;
        }

    } // namespace module
} // namespace PLPSLAM

#endif // PLPSLAM_MODULE_TWO_VIEW_TRIANGULATOR_H
