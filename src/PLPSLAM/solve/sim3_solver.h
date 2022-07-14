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

#ifndef PLPSLAM_SOLVE_SIM3_SOLVER_H
#define PLPSLAM_SOLVE_SIM3_SOLVER_H

#include "PLPSLAM/data/keyframe.h"

#include <vector>

#include <opencv2/core.hpp>

namespace PLPSLAM
{
    namespace solve
    {

        class sim3_solver
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            //! Constructor
            sim3_solver(data::keyframe *keyfrm_1, data::keyframe *keyfrm_2,
                        const std::vector<data::landmark *> &matched_lms_in_keyfrm_2,
                        const bool fix_scale = true, const unsigned int min_num_inliers = 20);

            //! Destructor
            virtual ~sim3_solver() = default;

            //! Find the most reliable Sim3 matrix via RANSAC
            void find_via_ransac(const unsigned int max_num_iter);

            //! Check if the solution is valid or not
            bool solution_is_valid() const
            {
                return solution_is_valid_;
            }

            //! Get the most reliable rotation from keyframe 2 to keyframe 1
            Mat33_t get_best_rotation_12()
            {
                return best_rot_12_;
            }

            //! Get the most reliable translation from keyframe 2 to keyframe 1
            Vec3_t get_best_translation_12()
            {
                return best_trans_12_;
            }

            //! Get the most reliable scale from keyframe 2 to keyframe 1
            float get_best_scale_12()
            {
                return best_scale_12_;
            }

        protected:
            //! compute Sim3 from three common points
            //! Estimate the similarity transformation matrix that transforms the coordinate system of points1 (3 points) to points2 (3 points)
            //! (In the input matrix, each column is [x_i, y_i, z_i] .T, and a total of 3 columns are arranged in the row direction)
            void compute_Sim3(const Mat33_t &pts_1, const Mat33_t &pts_2,
                              Mat33_t &rot_12, Vec3_t &trans_12, float &scale_12,
                              Mat33_t &rot_21, Vec3_t &trans_21, float &scale_21);

            //! count up inliers
            unsigned int count_inliers(const Mat33_t &rot_12, const Vec3_t &trans_12, const float scale_12,
                                       const Mat33_t &rot_21, const Vec3_t &trans_21, const float scale_21,
                                       std::vector<bool> &inliers);

            //! reproject points in camera (local) coordinates to the other image (as undistorted keypoints)
            void reproject_to_other_image(const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> &lm_coords_in_cam_1,
                                          std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> &reprojected_in_cam_2,
                                          const Mat33_t &rot_21, const Vec3_t &trans_21, const float scale_21, data::keyframe *keyfrm);

            //! reproject points in camera (local) coordinates to the same image (as undistorted keypoints)
            void reproject_to_same_image(const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> &lm_coords_in_cam,
                                         std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> &reprojected, data::keyframe *keyfrm);

        protected:
            //! Keyframe
            data::keyframe *keyfrm_1_;
            data::keyframe *keyfrm_2_;

            //! true: Find Sim3, false: Find SE3
            bool fix_scale_;

            // Variables to calculate in the constructor
            //! common associated points in keyframe1 and keyframe2
            //! Convert the 3D point coordinates common to keyframe1 and keyframe2 to the local coordinates of each keyframe and save them.
            std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> common_pts_in_keyfrm_1_;
            std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> common_pts_in_keyfrm_2_;

            //! Chi-square value with 2 degrees of freedom x reprojection error variance
            std::vector<float> chi_sq_x_sigma_sq_1_;
            std::vector<float> chi_sq_x_sigma_sq_2_;

            //! Feature point index of each key frame when outputting common 3D point coordinates
            std::vector<size_t> matched_indices_1_;
            std::vector<size_t> matched_indices_2_;

            //! Common 3D points
            unsigned int num_common_pts_ = 0;

            //! solution is valid or not
            bool solution_is_valid_ = false;
            //! most reliable rotation from keyframe 2 to keyframe 1
            Mat33_t best_rot_12_;
            //! most reliable translation from keyframe 2 to keyframe 1
            Vec3_t best_trans_12_;
            //! most reliable scale from keyframe 2 to keyframe 1
            float best_scale_12_;

            //! Image coordinates reprojected from common points
            //! Coordinates of 3D points reprojected on the image (distortion parameters do not apply)
            std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> reprojected_1_;
            std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> reprojected_2_;

            //! RANSAC parameters
            unsigned int min_num_inliers_;
        };

    } // namespace solve
} // namespace PLPSLAM

#endif // PLPSLAM_SOLVE_SIM3_SOLVER_H
