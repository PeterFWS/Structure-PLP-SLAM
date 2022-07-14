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

#include "PLPSLAM/solve/common.h"
#include "PLPSLAM/solve/homography_solver.h"
#include "PLPSLAM/util/converter.h"
#include "PLPSLAM/util/random_array.h"

#include <iostream>

namespace PLPSLAM
{
    namespace solve
    {

        homography_solver::homography_solver(const std::vector<cv::KeyPoint> &undist_keypts_1,
                                             const std::vector<cv::KeyPoint> &undist_keypts_2,
                                             const std::vector<std::pair<int, int>> &matches_12, const float sigma)
            : undist_keypts_1_(undist_keypts_1), // ref
              undist_keypts_2_(undist_keypts_2), // cur
              matches_12_(matches_12),
              sigma_(sigma)
        {
        }

        // FW: same constructor but with image cols and rows for graph-cut ransac
        homography_solver::homography_solver(const std::vector<cv::KeyPoint> &undist_keypts_1,
                                             const std::vector<cv::KeyPoint> &undist_keypts_2,
                                             const std::vector<std::pair<int, int>> &matches_12, const float sigma,
                                             const int img_cols, const int img_rows)
            : undist_keypts_1_(undist_keypts_1), // ref
              undist_keypts_2_(undist_keypts_2), // cur
              matches_12_(matches_12),
              sigma_(sigma),
              _source_image_cols(img_cols),
              _source_image_rows(img_rows)
        {
        }

        void homography_solver::find_via_ransac(const unsigned int max_num_iter, const bool recompute)
        {
            const auto num_matches = static_cast<unsigned int>(matches_12_.size());

            // 0. Normalize keypoint coordinates

            // apply normalization
            std::vector<cv::Point2f> normalized_keypts_1, normalized_keypts_2;
            Mat33_t transform_1, transform_2;
            normalize(undist_keypts_1_, normalized_keypts_1, transform_1);
            normalize(undist_keypts_2_, normalized_keypts_2, transform_2);

            const Mat33_t transform_2_inv = transform_2.inverse();

            // 1. Prepare for RANSAC

            // minimum number of samples (= 8)
            constexpr unsigned int min_set_size = 8;
            if (num_matches < min_set_size)
            {
                solution_is_valid_ = false;
                return;
            }

            // RANSAC variables
            best_score_ = 0.0;
            is_inlier_match_ = std::vector<bool>(num_matches, false);

            // minimum set of keypoint matches
            std::vector<cv::Point2f> min_set_keypts_1(min_set_size);
            std::vector<cv::Point2f> min_set_keypts_2(min_set_size);

            // shared variables in RANSAC loop
            // homography matrix from shot 1 to shot 2, and
            // homography matrix from shot 2 to shot 1,
            Mat33_t H_21_in_sac, H_12_in_sac;
            // inlier/outlier flags
            std::vector<bool> is_inlier_match_in_sac(num_matches, false);
            // score of homography matrix
            float score_in_sac;

            // 2. RANSAC loop

            for (unsigned int iter = 0; iter < max_num_iter; ++iter)
            {
                // 2-1. Create a minimum set
                const auto indices = util::create_random_array(min_set_size, 0U, num_matches - 1);
                for (unsigned int i = 0; i < min_set_size; ++i)
                {
                    const auto idx = indices.at(i);
                    min_set_keypts_1.at(i) = normalized_keypts_1.at(matches_12_.at(idx).first);
                    min_set_keypts_2.at(i) = normalized_keypts_2.at(matches_12_.at(idx).second);
                }

                // 2-2. Compute a homography matrix
                const Mat33_t normalized_H_21 = compute_H_21(min_set_keypts_1, min_set_keypts_2);
                H_21_in_sac = transform_2_inv * normalized_H_21 * transform_1;

                // 2-3. Check inliers and compute a score
                score_in_sac = check_inliers(H_21_in_sac, is_inlier_match_in_sac);

                // 2-4. Update the best model
                if (best_score_ < score_in_sac)
                {
                    best_score_ = score_in_sac;
                    best_H_21_ = H_21_in_sac;
                    is_inlier_match_ = is_inlier_match_in_sac;
                }
            }

            const auto num_inliers = std::count(is_inlier_match_.begin(), is_inlier_match_.end(), true);
            solution_is_valid_ = (best_score_ > 0.0) && (num_inliers >= min_set_size);

            if (!recompute || !solution_is_valid_)
            {
                return;
            }

            // 3. Recompute a homography matrix only with the inlier matches

            std::vector<cv::Point2f> inlier_normalized_keypts_1;
            std::vector<cv::Point2f> inlier_normalized_keypts_2;
            inlier_normalized_keypts_1.reserve(matches_12_.size());
            inlier_normalized_keypts_2.reserve(matches_12_.size());
            for (unsigned int i = 0; i < matches_12_.size(); ++i)
            {
                if (is_inlier_match_.at(i))
                {
                    inlier_normalized_keypts_1.push_back(normalized_keypts_1.at(matches_12_.at(i).first));
                    inlier_normalized_keypts_2.push_back(normalized_keypts_2.at(matches_12_.at(i).second));
                }
            }
            const Mat33_t normalized_H_21 = solve::homography_solver::compute_H_21(inlier_normalized_keypts_1, inlier_normalized_keypts_2);
            best_H_21_ = transform_2_inv * normalized_H_21 * transform_1; // Normalized Direct Linear Transform H = T2^-1 * H_normalized * T1
            best_score_ = check_inliers(best_H_21_, is_inlier_match_);
        }

        // // FW: the only difference is using 4 points (To be updated for planar-based tracking)
        // void homography_solver::find_via_ransac_extended(const unsigned int max_num_iter, const bool recompute)
        // {
        //     const auto num_matches = static_cast<unsigned int>(matches_12_.size());

        //     // 0. Normalize keypoint coordinates

        //     // apply normalization
        //     std::vector<cv::Point2f> normalized_keypts_1, normalized_keypts_2;
        //     Mat33_t transform_1, transform_2;
        //     normalize(undist_keypts_1_, normalized_keypts_1, transform_1);
        //     normalize(undist_keypts_2_, normalized_keypts_2, transform_2);

        //     const Mat33_t transform_2_inv = transform_2.inverse();

        //     // 1. Prepare for RANSAC

        //     // minimum number of samples (= 8)
        //     constexpr unsigned int min_set_size = 4;
        //     if (num_matches < min_set_size)
        //     {
        //         solution_is_valid_ = false;
        //         return;
        //     }

        //     // RANSAC variables
        //     best_score_ = 0.0;
        //     is_inlier_match_ = std::vector<bool>(num_matches, false);

        //     // minimum set of keypoint matches
        //     std::vector<cv::Point2f> min_set_keypts_1(min_set_size);
        //     std::vector<cv::Point2f> min_set_keypts_2(min_set_size);

        //     // shared variables in RANSAC loop
        //     // homography matrix from shot 1 to shot 2, and
        //     // homography matrix from shot 2 to shot 1,
        //     Mat33_t H_21_in_sac, H_12_in_sac;
        //     // inlier/outlier flags
        //     std::vector<bool> is_inlier_match_in_sac(num_matches, false);
        //     // score of homography matrix
        //     float score_in_sac;

        //     // 2. RANSAC loop

        //     for (unsigned int iter = 0; iter < max_num_iter; ++iter)
        //     {
        //         // 2-1. Create a minimum set
        //         const auto indices = util::create_random_array(min_set_size, 0U, num_matches - 1);
        //         for (unsigned int i = 0; i < min_set_size; ++i)
        //         {
        //             const auto idx = indices.at(i);
        //             min_set_keypts_1.at(i) = normalized_keypts_1.at(matches_12_.at(idx).first);
        //             min_set_keypts_2.at(i) = normalized_keypts_2.at(matches_12_.at(idx).second);
        //         }

        //         // 2-2. Compute a homography matrix
        //         const Mat33_t normalized_H_21 = compute_H_21(min_set_keypts_1, min_set_keypts_2);
        //         H_21_in_sac = transform_2_inv * normalized_H_21 * transform_1;

        //         // 2-3. Check inliers and compute a score
        //         score_in_sac = check_inliers(H_21_in_sac, is_inlier_match_in_sac);

        //         // 2-4. Update the best model
        //         if (best_score_ < score_in_sac)
        //         {
        //             best_score_ = score_in_sac;
        //             best_H_21_ = H_21_in_sac;
        //             is_inlier_match_ = is_inlier_match_in_sac;
        //         }
        //     }

        //     const auto num_inliers = std::count(is_inlier_match_.begin(), is_inlier_match_.end(), true);
        //     solution_is_valid_ = (best_score_ > 0.0) && (num_inliers >= min_set_size);

        //     if (!recompute || !solution_is_valid_)
        //     {
        //         return;
        //     }

        //     // 3. Recompute a homography matrix only with the inlier matches

        //     std::vector<cv::Point2f> inlier_normalized_keypts_1;
        //     std::vector<cv::Point2f> inlier_normalized_keypts_2;
        //     inlier_normalized_keypts_1.reserve(matches_12_.size());
        //     inlier_normalized_keypts_2.reserve(matches_12_.size());
        //     for (unsigned int i = 0; i < matches_12_.size(); ++i)
        //     {
        //         if (is_inlier_match_.at(i))
        //         {
        //             inlier_normalized_keypts_1.push_back(normalized_keypts_1.at(matches_12_.at(i).first));
        //             inlier_normalized_keypts_2.push_back(normalized_keypts_2.at(matches_12_.at(i).second));
        //         }
        //     }
        //     const Mat33_t normalized_H_21 = solve::homography_solver::compute_H_21(inlier_normalized_keypts_1, inlier_normalized_keypts_2);
        //     best_H_21_ = transform_2_inv * normalized_H_21 * transform_1; // Normalized Direct Linear Transform H = T2^-1 * H_normalized * T1
        //     best_score_ = check_inliers(best_H_21_, is_inlier_match_);
        // }

        // FW: find homography using graph-cut ransac
        void homography_solver::find_via_graph_cut_ransac()
        {
            // Parameters used for Homography estimation
            const double confidence_ = 0.99;                          // The RANSAC confidence value
            const int fps_ = 30;                                      // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
            const double inlier_outlier_threshold_homography_ = 2.00; // The used inlier-outlier threshold in GC-RANSAC for homography estimation.
            const double spatial_coherence_weight_ = 0.975;           // The weigd_t of the spatial coherence term in the graph-cut energy minimization.
            const size_t cell_number_in_neighborhood_graph_ = 8;      // The number of cells along each axis in the neighborhood graph.
            const double minimum_inlier_ratio_for_sprt_ = 0.1;        // An assumption about the minimum inlier ratio used for the SPRT test

            // [1] The detected point correspondences. Each row is of format "x1 y1 x2 y2"
            cv::Mat points(matches_12_.size(), 4, CV_64F);
            for (unsigned int i = 0; i < matches_12_.size(); i++)
            {
                points.at<double>(i, 0) = undist_keypts_1_.at(matches_12_.at(i).first).pt.x;  // x1
                points.at<double>(i, 1) = undist_keypts_1_.at(matches_12_.at(i).first).pt.y;  // y1
                points.at<double>(i, 2) = undist_keypts_2_.at(matches_12_.at(i).second).pt.x; // x2
                points.at<double>(i, 3) = undist_keypts_2_.at(matches_12_.at(i).second).pt.y; // y2
            }

            // [2] Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
            // in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
            std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
            start = std::chrono::system_clock::now();                      // The starting time of the neighborhood calculation

            int destination_image_cols = _source_image_cols; // for convenient
            int destination_image_rows = _source_image_rows;
            gcransac::neighborhood::GridNeighborhoodGraph<4> neighborhood(&points,
                                                                          {_source_image_cols / static_cast<double>(cell_number_in_neighborhood_graph_),
                                                                           _source_image_rows / static_cast<double>(cell_number_in_neighborhood_graph_),
                                                                           destination_image_cols / static_cast<double>(cell_number_in_neighborhood_graph_),
                                                                           destination_image_rows / static_cast<double>(cell_number_in_neighborhood_graph_)},
                                                                          cell_number_in_neighborhood_graph_);

            end = std::chrono::system_clock::now();                      // The end time of the neighborhood calculation
            std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
            // printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

            // // Checking if the neighborhood graph is initialized successfully.
            // if (!neighborhood.isInitialized())
            // {
            //     fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
            //     return;
            // }
            // else
            // {
            //     fprintf(stderr, "The neighborhood graph is initialized successfully.\n");
            // }

            // [3] Apply Graph-cut RANSAC
            gcransac::utils::DefaultHomographyEstimator estimator;
            std::vector<int> inliers;
            gcransac::Homography model; // Model model; ? from original cpp example

            // Initializing SPRT test (sequential probability test)
            gcransac::preemption::SPRTPreemptiveVerfication<gcransac::utils::DefaultHomographyEstimator> preemptive_verification(
                points,
                estimator,
                minimum_inlier_ratio_for_sprt_);

            gcransac::GCRANSAC<gcransac::utils::DefaultHomographyEstimator,
                               gcransac::neighborhood::GridNeighborhoodGraph<4>,
                               gcransac::MSACScoringFunction<gcransac::utils::DefaultHomographyEstimator>,
                               gcransac::preemption::SPRTPreemptiveVerfication<gcransac::utils::DefaultHomographyEstimator>>
                gcransac;

            gcransac.setFPS(fps_);                                                             // Set the desired FPS (-1 means no limit)
            gcransac.settings.threshold = inlier_outlier_threshold_homography_;                // The inlier-outlier threshold
            gcransac.settings.spatial_coherence_weight = spatial_coherence_weight_;            // The weight of the spatial coherence term
            gcransac.settings.confidence = confidence_;                                        // The required confidence in the results
            gcransac.settings.max_local_optimization_number = 50;                              // The maximm number of local optimizations
            gcransac.settings.max_iteration_number = 5000;                                     // The maximum number of iterations
            gcransac.settings.min_iteration_number = 50;                                       // The minimum number of iterations
            gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
            gcransac.settings.core_number = std::thread::hardware_concurrency();               // The number of parallel processes

            // Initialize the samplers
            // The main sampler is used inside the local optimization
            gcransac::sampler::ProgressiveNapsacSampler<4>
                main_sampler(&points,
                             {16, 8, 4, 2},                                 // The layer of grids. The cells of the finest grid are of dimension
                                                                            // (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
                             estimator.sampleSize(),                        // The size of a minimal sample
                             {static_cast<double>(_source_image_cols),      // The width of the source image
                              static_cast<double>(_source_image_rows),      // The height of the source image
                              static_cast<double>(destination_image_cols),  // The width of the destination image
                              static_cast<double>(destination_image_rows)}, // The height of the destination image
                             0.5);                                          // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling

            gcransac::sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

            // // Checking if the samplers are initialized successfully.
            // if (!main_sampler.isInitialized() ||
            //     !local_optimization_sampler.isInitialized())
            // {
            //     fprintf(stderr, "One of the samplers is not initialized successfully.\n");
            //     return;
            // }
            // else
            // {
            //     fprintf(stderr, "One of the samplers is initialized successfully.\n");
            // }

            // Start GC-RANSAC
            gcransac.run(points,
                         estimator,
                         &main_sampler,
                         &local_optimization_sampler,
                         &neighborhood,
                         model,
                         preemptive_verification);

            // Get the statistics of the results
            const gcransac::utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();

            // // Write statistics
            // printf("Elapsed time = %f secs\n", statistics.processing_time);
            // printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));
            // printf("Applied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
            // printf("Applied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
            // printf("Number of iterations = %d\n\n", static_cast<int>(statistics.iteration_number));

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    best_H_21_(i, j) = model.descriptor(i, j);
                }
            }

            // std::cout << best_H_21_ << std::endl;

            // [4] update information
            // inliers match
            is_inlier_match_ = std::vector<bool>(matches_12_.size(), false);
            for (unsigned int i = 0; i < statistics.inliers.size(); i++)
            {
                is_inlier_match_[statistics.inliers[i]] = true;
            }

            // best score
            best_score_ = 0.0;
            best_score_ = check_inliers(best_H_21_, is_inlier_match_); // check STE (symmetric transfor error)

            // valid solution
            constexpr unsigned int min_set_size = 8;
            solution_is_valid_ = (best_score_ > 0.0) && (statistics.inliers.size() >= min_set_size);
        }

        Mat33_t homography_solver::compute_H_21(const std::vector<cv::Point2f> &keypts_1, const std::vector<cv::Point2f> &keypts_2)
        {
            // https://www.uio.no/studier/emner/matnat/its/UNIK4690/v16/forelesninger/lecture_4_3-estimating-homographies-from-feature-correspondences.pdf

            assert(keypts_1.size() == keypts_2.size());

            const auto num_points = keypts_1.size();

            // construct lineear system Ah = 0
            typedef Eigen::Matrix<Mat33_t::Scalar, Eigen::Dynamic, 9> CoeffMatrix;
            CoeffMatrix A(2 * num_points, 9); // size: 2 x num_keypoints by 9
            for (unsigned int i = 0; i < num_points; ++i)
            {                                                                                                      // see slides from the link, pp. 11
                A.block<1, 3>(2 * i, 0) = Vec3_t::Zero();                                                          // 0 0 0
                A.block<1, 3>(2 * i, 3) = -util::converter::to_homogeneous(keypts_1.at(i));                        // -u1 -v1 -1
                A.block<1, 3>(2 * i, 6) = keypts_2.at(i).y * util::converter::to_homogeneous(keypts_1.at(i));      // v2*u1 v2*v1 v2
                A.block<1, 3>(2 * i + 1, 0) = util::converter::to_homogeneous(keypts_1.at(i));                     // u1 v1 1
                A.block<1, 3>(2 * i + 1, 3) = Vec3_t::Zero();                                                      // 0 0 0
                A.block<1, 3>(2 * i + 1, 6) = -keypts_2.at(i).x * util::converter::to_homogeneous(keypts_1.at(i)); //-u2*u1 -u2*v1 -u2
            }

            // SVD: A = USV^T
            // when S is diagonal with positive values in decending order, h is the last column of V
            const Eigen::JacobiSVD<CoeffMatrix> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const Eigen::Matrix<Mat33_t::Scalar, 9, 1> v = svd.matrixV().col(8);

            // reconstruct H from h
            // need transpose() because elements are contained as col-major after it was constructed from a pointer
            const Mat33_t H_21 = Mat33_t(v.data()).transpose();

            return H_21;
        }

        bool homography_solver::decompose(const Mat33_t &H_21, const Mat33_t &cam_matrix_1, const Mat33_t &cam_matrix_2,
                                          eigen_alloc_vector<Mat33_t> &init_rots, eigen_alloc_vector<Vec3_t> &init_transes, eigen_alloc_vector<Vec3_t> &init_normals)
        {
            // Motion and structure from motion in a piecewise planar environment
            // (Faugeras et al. in IJPRAI 1988)

            init_rots.reserve(8);
            init_transes.reserve(8);
            init_normals.reserve(8);

            const Mat33_t A = cam_matrix_2.inverse() * H_21 * cam_matrix_1;

            // Eq.(7) SVD
            const Eigen::JacobiSVD<Mat33_t> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const Mat33_t &U = svd.matrixU();
            const Vec3_t &lambda = svd.singularValues();
            const Mat33_t &V = svd.matrixV();
            const Mat33_t Vt = V.transpose();

            const float d1 = lambda(0);
            const float d2 = lambda(1);
            const float d3 = lambda(2);

            // check rank condition
            if (d1 / d2 < 1.0001 || d2 / d3 < 1.0001)
            {
                return false;
            }

            // intermediate variable in Eq.(8)
            const float s = U.determinant() * Vt.determinant();

            // x1 and x3 in Eq.(12)
            const float aux_1 = std::sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
            const float aux_3 = std::sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
            const std::array<float, 4> x1s = {{aux_1, aux_1, -aux_1, -aux_1}};
            const std::array<float, 4> x3s = {{aux_3, -aux_3, aux_3, -aux_3}};

            // when d' > 0

            // Eq.(13)
            const float aux_sin_theta = std::sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);
            const float cos_theta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
            const std::array<float, 4> aux_sin_thetas = {{aux_sin_theta, -aux_sin_theta, -aux_sin_theta, aux_sin_theta}};

            for (unsigned int i = 0; i < 4; ++i)
            {
                // Eq.(13)
                Mat33_t aux_rot = Mat33_t::Identity();
                aux_rot(0, 0) = cos_theta;
                aux_rot(0, 2) = -aux_sin_thetas.at(i);
                aux_rot(2, 0) = aux_sin_thetas.at(i);
                aux_rot(2, 2) = cos_theta;
                // Eq.(8)
                const Mat33_t init_rot = s * U * aux_rot * Vt;
                init_rots.push_back(init_rot);

                // Eq.(14)
                Vec3_t aux_trans{x1s.at(i), 0.0, -x3s.at(i)};
                aux_trans *= d1 - d3;
                // Eq.(8)
                const Vec3_t init_trans = U * aux_trans;
                init_transes.emplace_back(init_trans / init_trans.norm());

                // Eq.(9)
                const Vec3_t aux_normal{x1s.at(i), 0.0, x3s.at(i)};
                // Eq.(8)
                Vec3_t init_normal = V * aux_normal;
                if (init_normal(2) < 0)
                {
                    init_normal = -init_normal;
                }
                init_normals.push_back(init_normal);
            }

            // when d' < 0

            // Eq.(13)
            const float aux_sin_phi = std::sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);
            const float cos_phi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
            const std::array<float, 4> sin_phis = {{aux_sin_phi, -aux_sin_phi, -aux_sin_phi, aux_sin_phi}};

            for (unsigned int i = 0; i < 4; ++i)
            {
                // Eq.(15)
                Mat33_t aux_rot = Mat33_t::Identity();
                aux_rot(0, 0) = cos_phi;
                aux_rot(0, 2) = sin_phis.at(i);
                aux_rot(1, 1) = -1;
                aux_rot(2, 0) = sin_phis.at(i);
                aux_rot(2, 2) = -cos_phi;
                // Eq.(8)
                const Mat33_t init_rot = s * U * aux_rot * Vt;
                init_rots.push_back(init_rot);

                // Eq.(16)
                Vec3_t aux_trans{x1s.at(i), 0.0, x3s.at(i)};
                aux_trans(0) = x1s.at(i);
                aux_trans(1) = 0;
                aux_trans(2) = x3s.at(i);
                aux_trans *= d1 + d3;
                // Eq.(8)
                const Vec3_t init_trans = U * aux_trans;
                init_transes.emplace_back(init_trans / init_trans.norm());

                // Eq.(9)
                const Vec3_t aux_normal{x1s.at(i), 0.0, x3s.at(i)};
                Vec3_t init_normal = V * aux_normal;
                if (init_normal(2) < 0)
                {
                    init_normal = -init_normal;
                }
                init_normals.push_back(init_normal);
            }

            return true;
        }

        float homography_solver::check_inliers(const Mat33_t &H_21, std::vector<bool> &is_inlier_match)
        {
            const auto num_matches = matches_12_.size();

            // chi-squared value (p=0.05, n=2)
            constexpr float chi_sq_thr = 5.991;

            is_inlier_match.resize(num_matches);

            const Mat33_t H_12 = H_21.inverse();

            float score = 0;

            const float inv_sigma_sq = 1.0 / (sigma_ * sigma_);

            for (unsigned int i = 0; i < num_matches; ++i)
            {
                const auto &keypt_1 = undist_keypts_1_.at(matches_12_.at(i).first);
                const auto &keypt_2 = undist_keypts_2_.at(matches_12_.at(i).second);

                // 1. Transform to homogeneous coordinates

                const Vec3_t pt_1 = util::converter::to_homogeneous(keypt_1.pt);
                const Vec3_t pt_2 = util::converter::to_homogeneous(keypt_2.pt);

                // 2. Compute symmetric transfer error

                // 2-1. Transform a point in shot 1 to the epipolar line in shot 2,
                //      then compute a transfer error (= dot product)

                Vec3_t transformed_pt_1 = H_21 * pt_1;
                transformed_pt_1 = transformed_pt_1 / transformed_pt_1(2);

                const float dist_sq_1 = (pt_2 - transformed_pt_1).squaredNorm();

                // standardization
                const float chi_sq_1 = dist_sq_1 * inv_sigma_sq;

                // if a match is inlier, accumulate the score
                if (chi_sq_thr < chi_sq_1)
                {
                    is_inlier_match.at(i) = false;
                    continue;
                }
                else
                {
                    is_inlier_match.at(i) = true;
                    score += chi_sq_thr - chi_sq_1;
                }

                // 2. Transform a point in shot 2 to the epipolar line in shot 1,
                //    then compute a transfer error (= dot product)

                Vec3_t transformed_pt_2 = H_12 * pt_2;
                transformed_pt_2 = transformed_pt_2 / transformed_pt_2(2);

                const float dist_sq_2 = (pt_1 - transformed_pt_2).squaredNorm();

                // standardization
                const float chi_sq_2 = dist_sq_2 * inv_sigma_sq;

                // if a match is inlier, accumulate the score
                if (chi_sq_thr < chi_sq_2)
                {
                    is_inlier_match.at(i) = false;
                    continue;
                }
                else
                {
                    is_inlier_match.at(i) = true;
                    score += chi_sq_thr - chi_sq_2;
                }
            }

            return score;
        }

    } // namespace solve
} // namespace PLPSLAM
