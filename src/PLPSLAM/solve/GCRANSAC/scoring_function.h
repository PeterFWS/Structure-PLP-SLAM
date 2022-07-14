// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <math.h>
#include <random>
#include <vector>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

namespace gcransac
{
	/* RANSAC Scoring */
	struct Score
	{

		/* Number of inliers, rectangular gain function */
		size_t inlier_number;

		/* Score */
		double value;

		Score() : inlier_number(0),
				  value(0.0)
		{
		}

		OLGA_INLINE bool operator<(const Score &score_)
		{
			return value < score_.value;
		}

		OLGA_INLINE bool operator>(const Score &score_)
		{
			return *this > score_;
		}
	};

	template <class _ModelEstimator>
	class ScoringFunction
	{
	public:
		ScoringFunction()
		{
		}

		virtual ~ScoringFunction()
		{
		}

		virtual OLGA_INLINE Score getScore(const cv::Mat &points_,			   // The input data points
										   Model &model_,					   // The current model parameters
										   const _ModelEstimator &estimator_,  // The model estimator
										   const double threshold_,			   // The inlier-outlier threshold
										   std::vector<size_t> &inliers_,	   // The selected inliers
										   const Score &best_score_ = Score(), // The score of the current so-far-the-best model
										   const bool store_inliers_ = true) const = 0;

		virtual void initialize(const double threshold_,
								const size_t point_number_) = 0;
	};

	template <class _ModelEstimator>
	class MSACScoringFunction : public ScoringFunction<_ModelEstimator>
	{
	protected:
		double squared_truncated_threshold; // Squared truncated threshold
		size_t point_number;				// Number of points
		// Verify only every k-th point when doing the score calculation. This maybe is beneficial if
		// there is a time sensitive application and verifying the model on a subset of points
		// is enough.
		size_t verify_every_kth_point;

	public:
		MSACScoringFunction() : verify_every_kth_point(1)
		{
		}

		~MSACScoringFunction()
		{
		}

		void setSkippingParameter(const size_t verify_every_kth_point_)
		{
			verify_every_kth_point = verify_every_kth_point_;
		}

		void initialize(const double squared_truncated_threshold_,
						const size_t point_number_)
		{
			squared_truncated_threshold = squared_truncated_threshold_;
			point_number = point_number_;
		}

		// Return the score of a model w.r.t. the data points and the threshold
		OLGA_INLINE Score getScore(const cv::Mat &points_,			   // The input data points
								   Model &model_,					   // The current model parameters
								   const _ModelEstimator &estimator_,  // The model estimator
								   const double threshold_,			   // The inlier-outlier threshold
								   std::vector<size_t> &inliers_,	   // The selected inliers
								   const Score &best_score_ = Score(), // The score of the current so-far-the-best model
								   const bool store_inliers_ = true) const
		{
			Score score;		// The current score
			if (store_inliers_) // If the inlier should be stored, clear the variables
				inliers_.clear();
			double squared_residual; // The point-to-model residual

			// Iterate through all points, calculate the squared_residuals and store the points as inliers if needed.
			for (int point_idx = 0; point_idx < point_number; point_idx += verify_every_kth_point)
			{
				// Calculate the point-to-model residual
				squared_residual =
					estimator_.squaredResidual(points_.row(point_idx),
											   model_.descriptor);

				// If the residual is smaller than the threshold, store it as an inlier and
				// increase the score.
				if (squared_residual < squared_truncated_threshold)
				{
					if (store_inliers_) // Store the point as an inlier if needed.
						inliers_.emplace_back(point_idx);

					// Increase the inlier number
					++(score.inlier_number);
					// Increase the score. The original truncated quadratic loss is as follows:
					// 1 - residual^2 / threshold^2. For RANSAC, -residual^2 is enough.
					// It has been re-arranged as
					// score = 1 - residual^2 / threshold^2				->
					// score threshold^2 = threshold^2 - residual^2		->
					// score threshold^2 - threshold^2 = - residual^2.
					// This is faster to calculate and it is normalized back afterwards.
					score.value -= squared_residual; // Truncated quadratic cost
													 //score.value += 1.0 - squared_residual / squared_truncated_threshold; // Truncated quadratic cost
				}

				// Interrupt if there is no chance of being better than the best model
				if (point_number - point_idx + score.inlier_number < best_score_.inlier_number)
					return Score();
			}

			if (score.inlier_number == 0)
				return Score();

			// Normalizing the score to get back the original MSAC one.
			// This is not necessarily needed, but I keep it like this
			// maybe something will later be built on the exact MSAC score.
			score.value =
				(score.value + score.inlier_number * squared_truncated_threshold) /
				squared_truncated_threshold;

			// Return the final score
			return score;
		}
	};
}