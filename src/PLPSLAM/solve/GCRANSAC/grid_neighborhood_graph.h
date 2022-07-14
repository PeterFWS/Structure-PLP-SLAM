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

#include "PLPSLAM/solve/GCRANSAC/neighborhood_graph.h"
#include <vector>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>
#include "PLPSLAM/solve/GCRANSAC/estimator/estimator.h"

namespace gcransac
{
	namespace neighborhood
	{
		// The cell structure used in the HashMap
		template <size_t _DimensionNumber>
		class GridCell
		{
		public:
			// The cell index along a particular axis
			const std::vector<size_t> idx_along_axes;
			// The index of the cell used in the hashing function
			size_t index;

			GridCell(
				const std::vector<size_t> &idx_along_axes_,
				const std::vector<size_t> &cell_number_along_axes) : idx_along_axes(idx_along_axes_)
			{
				size_t offset = 1;
				index = 0;
				for (size_t dimensionIdx = 0; dimensionIdx < _DimensionNumber; ++dimensionIdx)
				{
					index += offset * idx_along_axes_[dimensionIdx];
					offset *= cell_number_along_axes[dimensionIdx];
				}
			}

			GridCell(
				const std::vector<size_t> &idx_along_axes_,
				const size_t &cell_number_along_all_axes_) : // The number of cells along all axis in both images
															 idx_along_axes(idx_along_axes_)
			{
				size_t offset = 1;
				index = 0;
				for (size_t dimensionIdx = 0; dimensionIdx < _DimensionNumber; ++dimensionIdx)
				{
					index += offset * idx_along_axes_[dimensionIdx];
					offset *= cell_number_along_all_axes_;
				}
			}

			// Two cells are equal of their indices along all axes in both
			// images are equal.
			bool operator==(const GridCell &o) const
			{

				for (size_t dimensionIdx = 0; dimensionIdx < _DimensionNumber; ++dimensionIdx)
					if (idx_along_axes[dimensionIdx] != o.idx_along_axes[dimensionIdx])
						return false;
				return true;
			}

			// The cells are ordered in ascending order along axes
			bool operator<(const GridCell &o) const
			{
				for (size_t dimensionIdx = 0; dimensionIdx < _DimensionNumber; ++dimensionIdx)
				{
					const auto &idx1 = idx_along_axes[dimensionIdx];
					const auto &idx2 = o.idx_along_axes[dimensionIdx];

					if (idx1 < idx2)
						return true;
					else if (idx1 > idx2)
						return true;
				}
				return false;
			}
		};
	}
}

namespace std
{
	template <size_t _DimensionNumber>
	struct hash<gcransac::neighborhood::GridCell<_DimensionNumber>>
	{
		std::size_t operator()(const gcransac::neighborhood::GridCell<_DimensionNumber> &coord) const noexcept
		{
			// The cell's index value is used in the hashing function
			return coord.index;
		}
	};
}

namespace gcransac
{
	namespace neighborhood
	{
		template <size_t _DimensionNumber>
		class GridNeighborhoodGraph : public NeighborhoodGraph<cv::Mat>
		{
		protected:
			std::vector<double> cell_sizes; // The size of a cell along a particular dimension

			// Number of cells along the image axes.
			size_t cell_number_along_all_axes;

			// The grid is stored in the HashMap where the key is defined
			// via the cell coordinates.
			std::unordered_map<GridCell<_DimensionNumber>, std::vector<size_t>> grid;

			// The pointer to the cell (i.e., key in the grid) for each point.
			// It is faster to store them than to recreate the cell structure
			// whenever is needed.
			std::vector<const GridCell<_DimensionNumber> *> cells_of_points;

		public:
			GridNeighborhoodGraph() : NeighborhoodGraph() {}

			GridNeighborhoodGraph(
				const cv::Mat *const container_, // The pointer of the container consisting of the data points.
				const std::vector<double> &cell_sizes_,
				const size_t &cell_number_along_all_axes_) : cell_number_along_all_axes(cell_number_along_all_axes_),
															 cell_sizes(cell_sizes_),
															 NeighborhoodGraph(container_)
			{
				if (container_->cols != _DimensionNumber)
				{
					fprintf(stderr, "The data dimension (%d) does not match with the expected dimension (%d).\n",
							container_->cols, _DimensionNumber);
					return;
				}

				initialized = initialize(container);
			}

			bool initialize(const cv::Mat *const container_);
			inline const std::vector<size_t> &getNeighbors(size_t point_idx_) const;
		};

		template <size_t _DimensionNumber>
		bool GridNeighborhoodGraph<_DimensionNumber>::initialize(const cv::Mat *const container_)
		{
			// The number of points
			const size_t point_number = container_->rows;

			// Pointer to the coordinates
			const double *points_ptr =
				reinterpret_cast<double *>(container_->data);

			// Iterate through the points and put each into the grid.
			std::vector<size_t> indices(_DimensionNumber);
			for (size_t row = 0; row < point_number; ++row)
			{
				for (size_t dimensionIdx = 0; dimensionIdx < _DimensionNumber; ++dimensionIdx)
				{
					// The index of the cell along the current axis
					indices[dimensionIdx] =
						*(points_ptr++) / cell_sizes[dimensionIdx];
				}

				// Constructing the cell structure which is used in the HashMap.
				const GridCell<_DimensionNumber> cell(indices,					   // The cell indices along the axes
													  cell_number_along_all_axes); // The number of cells along all axis in both images

				// Add the current point's index to the grid.
				grid[cell].push_back(row);
			}

			// Iterate through all cells and store the corresponding
			// cell pointer for each point.
			cells_of_points.resize(point_number);
			for (const auto &element : grid)
			{
				// Get the pointer of the cell.
				const GridCell<_DimensionNumber> *cell = &element.first;

				// Increase the edge number in the neighborhood graph.
				// All possible pairs of points in each cell are neighbors,
				// therefore, the neighbor number is "n choose 2" for the
				// current cell.
				const size_t n = element.second.size();
				neighbor_number += n * (n - 1) / 2;

				// Iterate through all points in the cell.
				for (const size_t &point_idx : element.second)
					cells_of_points[point_idx] = cell; // Store the cell pointer for each contained point.
			}

			return neighbor_number > 0;
		}

		template <size_t _DimensionNumber>
		inline const std::vector<size_t> &GridNeighborhoodGraph<_DimensionNumber>::getNeighbors(size_t point_idx_) const
		{
			// Get the pointer of the cell in which the point is.
			const GridCell<_DimensionNumber> *cell = cells_of_points[point_idx_];
			// Return the vector containing all the points in the cell.
			return grid.at(*cell);
		}
	}
}