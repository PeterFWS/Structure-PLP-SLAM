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

#ifndef PLPSLAM_INITIALIZE_PERSPECTIVE_H
#define PLPSLAM_INITIALIZE_PERSPECTIVE_H

#include "PLPSLAM/type.h"
#include "PLPSLAM/initialize/base.h"

#include <opencv2/core.hpp>

namespace PLPSLAM
{

    namespace data
    {
        class frame;
    } // namespace data

    namespace initialize
    {

        class perspective final : public base
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            perspective() = delete;

            //! Constructor
            perspective(const data::frame &ref_frm,
                        const unsigned int num_ransac_iters, const unsigned int min_num_triangulated,
                        const float parallax_deg_thr, const float reproj_err_thr);

            //! Destructor
            ~perspective() override;

            //! Initialize with the current frame
            bool initialize(const data::frame &cur_frm, const std::vector<int> &ref_matches_with_cur) override;

        private:
            //! Reconstruct the initial map with the H matrix
            //! (NOTE: the output variables will be set if succeeded)
            bool reconstruct_with_H(const Mat33_t &H_ref_to_cur, const std::vector<bool> &is_inlier_match);

            //! Reconstruct the initial map with the F matrix
            //! (NOTE: the output variables will be set if succeeded)
            bool reconstruct_with_F(const Mat33_t &F_ref_to_cur, const std::vector<bool> &is_inlier_match);

            //! Get the camera matrix from the camera object
            static Mat33_t get_camera_matrix(camera::base *camera);

            //! camera matrix of the reference frame
            const Mat33_t ref_cam_matrix_;
            //! camera matrix of the current frame
            Mat33_t cur_cam_matrix_;
        };

    } // namespace initialize
} // namespace PLPSLAM

#endif // PLPSLAM_INITIALIZE_PERSPECTIVE_H
