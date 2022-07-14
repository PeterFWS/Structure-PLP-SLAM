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

#ifndef PLPSLAM_OPTIMIZE_SIM3_OPTIMIZER_H
#define PLPSLAM_OPTIMIZE_SIM3_OPTIMIZER_H

#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <vector>

namespace PLPSLAM
{

    namespace data
    {
        class keyframe;
        class landmark;
    } // namespace data

    namespace optimize
    {

        // FW: by far I only know this optimizer is used in loop detection
        class transform_optimizer
        {
        public:
            /**
             * Constructor
             * @param fix_scale
             * @param num_iter
             */
            explicit transform_optimizer(const bool fix_scale, const unsigned int num_iter = 10);

            /**
             * Destructor
             */
            virtual ~transform_optimizer() = default;

            /**
             * Perform optimization
             * @param keyfrm_1
             * @param keyfrm_2
             * @param matched_lms_in_keyfrm_2
             * @param g2o_Sim3_12
             * @param chi_sq
             * @return
             */
            unsigned int optimize(data::keyframe *keyfrm_1, data::keyframe *keyfrm_2,
                                  std::vector<data::landmark *> &matched_lms_in_keyfrm_2,
                                  g2o::Sim3 &g2o_Sim3_12, const float chi_sq) const;

        private:
            //! transform is Sim3 or SE3
            const bool fix_scale_;

            //! number of iterations of optimization
            const unsigned int num_iter_;
        };

    } // namespace optimize
} // namespace PLPSLAM

#endif // PLPSLAM_OPTIMIZE_SIM3_OPTIMIZER_H
