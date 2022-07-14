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

#ifndef PLPSLAM_OPTIMIZE_POSE_OPTIMIZER_EXTENDED_LINE_H
#define PLPSLAM_OPTIMIZE_POSE_OPTIMIZER_EXTENDED_LINE_H

namespace PLPSLAM
{

    namespace data
    {
        class frame;
    }

    namespace optimize
    {

        class pose_optimizer_extended_line
        {
        public:
            /**
             * Constructor
             * @param num_trials
             * @param num_each_iter
             */
            explicit pose_optimizer_extended_line(const unsigned int num_trials = 4, const unsigned int num_each_iter = 10);

            /**
             * Destructor
             */
            virtual ~pose_optimizer_extended_line() = default;

            /**
             * Perform pose optimization
             * @param frm
             * @return
             */
            unsigned int optimize(data::frame &frm) const;

        private:
            //! Number of robust optimization trials
            const unsigned int num_trials_ = 4;

            //! Number of optimization iterations
            const unsigned int num_each_iter_ = 10;
        };
    }
}

#endif
