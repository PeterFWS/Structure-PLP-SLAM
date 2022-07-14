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

#ifndef PANGOLIN_VIEWER_COLOR_SCHEME_H
#define PANGOLIN_VIEWER_COLOR_SCHEME_H

#include <array>
#include <string>

namespace pangolin_viewer
{

    class color_scheme
    {
    public:
        explicit color_scheme(const std::string &color_set_str);

        virtual ~color_scheme() = default;

        //! background color
        std::array<float, 4> bg_{};
        //! grid color
        std::array<float, 3> grid_{};
        //! current camera color
        std::array<float, 3> curr_cam_{};
        //! keyframe line color
        std::array<float, 3> kf_line_{};
        //! graph edge line color
        std::array<float, 4> graph_line_{};
        //! landmark color
        std::array<float, 3> lm_{};
        //! local_landmark color
        std::array<float, 3> local_lm_{};

    private:
        void set_color_as_white();

        void set_color_as_black();

        void set_color_as_purple();

        static bool stricmp(const std::string &str1, const std::string &str2);
    };

} // namespace pangolin_viewer

#endif // PANGOLIN_VIEWER_COLOR_SCHEME_H
