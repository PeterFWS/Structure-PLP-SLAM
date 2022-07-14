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

#include "PLPSLAM/optimize/g2o/line3d.h"
#include <g2o/stuff/misc.h>

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {

            using namespace std;

            static inline ::g2o::Matrix3 _skew(const ::g2o::Vector3 &t)
            {
                ::g2o::Matrix3 S;
                S << 0, -t.z(), t.y(), t.z(), 0, -t.x(), -t.y(), t.x(), 0;
                return S;
            }

            Vector6 Line3D::toCartesian() const
            {
                Vector6 cartesian;
                cartesian.tail<3>() = d() / d().norm();
                ::g2o::Matrix3 W = -_skew(d());
                number_t damping = ::g2o::cst(1e-9);
                ::g2o::Matrix3 A = W.transpose() * W + (::g2o::Matrix3::Identity() * damping);
                cartesian.head<3>() = A.ldlt().solve(W.transpose() * w());
                return cartesian;
            }

            Line3D operator*(const ::g2o::Isometry3 &t, const Line3D &line)
            {
                Matrix6 A = Matrix6::Zero();
                A.block<3, 3>(0, 0) = t.linear();
                A.block<3, 3>(0, 3) = _skew(t.translation()) * t.linear();
                A.block<3, 3>(3, 3) = t.linear();
                Vector6 v = (Vector6)line;
                return Line3D(A * v);
            }

            Vector6 transformCartesianLine(const ::g2o::Isometry3 &t, const Vector6 &line)
            {
                Vector6 l;
                l.head<3>() = t * line.head<3>();
                l.tail<3>() = t.linear() * line.tail<3>();
                return normalizeCartesianLine(l);
            }

            Vector6 normalizeCartesianLine(const Vector6 &line)
            {
                ::g2o::Vector3 p0 = line.head<3>();
                ::g2o::Vector3 d0 = line.tail<3>();
                d0.normalize();
                p0 -= d0 * (d0.dot(p0));
                Vector6 nl;
                nl.head<3>() = p0;
                nl.tail<3>() = d0;
                return nl;
            }
        }
    }
}