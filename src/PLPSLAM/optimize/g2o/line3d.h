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

#ifndef PLPSLAM_OPTIMIZER_G2O_LINE3D_H
#define PLPSLAM_OPTIMIZER_G2O_LINE3D_H

#include "PLPSLAM/type.h"
#include <g2o/core/eigen_types.h>

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {
            // FW:
            //  (This file re-written from g2o library)

            typedef Eigen::Matrix<number_t, 6, 1> Vector6;
            typedef Eigen::Matrix<number_t, 6, 6> Matrix6;
            typedef Eigen::Matrix<number_t, 6, 4> Matrix6x4;

            struct OrthonormalLine3D
            {
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
                ::g2o::Matrix2 W;
                ::g2o::Matrix3 U;

                OrthonormalLine3D()
                {
                    W = ::g2o::Matrix2::Identity();
                    U = ::g2o::Matrix3::Identity();
                }
            };
            typedef struct OrthonormalLine3D OrthonormalLine3D;

            class Line3D : public Vector6
            {
            public:
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

                friend Line3D operator*(const ::g2o::Isometry3 &t, const Line3D &line);

                // 3D Line is parameterized using Plücker coordinates (6DOF)
                // - head(3) should be the "moment" of the line
                // - tail(3) should be the direction of the line
                // the Plücker coordinates (6DOF) will be converted as Orthonormal representation (4DOF) during optimization
                Line3D()
                {
                    *this << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
                }

                Line3D(const Vector6 &v)
                {
                    (Vector6 &)*this = v;
                }

                inline ::g2o::Vector3 w() const
                {
                    return head<3>();
                }

                inline ::g2o::Vector3 d() const
                {
                    return tail<3>();
                }

                inline void setW(const ::g2o::Vector3 &w_)
                {
                    head<3>() = w_;
                }

                inline void setD(const ::g2o::Vector3 &d_)
                {
                    tail<3>() = d_;
                }

                static inline Line3D fromCartesian(const Vector6 &cart)
                {
                    Line3D l;
                    ::g2o::Vector3 _p = cart.head<3>();
                    ::g2o::Vector3 _d = cart.tail<3>() * 1.0 / cart.tail<3>().norm();
                    _p -= _d * (_d.dot(_p));
                    l.setW(_p.cross(_p + _d));
                    l.setD(_d);

                    return l;
                }

                Vector6 toCartesian() const;

                // get Plücker coordinates from Orthonormal representation
                // L = (w_0 * u0, w_1 * u1)
                //  u0 - 0th column of matrix U
                //  u1 - 1th column of matrix U
                static inline Line3D fromOrthonormal(const OrthonormalLine3D &ortho)
                {
                    ::g2o::Vector3 w;
                    w.x() = ortho.U(0, 0) * ortho.W(0, 0);
                    w.y() = ortho.U(1, 0) * ortho.W(0, 0);
                    w.z() = ortho.U(2, 0) * ortho.W(0, 0);

                    ::g2o::Vector3 d;
                    d.x() = ortho.U(0, 1) * ortho.W(1, 0);
                    d.y() = ortho.U(1, 1) * ortho.W(1, 0);
                    d.z() = ortho.U(2, 1) * ortho.W(1, 0);

                    Line3D l;
                    l.setW(w);
                    l.setD(d);
                    l.normalize();

                    return l;
                }

                // get Orthonormal representation from Plücker coordinates
                static inline OrthonormalLine3D toOrthonormal(const Line3D &line)
                {
                    OrthonormalLine3D ortho;

                    ::g2o::Vector2 mags;
                    mags << line.d().norm(), line.w().norm();

                    number_t wn = 1.0 / mags.norm();
                    ortho.W << mags.y() * wn, -mags.x() * wn, mags.x() * wn, mags.y() * wn;

                    number_t mn = 1.0 / mags.y();
                    number_t dn = 1.0 / mags.x();
                    ::g2o::Vector3 mdcross;
                    mdcross = line.w().cross(line.d());
                    number_t mdcrossn = 1.0 / mdcross.norm();
                    ortho.U << line.w().x() * mn, line.d().x() * dn, mdcross.x() * mdcrossn,
                        line.w().y() * mn, line.d().y() * dn, mdcross.y() * mdcrossn,
                        line.w().z() * mn, line.d().z() * dn, mdcross.z() * mdcrossn;

                    return ortho;
                }

                inline void normalize()
                {
                    number_t n = 1.0 / d().norm();
                    (*this) *= n;
                }

                inline Line3D normalized() const
                {
                    return Line3D((Vector6)(*this) * (1.0 / d().norm()));
                }

                // operation "+" for Orthonormal representation
                inline void oplus(const ::g2o::Vector4 &v)
                {
                    OrthonormalLine3D ortho_estimate = toOrthonormal(*this);
                    OrthonormalLine3D ortho_update;
                    ortho_update.W << std::cos(v[3]), -std::sin(v[3]), std::sin(v[3]),
                        std::cos(v[3]);
                    ::g2o::Quaternion quat(std::sqrt(1 - v.head<3>().squaredNorm()), v[0], v[1], v[2]);
                    quat.normalize();
                    ortho_update.U = quat.toRotationMatrix();

                    ortho_estimate.U = ortho_estimate.U * ortho_update.U;
                    ortho_estimate.W = ortho_estimate.W * ortho_update.W;

                    *this = fromOrthonormal(ortho_estimate);
                    this->normalize();
                }

                // operation "-" for Orthonormal representation
                inline ::g2o::Vector4 ominus(const Line3D &line)
                {
                    OrthonormalLine3D ortho_estimate = toOrthonormal(*this);
                    OrthonormalLine3D ortho_line = toOrthonormal(line);

                    ::g2o::Matrix2 W_delta = ortho_estimate.W.transpose() * ortho_line.W;
                    ::g2o::Matrix3 U_delta = ortho_estimate.U.transpose() * ortho_line.U;

                    ::g2o::Vector4 delta;
                    ::g2o::Quaternion q(U_delta);
                    q.normalize();
                    delta[0] = q.x();
                    delta[1] = q.y();
                    delta[2] = q.z();
                    delta[3] = std::atan2(W_delta(1, 0), W_delta(0, 0));

                    return delta;
                }
            };

            Line3D operator*(const ::g2o::Isometry3 &t, const Line3D &line);

            Vector6 transformCartesianLine(const ::g2o::Isometry3 &t, const Vector6 &line);

            Vector6 normalizeCartesianLine(const Vector6 &line);

            static inline number_t mline_elevation(const number_t v[3])
            {
                return std::atan2(v[2], sqrt(v[0] * v[0] + v[1] * v[1]));
            }

            inline number_t getAzimuth(const ::g2o::Vector3 &direction)
            {
                return std::atan2(direction.y(), direction.x());
            }

            inline number_t getElevation(const ::g2o::Vector3 &direction)
            {
                return std::atan2(direction.z(), direction.head<2>().norm());
            }

        }
    }
}
#endif