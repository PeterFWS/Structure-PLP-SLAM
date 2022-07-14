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

#ifndef PLPSLAM_OPTIMIZER_G2O_EXTENDED_PLANE_3D_H
#define PLPSLAM_OPTIMIZER_G2O_EXTENDED_PLANE_3D_H

#include "PLPSLAM/type.h"
#include <opencv2/core/core.hpp>

#include <g2o/config.h>
#include <g2o/stuff/misc.h>
#include <g2o/core/eigen_types.h>

namespace PLPSLAM
{
    namespace optimize
    {
        namespace g2o
        {

            class Plane3D
            {
            public:
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                // FW:
                //  Minimal Parameterization of a 3D Plane:
                //       for applying graph optimization, we need a minimal parameterization of the plane:
                //       π = (azimuth, elevation, d), or (psi, phi, d)
                //       phi and psi should be restricted between (-π, π] to avoid singularity issue

                friend Plane3D operator*(const ::g2o::Isometry3 &t, const Plane3D &plane);

                Plane3D()
                {
                    ::g2o::Vector4 v;
                    v << 1.0, 0.0, 0.0, -1;
                    fromVector(v);
                }

                Plane3D(const ::g2o::Vector4 &v)
                {
                    fromVector(v);
                }

                inline ::g2o::Vector4 toVector() const
                {
                    return _coeffs;
                }

                inline const ::g2o::Vector4 &coeffs() const
                {
                    return _coeffs;
                }

                inline void fromVector(const ::g2o::Vector4 &coeffs_)
                {
                    _coeffs = coeffs_;
                    normalize(_coeffs);
                }

                // return  psi (azimuth)
                static number_t azimuth(const ::g2o::Vector3 &v)
                {
                    return std::atan2(v(1), v(0)); // arctan(y/x)
                }

                // return phi (elevation)
                static number_t elevation(const ::g2o::Vector3 &v)
                {
                    return std::atan2(v(2), v.head<2>().norm()); // arctan(z/(sqrt(x^2 + y^2)))
                }

                // return normalized distance
                number_t distance() const
                {
                    return -_coeffs(3);
                }

                // return nomalized normal
                ::g2o::Vector3 normal() const
                {
                    return _coeffs.head<3>();
                }

                // return rotation
                static ::g2o::Matrix3 rotation(const ::g2o::Vector3 &v)
                {
                    number_t _azimuth = azimuth(v);
                    number_t _elevation = elevation(v);

                    // !fix the error: Eigen::AngleAxis() not declared in the scope
                    // return (AngleAxis(_azimuth, Vector3::UnitZ()) * AngleAxis(-_elevation, Vector3::UnitY())).toRotationMatrix();
                    Eigen::AngleAxisd azimuth_v(_azimuth, ::g2o::Vector3::UnitZ());
                    Eigen::AngleAxisd elevation_v(-_elevation, ::g2o::Vector3::UnitY());
                    return (azimuth_v * elevation_v).toRotationMatrix();
                }

                // called in plane vertex -> update function, which is the ⊞ operator in manifold
                inline void oplus(const ::g2o::Vector3 &v)
                {
                    // notice here the input v: (azimuth, elevation, d)
                    // construct a normal from azimuth and elevation;
                    // x = cos(phi)cos(psi)
                    // y = cos(phi)sin(psi)
                    // d = sin(phi)
                    number_t _azimuth = v[0];
                    number_t _elevation = v[1];
                    number_t s = std::sin(_elevation), c = std::cos(_elevation);
                    ::g2o::Vector3 n(c * std::cos(_azimuth), c * std::sin(_azimuth), s);

                    // rotate the normal
                    ::g2o::Matrix3 R = rotation(normal());
                    number_t d = distance() + v[2];
                    _coeffs.head<3>() = R * n;
                    _coeffs(3) = -d;
                    normalize(_coeffs);
                }

                inline ::g2o::Vector3 ominus(const Plane3D &plane)
                {
                    // construct the rotation that would bring the plane normal in (1 0 0)
                    ::g2o::Matrix3 R = rotation(normal()).transpose();
                    ::g2o::Vector3 n = R * plane.normal();
                    number_t d = distance() - plane.distance();
                    return ::g2o::Vector3(azimuth(n), elevation(n), d);
                }

                // // for planes are perpendicular (see SP-SLAM)
                // inline ::g2o::Vector2 ominus_ver(const Plane3D &plane)
                // {
                //     // construct the rotation that would bring the plane normal in (1 0 0)
                //     ::g2o::Vector3 v = normal().cross(plane.normal());
                //     Eigen::AngleAxisd ver(M_PI / 2, v / v.norm());
                //     ::g2o::Vector3 b = ver * normal();

                //     ::g2o::Matrix3 R = rotation(b).transpose();
                //     ::g2o::Vector3 n = R * plane.normal();
                //     return ::g2o::Vector2(azimuth(n), elevation(n));
                // }

                // // for planes are parallel (see SP-SLAM)
                // inline ::g2o::Vector2 ominus_par(const Plane3D &plane)
                // {
                //     // construct the rotation that would bring the plane normal in (1 0 0)
                //     ::g2o::Vector3 nor = normal();
                //     if (plane.normal().dot(nor) < 0)
                //         nor = -nor;
                //     ::g2o::Matrix3 R = rotation(nor).transpose();
                //     ::g2o::Vector3 n = R * plane.normal();

                //     return ::g2o::Vector2(azimuth(n), elevation(n));
                // }

            protected:
                static inline void normalize(::g2o::Vector4 &coeffs)
                {
                    number_t n = coeffs.head<3>().norm();
                    coeffs = coeffs * (1. / n);

                    // FW: why?
                    if (coeffs(3) < 0.0)
                        coeffs = -coeffs;
                }

                // normalized plane parameters
                ::g2o::Vector4 _coeffs;
            };

            // overload the mathmatical operation *
            inline Plane3D operator*(const ::g2o::Isometry3 &t, const Plane3D &plane)
            {
                ::g2o::Vector4 v = plane._coeffs;
                ::g2o::Vector4 v2;
                ::g2o::Matrix3 R = t.rotation();
                v2.head<3>() = R * v.head<3>();
                v2(3) = v(3) - t.translation().dot(v2.head<3>());
                return Plane3D(v2);
            };

        } // namespace g2o
    }     // namespace optimize
} // namespace PLPSLAM

#endif