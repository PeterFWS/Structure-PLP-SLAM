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

#include "PLPSLAM/feature/line_extractor.h"

namespace PLPSLAM
{
    namespace feature
    {
        LineFeatureTracker::LineFeatureTracker(camera::base *camera)
            : _camera(camera), _frame_cnt(0)
        {
            // this idea from PL-VINS, we only use the line extracted from original image resolution
            _num_levels = 1;
            _scale_factor = 2;

            initialize();
        }

        void LineFeatureTracker::readIntrinsicParameter(const cv::Mat &img)
        {
            auto camera = static_cast<camera::perspective *>(_camera);

            // FW: Get _undist_map1, _undist_map2, _K
            float fx = camera->fx_;
            float fy = camera->fy_;
            float cx = camera->cx_;
            float cy = camera->cy_;
            // cv::Size imageSize = cv::Size(img.cols, img.rows); // (width, height)
            cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F); // rotation matrix, default as Identity matrix

            cv::Mat mapX = cv::Mat::zeros(img.rows, img.cols, CV_32F); // (height, width)
            cv::Mat mapY = cv::Mat::zeros(img.rows, img.cols, CV_32F);

            Eigen::Matrix3d R, R_inv;
            cv::cv2eigen(rmat, R);
            R_inv = R.inverse();

            // assume no skew
            Eigen::Matrix3d K_rect;
            K_rect << fx, 0, cx,
                0, fy, cy,
                0, 0, 1;

            Eigen::Matrix3d K_rect_inv = K_rect.inverse();
            for (int v = 0; v < img.rows; ++v)
            {
                for (int u = 0; u < img.cols; ++u)
                {
                    Eigen::Vector3d xo;
                    xo << u, v, 1;

                    Eigen::Vector3d uo = R_inv * K_rect_inv * xo;
                    Eigen::Vector2d p;
                    const auto z_inv = 1.0 / uo(2);
                    p(0) = fx * uo(0) * z_inv + cx;
                    p(1) = fy * uo(1) * z_inv + cy;

                    mapX.at<float>(v, u) = p(0);
                    mapY.at<float>(v, u) = p(1);
                }
            }

            cv::convertMaps(mapX, mapY, _undist_map1, _undist_map2, CV_32FC1, false);
            cv::eigen2cv(K_rect, _K);
        }

        void LineFeatureTracker::extract_LSD_LBD(const cv::Mat &img,
                                                 std::vector<cv::line_descriptor::KeyLine> &frame_keylsd,
                                                 cv::Mat &frame_lbd_descr,
                                                 std::vector<Vec3_t> &keyline_functions)
        {
            readIntrinsicParameter(img);
            cv::Mat img_temp;
            _frame_cnt++;

            // FW: https://docs.opencv.org/3.4/d1/da0/tutorial_remap.html
            //  src: Source image
            //  dst: Destination image of same size as src
            //  map_x: The mapping function in the x direction. It is equivalent to the first component of h(i,j)
            //  map_y: Same as above, but in y direction. Note that map_y and map_x are both of the same size as src
            //  INTER_LINEAR: The type of interpolation to use for non-integer pixels. This is by default.
            cv::remap(img, img_temp, _undist_map1, _undist_map2, CV_INTER_LINEAR);

            if (EQUALIZE) // Histogram equalization
            {
                cv::Ptr<cv::CLAHE> clache = cv::createCLAHE(3.0, cv::Size(8, 8));
                clache->apply(img_temp, img_temp);
            }

            // [1] initialize the LSD detector with customized parameters
            cv::Ptr<cv::line_descriptor::LSDDetectorC> lsd_detector = cv::line_descriptor::LSDDetectorC::createLSDDetectorC();
            cv::line_descriptor::LSDDetectorC::LSDOptions opts;
            opts.refine = 1;        // 1     The way found lines will be refined
            opts.scale = 0.5;       // 0.8   The scale of the image that will be used to find the lines. Range (0..1].
            opts.sigma_scale = 0.6; // 0.6   Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
            opts.quant = 2.0;       // 2.0   Bound to the quantization error on the gradient norm
            opts.ang_th = 22.5;     // 22.5	 Gradient angle tolerance in degrees
            opts.log_eps = 1.0;     // 0	 Detection threshold: -log10(NFA) > log_eps. Used only when advance refinement is chosen
            opts.density_th = 0.6;  // 0.7	 Minimal density of aligned region points in the enclosing rectangle.
            opts.n_bins = 1024;     // 1024  Number of bins in pseudo-ordering of gradient modulus.
            opts.min_length = _min_line_length * (std::min(img_temp.cols, img_temp.rows));

            // detect line segments in the image
            std::vector<cv::line_descriptor::KeyLine> lsd, keylsd;
            lsd_detector->detect(img_temp, lsd, _scale_factor, _num_levels, opts);

            // [2] initialize the LBD descriptor
            cv::Mat lbd_descr, keylbd_descr;
            cv::Ptr<cv::line_descriptor::BinaryDescriptor> binary_descriptor = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
            binary_descriptor->compute(img_temp, lsd, lbd_descr); // compute descriptor

            // aggregate lines from original image resolution (octave == 0) saved as keylsd
            for (unsigned int i = 0; i < lsd.size(); i++)
            {
                if (lsd[i].octave == 0 && lsd[i].lineLength >= 60)
                {
                    keylsd.push_back(lsd[i]);
                    keylbd_descr.push_back(lbd_descr.row(i));
                }
            }

            frame_keylsd = keylsd;
            frame_lbd_descr = keylbd_descr;

            // [3] calculate the parameters (functions) of the 2D line segments extracted in the image, later on used for triangulation
            for (std::vector<cv::line_descriptor::KeyLine>::iterator it = frame_keylsd.begin(); it != frame_keylsd.end(); ++it)
            {
                Vec3_t sp_l;
                sp_l << it->startPointX, it->startPointY, 1.0; // (x, y, 1) starting point

                Vec3_t ep_l;
                ep_l << it->endPointX, it->endPointY, 1.0; // (x, y, 1) ending point

                Vec3_t lineFunction;              // line function in 2D image, e.g. ax + by + c = 0
                lineFunction << sp_l.cross(ep_l); // cross product gives the normal vector
                lineFunction = lineFunction / sqrt(lineFunction(0) * lineFunction(0) + lineFunction(1) * lineFunction(1));
                keyline_functions.push_back(lineFunction);
            }
        }

        void LineFeatureTracker::initialize()
        {
            // calculate scale_factors at each level of image pyramid
            std::vector<float> scale_factors(_num_levels, 1.0);
            if (_num_levels > 1)
            {
                for (unsigned int level = 1; level < _num_levels; ++level)
                {
                    scale_factors.at(level) = _scale_factor * scale_factors.at(level - 1);
                }
                _scale_factors = scale_factors;
            }
            else
            {
                _scale_factors = scale_factors;
            }

            // calculate inv_scale_factors
            std::vector<float> inv_scale_factors(_num_levels, 1.0);
            if (_num_levels > 1)
            {
                for (unsigned int level = 1; level < _num_levels; ++level)
                {
                    inv_scale_factors.at(level) = (1.0f / _scale_factor) * inv_scale_factors.at(level - 1);
                }
                _inv_scale_factors = inv_scale_factors;
            }
            else
            {
                _inv_scale_factors = inv_scale_factors;
            }

            // calculate level_sigma_sq
            std::vector<float> level_sigma_sq(_num_levels, 1.0);
            float scale_factor_at_level = 1.0;
            if (_num_levels > 1)
            {
                for (unsigned int level = 1; level < _num_levels; ++level)
                {
                    scale_factor_at_level = _scale_factor * scale_factor_at_level;
                    level_sigma_sq.at(level) = scale_factor_at_level * scale_factor_at_level;
                }
                _level_sigma_sq = level_sigma_sq;
            }
            else
            {
                _level_sigma_sq = level_sigma_sq;
            }

            // calculate inv_level_sigma_sq
            std::vector<float> inv_level_sigma_sq(_num_levels, 1.0);
            scale_factor_at_level = 1.0;
            if (_num_levels > 1)
            {
                for (unsigned int level = 1; level < _num_levels; ++level)
                {
                    scale_factor_at_level = _scale_factor * scale_factor_at_level;
                    inv_level_sigma_sq.at(level) = 1.0f / (scale_factor_at_level * scale_factor_at_level);
                }
                _inv_level_sigma_sq = inv_level_sigma_sq;
            }
            else
            {
                _inv_level_sigma_sq = inv_level_sigma_sq;
            }
        }

        unsigned int LineFeatureTracker::get_num_scale_levels() const
        {
            return _num_levels;
        }

        float LineFeatureTracker::get_scale_factor() const
        {
            return _scale_factor;
        }

        std::vector<float> LineFeatureTracker::get_scale_factors() const
        {
            return _scale_factors;
        }

        std::vector<float> LineFeatureTracker::get_inv_scale_factors() const
        {
            return _inv_scale_factors;
        }

        std::vector<float> LineFeatureTracker::get_level_sigma_sq() const
        {
            return _level_sigma_sq;
        }

        std::vector<float> LineFeatureTracker::get_inv_level_sigma_sq() const
        {
            return _inv_level_sigma_sq;
        }
    }
}