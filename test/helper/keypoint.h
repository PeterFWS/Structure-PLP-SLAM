#ifndef PLPSLAM_TEST_HELPER_KEYPOINT_H
#define PLPSLAM_TEST_HELPER_KEYPOINT_H

#include "PLPSLAM/type.h"

using namespace PLPSLAM;

void create_keypoints(const Mat33_t &rot_cw, const Vec3_t &trans_cw, const Mat33_t &cam_matrix, const eigen_alloc_vector<Vec3_t> &landmarks,
                      std::vector<cv::Point2f> &keypts, const double noise_stddev = 0.0);

void create_keypoints(const Mat33_t &rot_cw, const Vec3_t &trans_cw, const Mat33_t &cam_matrix, const eigen_alloc_vector<Vec3_t> &landmarks,
                      std::vector<cv::KeyPoint> &keypts, const double noise_stddev = 0.0);

#endif // PLPSLAM_TEST_HELPER_KEYPOINT_H
