#ifndef PLPSLAM_TEST_HELPER_LANDMARK_H
#define PLPSLAM_TEST_HELPER_LANDMARK_H

#include "PLPSLAM/type.h"

#include <random>

using namespace PLPSLAM;

eigen_alloc_vector<Vec3_t> create_random_landmarks_in_space(const unsigned int num_landmarks,
                                                            const float space_lim);

eigen_alloc_vector<Vec3_t> create_random_landmarks_on_plane(const unsigned int num_landmarks,
                                                            const float space_lim, const Vec4_t &plane_coeffs);

#endif // PLPSLAM_TEST_HELPER_LANDMARK_H
