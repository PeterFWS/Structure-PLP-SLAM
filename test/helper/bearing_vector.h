#ifndef PLPSLAM_TEST_HELPER_BEARING_VECTOR_H
#define PLPSLAM_TEST_HELPER_BEARING_VECTOR_H

#include "PLPSLAM/type.h"

using namespace PLPSLAM;

void create_bearing_vectors(const Mat33_t &rot_cw, const Vec3_t &trans_cw, const eigen_alloc_vector<Vec3_t> &landmarks,
                            eigen_alloc_vector<Vec3_t> &bearings, const double noise_stddev = 0.0);

#endif // PLPSLAM_TEST_HELPER_BEARING_VECTOR_H
