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

#ifndef PLPSLAM_DATA_BOW_VOCABULARY_H
#define PLPSLAM_DATA_BOW_VOCABULARY_H

#ifdef USE_DBOW2
#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>
#else
#include <fbow/fbow.h>
#endif // USE_DBOW2

namespace PLPSLAM
{
    namespace data
    {

#ifdef USE_DBOW2

        typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> bow_vocabulary;

#else

        typedef fbow::Vocabulary bow_vocabulary;

#endif // USE_DBOW2

    } // namespace data
} // namespace PLPSLAM

#endif // PLPSLAM_DATA_BOW_VOCABULARY_H
