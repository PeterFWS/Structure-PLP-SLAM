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

#include "PLPSLAM/util/random_array.h"

#include <random>
#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>

namespace PLPSLAM
{
    namespace util
    {

        std::mt19937 create_random_engine()
        {
            std::random_device random_device;
            std::vector<std::uint_least32_t> v(10);
            std::generate(v.begin(), v.end(), std::ref(random_device));
            std::seed_seq seed(v.begin(), v.end());
            return std::mt19937(seed);
        }

        template <typename T>
        std::vector<T> create_random_array(const size_t size, const T rand_min, const T rand_max)
        {
            assert(rand_min <= rand_max);
            assert(size <= static_cast<size_t>(rand_max - rand_min + 1));

            // メルセンヌ・ツイスタ作成
            auto random_engine = create_random_engine();
            std::uniform_int_distribution<T> uniform_int_distribution(rand_min, rand_max);

            // sizeより少し大きくランダム数列(重複あり)を作成する
            const auto make_size = static_cast<size_t>(size * 1.2);

            // vがsizeになるまで繰り返す
            std::vector<T> v;
            v.reserve(size);
            while (v.size() != size)
            {
                // ランダム整数列を順に追加(重複がある可能性がある)
                while (v.size() < make_size)
                {
                    v.push_back(uniform_int_distribution(random_engine));
                }

                // ソートして重複を除く -> 重複が除かれた数列の末尾のイテレータがunique_endに入る
                std::sort(v.begin(), v.end());
                auto unique_end = std::unique(v.begin(), v.end());

                // vのサイズが大きすぎたら，sizeまでのイテレータに変えておく
                if (size < static_cast<size_t>(std::distance(v.begin(), unique_end)))
                {
                    unique_end = std::next(v.begin(), size);
                }

                // 重複部分から最後までを削除する
                v.erase(unique_end, v.end());
            }

            // 昇順になっているのでシャッフル
            std::shuffle(v.begin(), v.end(), random_engine);

            return v;
        }

        // 明示的に実体化しておく
        template std::vector<int> create_random_array(size_t, int, int);
        template std::vector<unsigned int> create_random_array(size_t, unsigned int, unsigned int);

    } // namespace util
} // namespace PLPSLAM
