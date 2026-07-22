// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "diskann_distance_estimator.h"

namespace zvec {
namespace core {

std::map<std::string, DiskAnnDistanceEstimator::Factory> &
DiskAnnDistanceEstimator::registry() {
  static std::map<std::string, Factory> r;
  return r;
}

void DiskAnnDistanceEstimator::register_factory(const std::string &name,
                                                Factory factory) {
  registry()[name] = std::move(factory);
}

DiskAnnDistanceEstimator::Pointer DiskAnnDistanceEstimator::create(
    const std::string &name) {
  auto it = registry().find(name);
  if (it == registry().end()) {
    return nullptr;
  }
  return it->second();
}

bool DiskAnnDistanceEstimator::has_factory(const std::string &name) {
  return registry().find(name) != registry().end();
}

}  // namespace core
}  // namespace zvec
