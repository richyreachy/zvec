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

// This translation unit pulls in the RaBitQ distance estimator header,
// which contains a static factory registration.  Without a .cc file, the
// header-only registration would never be compiled into the turbo library.

#include "rabit_distance_estimator.h"

namespace zvec {
namespace turbo {

// Ensure the static registration variable is referenced so it is not
// stripped by the linker.
bool rabit_distance_estimator_registered() {
  return s_rabit_registered;
}

}  // namespace turbo
}  // namespace zvec
