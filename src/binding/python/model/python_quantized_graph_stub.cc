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
#include <stdexcept>
#include <pybind11/pybind11.h>
#include "algorithm/quantized_graph/quantized_graph_support.h"

namespace py = pybind11;
namespace zvec {
namespace {
struct PythonQuantizedGraph {};
}

void RegisterQuantizedGraph(py::module_ &m) {
  auto graph = py::class_<PythonQuantizedGraph>(
      m, "QuantizedGraph",
      "Standalone FP32 graph with input row ordinals as IDs.");
  graph.def_static("supported", &core::QuantizedGraphSupported);
  auto unavailable = [](py::args, py::kwargs) {
    throw std::runtime_error(
        "This zvec build does not support quantized graphs");
  };
  graph.def_static("open", unavailable).def_static("build", unavailable)
      .def("save", unavailable);
}

}  // namespace zvec
