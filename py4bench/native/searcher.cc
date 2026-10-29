// Copyright 2026-present the zvec project. Licensed under Apache-2.0.
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <exception>
#include <filesystem>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <zvec/db/collection.h>
#include <zvec/db/doc_iterator.h>
#include <zvec/core/interface/index_factory.h>

namespace zvec {
namespace {
namespace py = pybind11;

template <class T> T take(Result<T> result) {
  if (!result) throw std::runtime_error(result.error().message());
  return std::move(result.value());
}

// Clone the polymorphic configuration once: subsequent caller mutation must
// neither change the search nor race with a search running without the GIL.
QueryParams::Ptr clone_params(const QueryParams::Ptr &p) {
  if (!p) return nullptr;
#define COPY_PARAM(T) \
  if (auto value = dynamic_cast<const T *>(p.get())) \
    return std::make_shared<T>(*value)
  COPY_PARAM(HnswQueryParams);
  COPY_PARAM(VamanaQueryParams);
  COPY_PARAM(FlatQueryParams);
  COPY_PARAM(IVFQueryParams);
  COPY_PARAM(HnswRabitqQueryParams);
  COPY_PARAM(IvfRabitqQueryParams);
  COPY_PARAM(DiskAnnQueryParams);
#undef COPY_PARAM
  throw std::invalid_argument("py4bench: unsupported query parameter type");
}

template <class T> py::array_t<T> own(std::vector<T> values) {
  auto buffer = std::make_unique<std::vector<T>>(std::move(values));
  auto *ptr = buffer.get();
  py::capsule owner(ptr, [](void *p) { delete static_cast<std::vector<T> *>(p); });
  buffer.release();
  return py::array_t<T>({static_cast<py::ssize_t>(ptr->size())},
                        {sizeof(T)}, ptr->data(), owner);
}

class Searcher {
 public:
  Searcher(const std::string &path, const std::string &index_json,
           const std::string &query_json) {
    py::gil_scoped_release release;
    auto param = core_interface::IndexFactory::DeserializeIndexParamFromJson(index_json);
    if (!param || param->index_type != core_interface::IndexType::kHNSW ||
        param->data_type != core_interface::DataType::DT_FP32)
      throw std::invalid_argument("core search requires kHNSW / DT_FP32 index parameters");
    dimension_ = param->dimension;
    core_query_ = core_interface::IndexFactory::QueryParamDeserializeFromJson<
        core_interface::HNSWQueryParam>(query_json);
    if (!core_query_) throw std::invalid_argument("invalid core HNSW query JSON");
    core_ = core_interface::IndexFactory::CreateAndInitIndex(*param);
    if (!core_) throw std::invalid_argument("cannot initialize core index");
    core_interface::StorageOptions options;
    options.type = core_interface::StorageOptions::StorageType::kMMAP;
    options.read_only = true;
    if (core_->open(path, options) != 0) {
      core_->close();
      throw std::runtime_error("cannot open core index (check path, parameters and ISA support)");
    }
  }

  ~Searcher() { if (core_) core_->close(); }

  Searcher(Collection::Ptr collection, std::string field,
           QueryParams::Ptr params, std::string filter)
      : collection_(std::move(collection)) {
    if (!collection_) throw std::invalid_argument("collection is required");
    query_.target_.field_name_ = std::move(field);
    query_.target_.query_params_ = clone_params(params);
    query_.filter_ = std::move(filter);
    py::gil_scoped_release release;
    if (!take(collection_->options()).read_only_)
      throw std::invalid_argument("py4bench requires a read-only collection");
    auto schema = take(collection_->schema());
    const auto *f = schema.get_vector_field(query_.target_.field_name_);
    if (!f || f->data_type() != DataType::VECTOR_FP32)
      throw std::invalid_argument("py4bench requires a dense FP32 vector field");
    dimension_ = f->dimension();

    // Numeric primary keys are benchmark dataset IDs, never insertion offsets.
    IteratorOptions options;
    options.include_vector_ = false;
    options.output_fields_ = std::vector<std::string>{};
    auto iterator = take(collection_->create_iterator(options));
    std::unordered_set<int64_t> unique;
    uint64_t maximum = 0;
    while (auto doc = take(iterator->next())) {
      const auto &pk = doc->pk_ref();
      int64_t id = -1;
      auto parsed = std::from_chars(pk.data(), pk.data() + pk.size(), id);
      if (parsed.ec != std::errc{} || parsed.ptr != pk.data() + pk.size() || id < 0)
        throw std::invalid_argument("py4bench requires non-negative int64 primary keys");
      if (!unique.insert(id).second)
        throw std::invalid_argument("primary keys must map to unique integer IDs");
      sparse_ids_.emplace(doc->doc_id(), id);
      maximum = std::max(maximum, doc->doc_id());
    }
    iterator->close();
    if (!sparse_ids_.empty() && maximum / 4 <= sparse_ids_.size()) {
      dense_ids_.assign(maximum + 1, -1);
      for (const auto &item : sparse_ids_) dense_ids_[item.first] = item.second;
      sparse_ids_.clear();
      sparse_ids_.rehash(0);
    }
  }

  void validate(const py::array &q, int ndim, int k) const {
    if (k <= 0) throw std::invalid_argument("k must be positive");
    if (!q.dtype().is(py::dtype::of<float>()) || q.ndim() != ndim ||
        q.shape(ndim - 1) != dimension_ || !(q.flags() & py::array::c_style) ||
        reinterpret_cast<uintptr_t>(q.data()) % alignof(float))
      throw std::invalid_argument("expected aligned C-contiguous float32 array with matching dimension");
  }

  std::vector<int64_t> run(const float *data, int k) const {
    if (core_) {
      auto param = core_query_->clone();
      param->topk = k;
      std::vector<int64_t> ids(k, -1);
      const int ret = core_->search_internal_ids(
          {core_interface::DenseVector{data}}, param, ids.data(), nullptr);
      if (ret != 0) throw std::runtime_error("core search failed: " + std::to_string(ret));
      ids.erase(std::remove(ids.begin(), ids.end(), -1), ids.end());
      return ids;
    }
    // Each invocation owns its query and result; concurrent callers share only
    // immutable configuration and mapping. The collection manages search scratch.
    auto query = query_;
    query.topk_ = k;
    query.target_.clause_ = VectorViewClause{
        std::string_view(reinterpret_cast<const char *>(data), dimension_ * sizeof(float)), {}, {}};
    auto result = take(collection_->query_internal_ids(query, false));
    size_t count = 0;
    for (auto internal : result.ids) {
      if (internal == -1) continue;
      int64_t external = -1;
      if (internal >= 0 && static_cast<uint64_t>(internal) < dense_ids_.size()) {
        external = dense_ids_[internal];
      } else {
        auto found = sparse_ids_.find(internal);
        if (found != sparse_ids_.end()) external = found->second;
      }
      if (external < 0) throw std::runtime_error("unmapped internal ID in collection snapshot");
      result.ids[count++] = external;
    }
    result.ids.resize(count);
    return std::move(result.ids);
  }

  py::object search(const py::array &q, int k, bool as_list) const {
    validate(q, 1, k);
    const auto *data = static_cast<const float *>(q.data());
    std::vector<int64_t> ids;
    {
      py::gil_scoped_release release;
      ids = run(data, k);
    }
    if (as_list) return py::cast(std::move(ids));
    return own(std::move(ids));
  }

  py::list batch(const py::array &q, int k, int threads) const {
    validate(q, 2, k);
    if (threads <= 0) throw std::invalid_argument("threads must be positive");
    const auto *data = static_cast<const float *>(q.data());
    const size_t rows = q.shape(0);
    std::vector<std::vector<int64_t>> output(rows);
    {
      py::gil_scoped_release release;
      if (threads == 1 || rows < 2) {
        for (size_t i = 0; i < rows; ++i) output[i] = run(data + i * dimension_, k);
      } else {
        std::atomic<size_t> next{0};
        std::exception_ptr error;
        std::mutex error_mutex;
        auto worker = [&]() {
          try {
            for (size_t i; (i = next.fetch_add(1)) < rows;)
              output[i] = run(data + i * dimension_, k);
          } catch (...) {
            next.store(rows);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) error = std::current_exception();
          }
        };
        std::vector<std::thread> workers;
        try {
          for (size_t i = 0; i < std::min(rows, static_cast<size_t>(threads)); ++i)
            workers.emplace_back(worker);
        } catch (...) {
          next.store(rows);
          for (auto &t : workers) t.join();
          throw;
        }
        for (auto &t : workers) t.join();
        if (error) std::rethrow_exception(error);
      }
    }
    py::list result;
    for (auto &ids : output) result.append(own(std::move(ids)));
    return result;
  }

  double time_native(const py::array &q, int k) const {
    validate(q, 2, k);
    const auto *data = static_cast<const float *>(q.data());
    const size_t rows = q.shape(0);
    py::gil_scoped_release release;
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < rows; ++i) {
      // Same run() and result allocation as search(), excluding only Python
      // dispatch/materialization. This is a bridge baseline, not a graph kernel.
      auto ids = run(data + i * dimension_, k);
      if (ids.size() > static_cast<size_t>(k))
        throw std::runtime_error("unexpected result count");
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }

 private:
  Collection::Ptr collection_;
  SearchQuery query_;
  int dimension_;
  std::vector<int64_t> dense_ids_;
  std::unordered_map<uint64_t, int64_t> sparse_ids_;
  core_interface::Index::Pointer core_;
  core_interface::BaseIndexQueryParam::Pointer core_query_;
};

void build_core(const std::string &path, const py::array &vectors,
                const std::string &index_json) {
  auto param = core_interface::IndexFactory::DeserializeIndexParamFromJson(index_json);
  if (!param || param->index_type != core_interface::IndexType::kHNSW ||
      param->data_type != core_interface::DataType::DT_FP32)
    throw std::invalid_argument("core build requires kHNSW / DT_FP32 parameters");
  if (!vectors.dtype().is(py::dtype::of<float>()) || vectors.ndim() != 2 ||
      vectors.shape(1) != param->dimension || !(vectors.flags() & py::array::c_style) ||
      reinterpret_cast<uintptr_t>(vectors.data()) % alignof(float) ||
      static_cast<uint64_t>(vectors.shape(0)) > UINT32_MAX)
    throw std::invalid_argument("expected aligned contiguous FP32 matrix matching dimension");
  auto &hnsw = dynamic_cast<core_interface::HNSWIndexParam &>(*param);
  // QG is a read-only projection of the built full-precision graph.
  hnsw.use_quantized_graph = false;
  hnsw.quantized_graph_model.clear();
  const auto *data = static_cast<const float *>(vectors.data());
  const size_t rows = vectors.shape(0);
  py::gil_scoped_release release;
  if (std::filesystem::exists(path))
    throw std::invalid_argument("refusing to overwrite an existing core index");
  auto index = core_interface::IndexFactory::CreateAndInitIndex(*param);
  if (!index) throw std::invalid_argument("cannot initialize core builder");
  auto check = [](int rc) {
    if (rc) throw std::runtime_error("core build failed: " + std::to_string(rc));
  };
  core_interface::StorageOptions options;
  options.type = core_interface::StorageOptions::StorageType::kMMAP;
  options.create_new = true;
  try {
    check(index->open(path, options));
    for (uint32_t i = 0; i < rows; ++i)
      check(index->add({core_interface::DenseVector{data + i * param->dimension}}, i));
    check(index->flush());
    check(index->close());
  } catch (...) {
    index->close();
    throw;
  }
}
}  // namespace

void BindPy4Bench(pybind11::module_ &m) {
  py::class_<Searcher>(m, "_BenchSearcher")
      .def(py::init<const std::string &, const std::string &, const std::string &>(),
           py::arg("path"), py::arg("index_json"), py::arg("query_json"))
      .def(py::init<Collection::Ptr, std::string, QueryParams::Ptr, std::string>(),
           py::arg("collection"), py::arg("field"), py::arg("params") = nullptr,
           py::arg("filter") = "")
      .def("search", [](const Searcher &s, const py::array &q, int k) {
        return s.search(q, k, false);
      }, py::arg("query").noconvert(), py::arg("k") = 10)
      .def("search_list", [](const Searcher &s, const py::array &q, int k) {
        return s.search(q, k, true);
      }, py::arg("query").noconvert(), py::arg("k") = 10)
      .def("search_batch", &Searcher::batch, py::arg("queries").noconvert(),
           py::arg("k") = 10, py::arg("threads") = 1)
      .def("time_native", &Searcher::time_native, py::arg("queries").noconvert(),
           py::arg("k") = 10);
  m.def("_BuildBenchCoreIndex", &build_core, py::arg("path"),
        py::arg("vectors").noconvert(), py::arg("index_json"));
}
}  // namespace zvec
