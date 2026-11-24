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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <zvec/core/framework/index_factory.h>
#include "algorithm/hnsw/hnsw_streamer_entity.h"
#include "algorithm/quantized_graph/quantized_graph_index.h"
#include "algorithm/quantized_graph/quantized_graph_query.h"
#include "bench/qg_build_options.h"
#include "bench_result.h"
#include "helper.h"
#include <yaml-cpp/yaml.h>

static bool g_debug_mode = 0;

//------------------------------------------------------------
// Bench
//------------------------------------------------------------
enum RetrievalMode { RM_UNDEFINED = 0, RM_DENSE = 1, RM_SPARSE = 2 };

enum FilterMode { FM_UNDEFINED = 0, FM_NONE = 1, FM_TAG = 2 };

template <typename T>
class Bench {
 public:
  Bench(size_t threads, size_t bench_secs, size_t batch_count,
        RetrievalMode &retrieval_mode, FilterMode filter_mode)
      : threads_(threads),
        bench_secs_(bench_secs),
        batch_count_(batch_count),
        retrieval_mode_{retrieval_mode},
        filter_mode_{filter_mode} {
    if (threads_ == 0) {
      pool_ = make_shared<ThreadPool>();
      threads_ = pool_->count();
      cout << "Using cpu count as thread pool count[" << threads_ << "]"
           << endl;
    } else {
      pool_ = make_shared<ThreadPool>(threads_, false);
      threads_ = pool_->count();
      cout << "Using thread pool count[" << threads_ << "]" << endl;
    }
    if (batch_count_ < 1) {
      batch_count_ = 1;
    }
  }

  static void stop(int signo) {
    if (stop_now) {
      exit(signo);
    }
    stop_now = true;
    cout << "\rTrying to stop. press [Ctrl+C] again kill immediately." << endl
         << flush;
  }

  bool load_query(const std::string &query_file, const std::string &first_sep,
                  const std::string &second_sep) {
    TxtInputReader<T> reader;
    vector<vector<T>> queries;
    vector<SparseData<T>> sparse_data;
    vector<vector<uint64_t>> taglists;

    if (!reader.load_query(query_file, first_sep, second_sep, queries,
                           sparse_data, taglists)) {
      LOG_ERROR("Load query error");
      return false;
    }

    if (batch_count_ == 1) {
      batch_queries_ = queries;

      for (size_t i = 0; i < sparse_data.size(); ++i) {
        vector<uint32_t> sparse_count;
        sparse_count.push_back(sparse_data[i].count);

        batch_sparse_counts_.push_back(sparse_count);
        batch_sparse_indices_.push_back(sparse_data[i].indices);
        batch_sparse_features_.push_back(sparse_data[i].features);
      }

      for (size_t i = 0; i < taglists.size(); ++i) {
        vector<vector<uint64_t>> new_taglists;
        new_taglists.push_back(taglists[i]);

        batch_taglists_.push_back(std::move(new_taglists));
      }
    } else {
      size_t num_batch = (queries.size() + batch_count_ - 1) / batch_count_;
      size_t idx = 0;
      for (size_t n = 0; n < num_batch; ++n) {
        vector<T> batch_query;
        vector<uint32_t> batch_sparse_count;
        vector<uint32_t> batch_sparse_indices;
        vector<T> batch_sparse_feature;
        vector<vector<uint64_t>> batch_taglists;

        for (size_t i = 0; i < batch_count_; ++i) {
          for (size_t k = 0; k < queries[idx].size(); ++k) {
            batch_query.push_back(queries[idx][k]);
          }

          batch_sparse_count.push_back(sparse_data[idx].count);

          for (size_t k = 0; k < sparse_data[idx].indices.size(); ++k) {
            batch_sparse_indices.push_back(sparse_data[idx].indices[k]);
          }

          for (size_t k = 0; k < sparse_data[idx].features.size(); ++k) {
            batch_sparse_feature.push_back(sparse_data[idx].features[k]);
          }

          if (taglists.size() > idx) {
            batch_taglists.push_back(taglists[idx]);
          }

          idx = (idx + 1) % queries.size();
        }

        batch_queries_.push_back(batch_query);
        batch_sparse_counts_.push_back(batch_sparse_count);
        batch_sparse_indices_.push_back(batch_sparse_indices);
        batch_sparse_features_.push_back(batch_sparse_feature);
        batch_taglists_.push_back(batch_taglists);
      }
    }

    dim_ = queries[0].size();
    if (typeid(T) == typeid(float)) {
      qmeta_.set_meta(IndexMeta::DataType::DT_FP32, dim_);
    } else if (typeid(T) == typeid(int8_t)) {
      qmeta_.set_meta(IndexMeta::DataType::DT_INT8, dim_);
    } else {
      LOG_ERROR("unsupported type");
      return false;
    }

    cout << "Load query done!" << endl;
    return true;
  }

  void run(core_interface::Index::Pointer index,
           core_interface::BaseIndexQueryParam::Pointer query_param,
           int max_iter, int topk) {
    // Check
    if (batch_queries_.size() == 0) {
      return;
    }

    query_param_ = query_param;
    query_param_->topk = topk;
    query_param_->is_linear = false;

    // A fresh tool run must not inherit the previous run's stop flag.
    stop_now.store(false, std::memory_order_relaxed);
    // Do bench
    signal(SIGINT, stop);
    bench_result_.mark_start();
    auto start_time = Monotime::MilliSeconds();
    for (size_t i = 0; i < threads_; ++i) {
      pool_->execute(this, &Bench<T>::start_bench, index, max_iter, &stop_now);
    }

    while (!pool_->is_finished()) {
      this_thread::sleep_for(chrono::milliseconds(1));
      if (Monotime::MilliSeconds() - start_time > bench_secs_ * 1000) {
        stop_now = true;
      }
    }

    pool_->wait_finish();

    bench_result_.mark_end();
    bench_result_.print();
  }

  void set_tag_lists(const std::vector<std::vector<uint64_t>> &id_to_tags_list,
                     const std::vector<uint64_t> &tag_key_list) {
    id_to_tags_list_ = id_to_tags_list;
    tag_key_list_ = tag_key_list;
  }

 private:
  void start_bench(core_interface::Index::Pointer index, size_t max_iter,
                   const std::atomic<bool> *is_stop) {
    size_t thread_index = pool_->indexof_this();

    size_t i = thread_index;
    for (; i < max_iter && !is_stop->load(std::memory_order_relaxed);
         i += threads_) {
      int idx = i % batch_queries_.size();

      // prefilter
      FilterResultCache filter_cache;
      std::shared_ptr<IndexFilter> filter_ptr = nullptr;
      if (filter_mode_ == FM_TAG) {
        if (batch_taglists_[idx].size() != 1) {
          LOG_ERROR("query tag list not equal to one!");
          return;
        }

        int ret = filter_cache.filter(id_to_tags_list_, batch_taglists_[idx][0],
                                      tag_key_list_);
        if (ret != 0) {
          LOG_ERROR("prefilter failed, idx: %d", idx);
          return;
        }

        auto filter_func = [&](uint64_t key) { return filter_cache.find(key); };

        filter_ptr = std::make_shared<IndexFilter>();
        filter_ptr->set(filter_func);
      }

      auto query_param = query_param_->clone();
      query_param->filter = filter_ptr;


      // Do knn_search
      uint64_t start = Monotime::MicroSeconds();
      int ret;
      if (retrieval_mode_ == RM_DENSE) {
        if (batch_count_ == 1) {
          ret = do_knn_search<T>(index, batch_queries_[idx], query_param);
        } else {
          ret = do_knn_search_batch<T>(index, batch_queries_[idx], query_param);
        }

        if (ret != 0) {
          LOG_ERROR("Failed to knn search, ret=%d %s", ret,
                    IndexError::What(ret));
          return;
        }
      } else {
        std::string mode = retrieval_mode_ == 1 ? "Dense" : "Sparse";
        LOG_ERROR("unsupported retrieval mode: %s", mode.c_str());
      }

      uint64_t end = Monotime::MicroSeconds();

      // Do sample
      bench_result_.add_time(batch_count_, end - start);
    }
  }

  template <typename U>
  typename std::enable_if<
      std::is_same<float, U>::value || std::is_same<int8_t, U>::value ||
          std::is_same<uint32_t, U>::value || std::is_same<uint64_t, U>::value,
      int>::type
  do_knn_search(core_interface::Index::Pointer index, const vector<U> &query,
                core_interface::BaseIndexQueryParam::Pointer query_param) {
    core_interface::DenseVector dense_query;
    dense_query.data = query.data();
    core_interface::VectorData query_data;
    query_data.vector = dense_query;

    core_interface::SearchResult search_result;
    int ret = index->search(query_data, query_param, &search_result);
    if (ret < 0) {
      return ret;
    }

    if (search_result.doc_list_.empty()) {
      LOG_ERROR("Search results is empty");
    }

    return 0;
  }

  template <typename U>
  typename std::enable_if<
      std::is_same<float, U>::value || std::is_same<int8_t, U>::value ||
          std::is_same<uint32_t, U>::value || std::is_same<uint64_t, U>::value,
      int>::type
  do_knn_search_batch(
      core_interface::Index::Pointer index, const vector<U> &query,
      core_interface::BaseIndexQueryParam::Pointer query_param) {
    // For batch search, we search each query separately
    size_t qnum = query.size() / dim_;
    for (size_t i = 0; i < qnum; ++i) {
      core_interface::DenseVector dense_query;
      dense_query.data = query.data() + i * dim_;
      core_interface::VectorData query_data;
      query_data.vector = dense_query;

      core_interface::SearchResult search_result;
      int ret = index->search(query_data, query_param, &search_result);
      if (ret < 0) {
        return ret;
      }

      if (search_result.doc_list_.empty()) {
        LOG_ERROR("Search results is empty for batch query %zu", i);
      }
    }

    return 0;
  }

 private:
  IndexQueryMeta qmeta_{};
  size_t threads_;
  size_t bench_secs_;
  size_t batch_count_;
  size_t dim_;
  shared_ptr<ThreadPool> pool_;
  core_interface::BaseIndexQueryParam::Pointer query_param_;

  vector<vector<T>> batch_queries_;
  vector<vector<uint32_t>> batch_sparse_counts_;
  vector<vector<uint32_t>> batch_sparse_indices_;
  vector<vector<T>> batch_sparse_features_;
  vector<vector<vector<uint64_t>>> batch_taglists_;

  // Tag lists for filtering
  std::vector<std::vector<uint64_t>> id_to_tags_list_;
  std::vector<uint64_t> tag_key_list_;

  BenchResult bench_result_;
  RetrievalMode retrieval_mode_{RM_UNDEFINED};
  FilterMode filter_mode_{FM_NONE};
  static std::atomic<bool> stop_now;
};

template <typename T>
std::atomic<bool> Bench<T>::stop_now{false};

//------------------------------------------------------------
// Sparse Bench
//------------------------------------------------------------
template <typename T>
class SparseBench {
 public:
  SparseBench(size_t threads, size_t bench_secs, size_t batch_count,
              FilterMode filter_mode)
      : threads_(threads),
        bench_secs_(bench_secs),
        batch_count_(batch_count),
        filter_mode_{filter_mode} {
    if (threads_ == 0) {
      pool_ = make_shared<ThreadPool>();
      threads_ = pool_->count();
      cout << "Using cpu count as thread pool count[" << threads_ << "]"
           << endl;
    } else {
      pool_ = make_shared<ThreadPool>(threads_, false);
      threads_ = pool_->count();
      cout << "Using thread pool count[" << threads_ << "]" << endl;
    }
    if (batch_count_ < 1) {
      batch_count_ = 1;
    }
  }

  static void stop(int signo) {
    if (stop_now) {
      exit(signo);
    }
    stop_now = true;
    cout << "\rTrying to stop. press [Ctrl+C] again kill immediately." << endl
         << flush;
  }

  bool load_query(const std::string &query_file, const std::string &first_sep,
                  const std::string &second_sep) {
    TxtInputReader<T> reader;
    vector<vector<T>> queries;
    vector<SparseData<T>> sparse_data;
    vector<vector<uint64_t>> taglists;

    if (!reader.load_query(query_file, first_sep, second_sep, queries,
                           sparse_data, taglists)) {
      LOG_ERROR("Load query error");
      return false;
    }

    linear_sparse_data_ = sparse_data;

    if (batch_count_ == 1) {
      for (size_t i = 0; i < sparse_data.size(); ++i) {
        vector<uint32_t> sparse_count;
        sparse_count.push_back(sparse_data[i].count);

        batch_sparse_counts_.push_back(sparse_count);
        batch_sparse_indices_.push_back(sparse_data[i].indices);
        batch_sparse_features_.push_back(sparse_data[i].features);
      }

      for (size_t i = 0; i < taglists.size(); ++i) {
        vector<vector<uint64_t>> new_taglists;
        new_taglists.push_back(taglists[i]);

        batch_taglists_.push_back(std::move(new_taglists));
      }
    } else {
      size_t num_batch = (queries.size() + batch_count_ - 1) / batch_count_;
      size_t idx = 0;
      for (size_t n = 0; n < num_batch; ++n) {
        vector<uint32_t> batch_sparse_count;
        vector<uint32_t> batch_sparse_indices;
        vector<T> batch_sparse_feature;
        vector<vector<uint64_t>> batch_taglists;

        for (size_t i = 0; i < batch_count_; ++i) {
          batch_sparse_count.push_back(sparse_data[idx].count);

          for (size_t k = 0; k < sparse_data[idx].indices.size(); ++k) {
            batch_sparse_indices.push_back(sparse_data[idx].indices[k]);
          }

          for (size_t k = 0; k < sparse_data[idx].features.size(); ++k) {
            batch_sparse_feature.push_back(sparse_data[idx].features[k]);
          }

          if (taglists.size() > idx) {
            batch_taglists.push_back(taglists[idx]);
          }

          idx = (idx + 1) % queries.size();
        }

        batch_sparse_counts_.push_back(batch_sparse_count);
        batch_sparse_indices_.push_back(batch_sparse_indices);
        batch_sparse_features_.push_back(batch_sparse_feature);
        batch_taglists_.push_back(batch_taglists);
      }
    }

    if (typeid(T) == typeid(float)) {
      qmeta_.set_data_type(IndexMeta::DataType::DT_FP32);
    } else if (typeid(T) == typeid(int8_t)) {
      qmeta_.set_data_type(IndexMeta::DataType::DT_INT8);
    } else {
      LOG_ERROR("unsupported type");
      return false;
    }

    cout << "Load query done!" << endl;
    return true;
  }

  void run(core_interface::Index::Pointer index,
           core_interface::BaseIndexQueryParam::Pointer query_param,
           int max_iter, int topk) {
    // Check
    if (batch_sparse_counts_.size() == 0) {
      return;
    }

    query_param_ = query_param;
    query_param_->topk = topk;
    query_param_->is_linear = false;

    // A fresh tool run must not inherit the previous run's stop flag.
    stop_now.store(false, std::memory_order_relaxed);
    // Do bench
    signal(SIGINT, stop);
    bench_result_.mark_start();
    auto start_time = Monotime::MilliSeconds();
    for (size_t i = 0; i < threads_; ++i) {
      pool_->execute(this, &SparseBench<T>::start_bench, index, max_iter,
                     &stop_now);
    }

    while (!pool_->is_finished()) {
      this_thread::sleep_for(chrono::milliseconds(1));
      if (Monotime::MilliSeconds() - start_time > bench_secs_ * 1000) {
        stop_now = true;
      }
    }

    pool_->wait_finish();

    bench_result_.mark_end();
    bench_result_.print();
  }

  void set_tag_lists(const std::vector<std::vector<uint64_t>> &id_to_tags_list,
                     const std::vector<uint64_t> &tag_key_list) {
    id_to_tags_list_ = id_to_tags_list;
    tag_key_list_ = tag_key_list;
  }

 private:
  void start_bench(core_interface::Index::Pointer index, size_t max_iter,
                   const std::atomic<bool> *is_stop) {
    size_t thread_index = pool_->indexof_this();

    size_t i = thread_index;
    size_t sparse_query_size = batch_sparse_indices_.size();
    for (; i < max_iter && !is_stop->load(std::memory_order_relaxed);
         i += threads_) {
      int idx = i % sparse_query_size;

      // prefilter
      FilterResultCache filter_cache;
      std::shared_ptr<IndexFilter> filter_ptr = nullptr;
      if (filter_mode_ == FM_TAG) {
        if (batch_taglists_[idx].size() != 1) {
          LOG_ERROR("query tag list not equal to one!");
          return;
        }

        int ret = filter_cache.filter(id_to_tags_list_, batch_taglists_[idx][0],
                                      tag_key_list_);
        if (ret != 0) {
          LOG_ERROR("prefilter failed, idx: %d", idx);
          return;
        }

        auto filter_func = [&](uint64_t key) { return filter_cache.find(key); };

        filter_ptr = std::make_shared<IndexFilter>();
        filter_ptr->set(filter_func);
      }

      auto query_param = query_param_->clone();
      query_param->filter = filter_ptr;

      // Do knn_search
      uint64_t start = Monotime::MicroSeconds();
      int ret;
      if (batch_count_ == 1) {
        if (batch_sparse_counts_[idx].size() != 1) {
          LOG_ERROR("Sparse count size should be 1, since batch count is 1");
          return;
        }
        ret = do_knn_search<T>(index, batch_sparse_counts_[idx][0],
                               batch_sparse_indices_[idx],
                               batch_sparse_features_[idx], query_param);
      } else {
        ret = do_knn_search_batch<T>(
            index, batch_sparse_counts_[idx], batch_sparse_indices_[idx],
            batch_sparse_features_[idx], idx, query_param);
      }

      if (ret != 0) {
        LOG_ERROR("Failed to sparse knn search, ret=%d %s", ret,
                  IndexError::What(ret));
        return;
      }

      uint64_t end = Monotime::MicroSeconds();

      // Do sample
      bench_result_.add_time(batch_count_, end - start);
    }
  }

  // sparse search - single query
  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_knn_search(core_interface::Index::Pointer index,
                const uint32_t sparse_count,
                const vector<uint32_t> &sparse_indices,
                const vector<U> &sparse_feature,
                core_interface::BaseIndexQueryParam::Pointer query_param) {
    core_interface::SparseVector sparse_query;
    sparse_query.count = sparse_count;
    sparse_query.indices = sparse_indices.data();
    sparse_query.values = sparse_feature.data();
    core_interface::VectorData query_data;
    query_data.vector = sparse_query;

    core_interface::SearchResult search_result;
    int ret = index->search(query_data, query_param, &search_result);
    if (ret < 0) {
      return ret;
    }

    if (search_result.doc_list_.empty()) {
      LOG_ERROR("Search results is empty");
    }

    return 0;
  }

  template <typename U>
  typename std::enable_if<std::is_same<int8_t, U>::value ||
                              std::is_same<uint32_t, U>::value ||
                              std::is_same<uint64_t, U>::value,
                          int>::type
  do_knn_search(core_interface::Index::Pointer /*index*/,
                const uint32_t /*sparse_count*/,
                const vector<uint32_t> & /*sparse_indices*/,
                const vector<U> & /*sparse_feature*/,
                core_interface::BaseIndexQueryParam::Pointer /*query_param*/) {
    return IndexError_Unsupported;
  }

  // sparse search - batch
  template <typename U>
  typename std::enable_if<std::is_same<float, U>::value, int>::type
  do_knn_search_batch(
      core_interface::Index::Pointer index,
      const vector<uint32_t> &sparse_count,
      const vector<uint32_t> & /*sparse_indices*/,
      const vector<U> & /*sparse_feature*/, size_t batch_idx,
      core_interface::BaseIndexQueryParam::Pointer query_param) {
    // For batch search, search each query separately
    for (size_t i = 0; i < sparse_count.size(); ++i) {
      size_t query_idx = batch_idx * batch_count_ + i;
      if (query_idx >= linear_sparse_data_.size()) {
        break;
      }

      const auto &single_sparse = linear_sparse_data_[query_idx];
      core_interface::SparseVector sparse_query;
      sparse_query.count = single_sparse.count;
      sparse_query.indices = single_sparse.indices.data();
      sparse_query.values = single_sparse.features.data();
      core_interface::VectorData query_data;
      query_data.vector = sparse_query;

      core_interface::SearchResult search_result;
      int ret = index->search(query_data, query_param, &search_result);
      if (ret < 0) {
        return ret;
      }

      if (search_result.doc_list_.empty()) {
        LOG_ERROR("Search results is empty for batch query %zu", i);
      }
    }

    return 0;
  }

  template <typename U>
  typename std::enable_if<std::is_same<int8_t, U>::value ||
                              std::is_same<uint32_t, U>::value ||
                              std::is_same<uint64_t, U>::value,
                          int>::type
  do_knn_search_batch(
      core_interface::Index::Pointer /*index*/,
      const vector<uint32_t> & /*sparse_count*/,
      const vector<uint32_t> & /*sparse_indices*/,
      const vector<U> & /*sparse_feature*/, size_t /*batch_idx*/,
      core_interface::BaseIndexQueryParam::Pointer /*query_param*/) {
    return IndexError_Unsupported;
  }

 private:
  IndexQueryMeta qmeta_{};
  size_t threads_;
  size_t bench_secs_;
  size_t batch_count_;
  core_interface::BaseIndexQueryParam::Pointer query_param_;
  shared_ptr<ThreadPool> pool_;

  vector<SparseData<T>> linear_sparse_data_;
  vector<vector<uint32_t>> batch_sparse_counts_;
  vector<vector<uint32_t>> batch_sparse_indices_;
  vector<vector<T>> batch_sparse_features_;
  vector<vector<vector<uint64_t>>> batch_taglists_;

  // Tag lists for filtering
  std::vector<std::vector<uint64_t>> id_to_tags_list_;
  std::vector<uint64_t> tag_key_list_;

  FilterMode filter_mode_{FM_NONE};
  BenchResult bench_result_;
  static std::atomic<bool> stop_now;
};
template <typename T>
std::atomic<bool> SparseBench<T>::stop_now{false};

bool check_config(YAML::Node &config_node) {
  auto common = config_node["IndexCommon"];
  if (!common) {
    LOG_ERROR("Can not find [IndexCommon] in config");
    return false;
  }
  if (!common["IndexConfig"]) {
    LOG_ERROR("Can not find [IndexConfig] in config");
    return false;
  }
  if (!common["IndexPath"]) {
    LOG_ERROR("Can not find [IndexPath] in config");
    return false;
  }
  if (!common["TopK"]) {
    LOG_ERROR("Can not find [TopK] in config");
    return false;
  }
  if (!common["QueryFile"]) {
    LOG_ERROR("Can not find [QueryFile] in config");
    return false;
  }

  auto query_config = config_node["QueryConfig"];
  if (!query_config) {
    LOG_ERROR("Can not find [QueryConfig] in config");
    return false;
  }
  if (!query_config["QueryParam"]) {
    LOG_ERROR("Can not find [QueryConfig.QueryParam] in config");
    return false;
  }


  return true;
}

void usage() {
  cout << "Usage: bench CONFIG.yaml [plugin file path]\n"
       << "       bench qg build PROFILE TRAIN.f32 NEW_DIRECTORY\n"
       << "       bench qg query PROFILE SNAPSHOT_DIRECTORY QUERIES.f32 "
          "NEW_DIRECTORY\n"
       << "       bench qg {export CONFIG | build CONFIG PROFILE | query "
          "CONFIG PROFILE OUT}"
       << endl;
}


int bench(std::string &query_type, size_t thread_count, size_t batch_count,
          size_t top_k, string query_file, string &first_sep,
          string &second_sep, size_t bench_secs, size_t iter_count,
          core_interface::Index::Pointer index,
          core_interface::BaseIndexQueryParam::Pointer query_param,
          string &index_dir, RetrievalMode retrieval_mode,
          FilterMode filter_mode) {
  if (filter_mode == FM_TAG && batch_count > 1) {
    LOG_ERROR("filter mode can not be run in batch mode");
    return -1;
  }

  std::vector<std::vector<uint64_t>> id_to_tags_list;
  std::vector<uint64_t> tag_key_list;
  // Load tag lists if available
  load_taglists(index_dir, id_to_tags_list, tag_key_list);

  if (query_type == "float") {
    Bench<float> bench(thread_count, bench_secs, batch_count, retrieval_mode,
                       filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    bench.set_tag_lists(id_to_tags_list, tag_key_list);
    bench.run(index, query_param, iter_count, top_k);
  } else if (query_type == "int8") {
    Bench<int8_t> bench(thread_count, bench_secs, batch_count, retrieval_mode,
                        filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    bench.set_tag_lists(id_to_tags_list, tag_key_list);
    bench.run(index, query_param, iter_count, top_k);
  } else if (query_type == "binary") {
    Bench<uint32_t> bench(thread_count, bench_secs, batch_count, retrieval_mode,
                          filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    bench.set_tag_lists(id_to_tags_list, tag_key_list);
    bench.run(index, query_param, iter_count, top_k);
  } else if (query_type == "binary64") {
    Bench<uint64_t> bench(thread_count, bench_secs, batch_count, retrieval_mode,
                          filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    bench.set_tag_lists(id_to_tags_list, tag_key_list);
    bench.run(index, query_param, iter_count, top_k);
  } else {
    LOG_ERROR("Can not recognize type: %s", query_type.c_str());
  }

  return 0;
}

int bench_sparse(std::string &query_type, size_t thread_count,
                 size_t batch_count, size_t top_k, string query_file,
                 string &first_sep, string &second_sep, size_t bench_secs,
                 size_t iter_count, core_interface::Index::Pointer index,
                 core_interface::BaseIndexQueryParam::Pointer query_param,
                 string &index_dir, FilterMode filter_mode) {
  if (filter_mode == FM_TAG && batch_count > 1) {
    LOG_ERROR("filter mode can not be run in batch mode");
    return -1;
  }

  std::vector<std::vector<uint64_t>> id_to_tags_list;
  std::vector<uint64_t> tag_key_list;
  // Load tag lists if available
  load_taglists(index_dir, id_to_tags_list, tag_key_list);

  if (query_type == "float") {
    SparseBench<float> bench(thread_count, bench_secs, batch_count,
                             filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    bench.set_tag_lists(id_to_tags_list, tag_key_list);
    bench.run(index, query_param, iter_count, top_k);
  } else if (query_type == "int8") {
    SparseBench<int8_t> bench(thread_count, bench_secs, batch_count,
                              filter_mode);
    bench.load_query(query_file, first_sep, second_sep);
    bench.set_tag_lists(id_to_tags_list, tag_key_list);
    bench.run(index, query_param, iter_count, top_k);
  } else {
    LOG_ERROR("Can not recognize type: %s", query_type.c_str());
  }

  return 0;
}

// The native QG path also accepts frozen campaign configurations.
namespace {
namespace qg_bench {
void check(bool ok, const std::string &what) {
  if (!ok) throw std::runtime_error(what);
}
void status(int rc, const char *what) {
  if (rc)
    throw std::runtime_error(std::string(what) + ": " + std::to_string(rc));
}
template <class Buffer>
Buffer read(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  check(bool(f), "open " + path);
  const auto size = f.tellg();
  constexpr auto width = sizeof(typename Buffer::value_type);
  check(size >= 0 && size % width == 0, "size " + path);
  Buffer data;
  data.resize(size / width);
  f.seekg(0);
  check(bool(f.read(reinterpret_cast<char *>(data.data()), size)),
        "read " + path);
  return data;
}
template <class Buffer>
void write(const std::string &path, const Buffer &data) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(data.data()),
          data.size() * sizeof(typename Buffer::value_type));
  f.close();
  check(bool(f), "write " + path);
}
// Frozen exports carry external IDs; independent snapshots use row ordinals.
struct Graph : QuantizedGraphData {
  std::vector<uint64_t> ids;
  Graph(const YAML::Node &c, const std::string &snapshot = {}) {
    dimension = c["dimension"].as<size_t>();
    const size_t count = snapshot.empty() ? c["train_count"].as<size_t>()
                             : c["train_shape"][0].as<size_t>();
    const auto p =
        snapshot.empty() ? c["provider_prefix"].as<std::string>() : snapshot;
    vectors = read<std::vector<float>>(p + ".vectors");
    offsets = read<std::vector<uint64_t>>(p + ".offsets");
    edges = read<std::vector<uint32_t>>(p + ".edges");
    check(dimension && count && vectors.size() % dimension == 0 &&
              size() == count && offsets.size() == count + 1 &&
              offsets.front() == 0 && offsets.back() == edges.size(),
          "provider shape");
    if (snapshot.empty()) {
      ids = read<std::vector<uint64_t>>(p + ".ids");
      check(ids.size() == count, "provider ID count");
      auto sorted = ids;
      std::sort(sorted.begin(), sorted.end());
      for (size_t i = 0; i < count; ++i) check(sorted[i] == i, "provider IDs");
    }
    const auto metric = c["metric"].as<std::string>();
    cosine = metric == (snapshot.empty() ? "cosine" : "angular");
    for (size_t i = 0; i < count; ++i) {
      check(offsets[i] <= offsets[i + 1], "provider offsets");
      double norm = 0;
      for (size_t d = 0; d < dimension; ++d) {
        float x = vectors[i * dimension + d];
        check(std::isfinite(x), "nonfinite vector");
        norm += double(x) * x;
      }
      if (cosine)
        check(std::abs(norm - 1) < 2e-5,
              "cosine provider must be normalized exactly once upstream");
    }
    for (auto id : edges) check(id < count, "edge ID range");
  }
};
void export_graph(const YAML::Node &c) {
  auto base = c["base_graph"];
  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  check(bool(storage), "storage factory");
  status(storage->init({}), "storage init");
  status(storage->open(base["source_index"].as<std::string>(), false),
         "base graph open");
  IndexStreamer::Stats stats;
  HnswMmapStreamerEntity entity(stats);
  entity.set_scaling_factor(HnswEntity::kDefaultScalingFactor);
  entity.set_upper_neighbor_cnt(base["m"].as<size_t>());
  entity.set_l0_neighbor_cnt(base["m"].as<size_t>() * 2);
  entity.set_vector_size(c["dimension"].as<size_t>() * 4 +
                         (c["metric"].as<std::string>() == "cosine" ? 4 : 0));
  entity.set_get_vector(true);
  status(entity.init(c["train_count"].as<size_t>()), "entity init");
  status(entity.open(storage, uint64_t{1} << 40, true), "entity open");
  const size_t n = entity.doc_cnt(), dim = c["dimension"].as<size_t>();
  check(n == c["train_count"].as<size_t>(), "base count");
  check(entity.vector_size() >= dim * 4, "base precision/stride");
  check(entity.ef_construction() == base["ef_construction"].as<size_t>(),
        "base ef_construction");
  std::vector<float> vectors(n * dim);
  std::vector<uint64_t> ids(n), offsets{0};
  std::vector<uint32_t> edges;
  for (uint32_t id = 0; id < n; ++id) {
    ids[id] = entity.get_key(id);
    IndexStorage::MemoryBlock v;
    status(entity.get_vector(id, v), "base vector");
    std::memcpy(vectors.data() + size_t(id) * dim, v.data(), dim * 4);
    auto e = entity.get_neighbors(0, id);
    for (size_t j = 0; j < e.size(); ++j) edges.push_back(e[j]);
    offsets.push_back(edges.size());
  }
  const auto p = c["provider_prefix"].as<std::string>();
  write(p + ".vectors", vectors);
  write(p + ".ids", ids);
  write(p + ".offsets", offsets);
  write(p + ".edges", edges);
  std::cout << "exported " << n << " vectors, " << edges.size() << " edges\n";
}
void reject_legacy(const YAML::Node &n) {
  if (n.IsMap())
    for (const auto &entry : n) {
      const auto key = entry.first.as<std::string>();
      check(key.rfind("quantized_graph_", 0) != 0 &&
                key != "use_quantized_graph" && key != "QuantizedGraph" &&
                key != "QuantizedGraphRouter" && key != "ef_search",
            "legacy field rejected: " + key);
      reject_legacy(entry.second);
    }
  else if (n.IsSequence())
    for (const auto &entry : n) reject_legacy(entry);
}
#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
QuantizedGraphEncodingOptions build_options(const YAML::Node &p) {
  auto c = p["encoding"];
  auto o = qg_bench_options::encoding_options(c, true);
  o.threads = c["threads"].as<size_t>();
  check(o.valid() && o.threads > 0, "build options");
  check(o.rotation_views == p["router"]["rotation_views"].as<size_t>(),
        "rotation_views mismatch");
  return o;
}
void build(const YAML::Node &c, const YAML::Node &p) {
  auto g = std::make_shared<Graph>(c);
  auto o = build_options(p);
  const auto dim = g->dimension;
  const auto r = p["router"];
  QuantizedGraphBuilder::Result result;
  std::string codec;
  status(QuantizedGraphIndex::train_codec(*g, g->size(), dim, g->cosine, false, o, codec),
         "train codec");
  status(QuantizedGraphIndex::train_router(
             *g, g->size(), dim, g->cosine, r["anchors"].as<size_t>(),
             r["projected_dimension"].as<size_t>(), r["seed"].as<uint64_t>(),
             result.router, false, o.rotation_views), "train router");
  result.index = std::make_shared<QuantizedGraphIndex>(
      dim, o.max_neighbors, g->cosine, dim * sizeof(float), result.router,
      false, codec);
  status(result.index->prebuild(*g, g->size(), o.threads), "prebuild QG");
  result.data = g;
  status(result.save(p["index_prefix"].as<std::string>() + ".zvec", g->ids),
         "save QG index");
}

void build_from_rows(const YAML::Node &profile, const std::string &input,
                     const std::filesystem::path &output) {
  namespace fs = std::filesystem;
  const auto shape = profile["train_shape"].as<std::vector<size_t>>();
  const auto dim = profile["dimension"].as<size_t>();
  const auto metric = profile["metric"].as<std::string>();
  check(shape.size() == 2 && dim && shape[1] == dim && shape[0] &&
            (metric == "euclidean" || metric == "angular"),
        "profile shape/metric");
  check(!fs::exists(output), "snapshot output already exists");
  auto rows = read<std::vector<float>>(input);
  check(rows.size() % dim == 0 && rows.size() / dim == shape[0],
        "training shape");
  QuantizedGraphBuilder::Result result;
  status(QuantizedGraphBuilder::build(
             std::move(rows), dim, metric == "angular",
             zvec::core::qg_bench_options::build_options(profile), result),
         "build independent graph");

  const fs::path staging = output.string() + ".building";
  check(fs::create_directory(staging),
        "snapshot staging directory already exists");
  try {
    status(result.save((staging / "index.zvec").string()), "save QG index");
    fs::rename(staging, output);
  } catch (...) {
    fs::remove_all(staging);
    throw;
  }
}

QuantizedGraphQuery::Options search_options(const YAML::Node &c) {
  QuantizedGraphQuery::Options o;
  o.margin = c["margin"].as<float>();
  o.bucket_width = c["bucket_width"].as<float>();
  o.max_expansions = c["max_expansions"].as<uint32_t>();
  if (c["route_beam"]) o.route_beam = c["route_beam"].as<uint32_t>();
  if (c["uncertainty_bonus"])
    o.uncertainty_bonus = c["uncertainty_bonus"].as<float>();
  return o;
}
void query(const YAML::Node &c, const YAML::Node &p, const std::string &dest,
           const std::string &snapshot = {},
           const std::string &query_file = {}) {
  const bool independent = !snapshot.empty();
  if (!independent) build_options(p);
  const auto prefix = independent ? snapshot : p["index_prefix"].as<std::string>();
  const auto pages = independent ? zvec::core::qg_bench_options::page_params(p["page_params"])
                                 : zvec::ailego::Params{};
  QuantizedGraphBuilder::Result result;
  std::vector<uint64_t> keys;
  const auto file = independent && std::filesystem::is_regular_file(prefix)
                        ? prefix : prefix + ".zvec";
  if (std::filesystem::exists(file)) {
    status(result.open(file, pages, independent ? nullptr : &keys),
           "open QG index");
  } else {
    auto provider = std::make_shared<Graph>(c, snapshot);
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    check(bool(storage), "snapshot storage factory");
    status(storage->init({}), "init snapshot storage");
    status(storage->open(prefix + ".qg", false), "open legacy snapshot");
    const auto router = read<std::string>(prefix + ".router");
    status(QuantizedGraphIndex::load(
               *provider, provider->size(), provider->dimension, provider->cosine,
               provider->dimension * sizeof(float), router, false,
               storage->get(QuantizedGraphIndex::kSnapshotSegment), result.index, pages),
           "load legacy QG snapshot");
    keys = std::move(provider->ids);
    result.data = std::move(provider);
  }
  const auto &g = *result.data;
  const size_t dim = g.dimension;
  const bool cosine = g.cosine;
  check(dim == c["dimension"].as<size_t>() &&
            g.size() == (independent ? c["train_shape"][0] : c["train_count"]).as<size_t>() &&
            cosine == (c["metric"].as<std::string>() == (independent ? "angular" : "cosine")),
        "QG index differs from profile");
  auto metric =
      IndexFactory::CreateMetric(cosine ? "InnerProduct" : "SquaredEuclidean");
  check(bool(metric), "metric factory");
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, dim);
  status(metric->init(meta, {}), "metric init");
  auto distance = metric->distance();
  auto exact = [distance, cosine](const void *a, const void *b, size_t d,
                                  float *out) {
    distance(a, b, d, out);
    if (cosine) *out += 1.f;
  };
  auto queries = read<std::vector<float>>(
      independent ? query_file : c["queries_file"].as<std::string>());
  const size_t nq =
      independent ? queries.size() / dim : c["query_count"].as<size_t>();
  const size_t k = independent ? 10 : c["protocol"]["topk"].as<size_t>();
  check(nq > 0, "empty query set");
  check(queries.size() == nq * dim, "query shape");
  for (size_t q = 0; q < nq; ++q) {
    double norm = 0;
    for (size_t d = 0; d < dim; ++d) {
      float x = queries[q * dim + d];
      check(std::isfinite(x), "query finite");
      norm += double(x) * x;
    }
    if (cosine && independent) {
      check(norm > 0, "zero cosine query");
      const auto scale = 1 / std::sqrt(norm);
      for (size_t d = 0; d < dim; ++d) queries[q * dim + d] *= scale;
    } else if (cosine) {
      check(std::abs(norm - 1) < 2e-5,
            "query must be normalized exactly once upstream");
    }
  }
  std::filesystem::create_directories(dest);
  QuantizedGraphQuery ctx(dim, exact);
  const auto points = independent ? c["settings"] : c["points"];
  const auto rounds = independent ? 3 : c["protocol"]["rounds"].as<int>();
  for (const auto &point : points) {
    if (!independent &&
        point["profile"].as<std::string>() != p["name"].as<std::string>())
      continue;
    const auto name =
        independent
            ? std::to_string(std::lround(100 * point["target"].as<double>()))
            : point["name"].as<std::string>();
    const auto search = independent ? point["options"] : point["search"];
    auto options = search_options(search);
    const auto capacity = independent && !search["capacity"]
                              ? size_t{128}
                              : (independent ? search : point)["capacity"].as<size_t>();
    const auto scan = independent ? 0 : point["scan_limit"].as<uint64_t>();
    for (int round = -1; round < rounds; ++round) {
      std::vector<uint64_t> ids(nq * k), ns(nq), expanded(nq), comparisons(nq);
      std::vector<float> scores(nq * k);
      for (size_t q = 0; q < nq; ++q) {
        auto start = std::chrono::steady_clock::now();
        status(
            ctx.reset(queries.data() + q * dim, k, capacity, options, scan),
            "reset query");
        status(result.index->search(g, ctx), "QG search");
        const auto &hits = ctx.results();
        check(hits.size() >= k, "fewer than topk results");
        for (size_t j = 0; j < k; ++j) {
          const auto id = hits[j].first;
          ids[q * k + j] = independent ? id : keys[id];
          scores[q * k + j] = hits[j].second;
        }
        auto stop = std::chrono::steady_clock::now();
        ns[q] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start)
                .count();
        expanded[q] = ctx.expanded();
        comparisons[q] = ctx.comparisons();
        check(expanded[q] > 0, "QG path did not expand any nodes");
      }
      if (round < 0) continue;
      auto file = dest + "/" + name + "-round" + std::to_string(round);
      write(file + ".ids", ids);
      write(file + ".scores", scores);
      write(file + ".ns", ns);
      write(file + ".expanded", expanded);
      write(file + ".comparisons", comparisons);
      std::cout << name << " round " << round << " QPS "
                << 1e9 * nq / std::accumulate(ns.begin(), ns.end(), double{0})
                << std::endl;
    }
  }
}
#endif

int run(int argc, char **argv) {
  try {
    check(argc >= 3,
          "usage: bench qg {export CONFIG | build CONFIG PROFILE | query "
          "CONFIG PROFILE OUT}");
    const auto config = YAML::LoadFile(argv[2]);
#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
    if (config["graph_format"]) {
      check(config["graph_format"].as<std::string>() == "independent-fp32-v1" &&
                QuantizedGraphSupported(),
            "unsupported independent QG profile or CPU");
      const std::string_view mode(argv[1]);
      if (mode == "build" && argc == 5) {
        build_from_rows(config, argv[3], argv[4]);
      } else if (mode == "query" && argc == 6) {
        check(!std::filesystem::exists(argv[5]), "query output already exists");
        std::filesystem::path file(argv[3]);
        if (std::filesystem::is_directory(file)) file /= "index";
        query(config, config, argv[5], file.string(), argv[4]);
      } else {
        throw std::runtime_error(
            "expected build PROFILE TRAIN.f32 NEW_DIRECTORY or "
            "query PROFILE SNAPSHOT_DIRECTORY QUERIES.f32 NEW_DIRECTORY");
      }
      return 0;
    }
#endif
    reject_legacy(config);
    check(config["schema_version"].as<int>() == 2 &&
              config["engine"].as<std::string>() == "QuantizedGraphIndex",
          "new API config required");
    const std::string_view mode(argv[1]);
    if (mode == "export") {
      export_graph(config);
      return 0;
    }
#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
    check(QuantizedGraphSupported(),
          "AVX512/VNNI unavailable; refusing fallback");
    check(argc >= 4, "profile required");
    const auto profile = config["profiles"][argv[3]];
    check(bool(profile), "unknown profile");
    if (mode == "build") {
      build(config, profile);
    } else if (mode == "query") {
      check(argc == 5, "output required");
      query(config, profile, argv[4]);
    } else {
      throw std::runtime_error("unknown command");
    }
    return 0;
#else
    throw std::runtime_error("QG not compiled; refusing fallback");
#endif
  } catch (const std::exception &e) {
    std::cerr << "qg_bench: " << e.what() << std::endl;
    return 1;
  }
}
}  // namespace qg_bench
}  // namespace

int main(int argc, char *argv[]) {
  if (argc > 1 && std::string_view(argv[1]) == "qg") {
    return qg_bench::run(argc - 1, argv + 1);
  }
  if (argc > 0 && std::filesystem::path(argv[0]).stem() == "qg_bench") {
    return qg_bench::run(argc, argv);
  }
  if (argc < 2) {
    usage();
    return -1;
  }

  IndexPluginBroker broker;
  std::string error;
  for (int i = 2; i < argc; ++i) {
    if (!broker.emplace(argv[i], &error)) {
      LOG_ERROR("Failed to load plugin: %s (%s)", argv[i], error.c_str());
      return -1;
    }
  }

  YAML::Node config_node;
  try {
    config_node = YAML::LoadFile(argv[1]);
  } catch (...) {
    LOG_ERROR("Load YAML file[%s] failed!", argv[1]);
    return -1;
  }

  if (!check_config(config_node)) {
    return -1;
  }
  auto config_common = config_node["IndexCommon"];

  map<string, int> log_level_map = {
      {"debug", zvec::ailego::Logger::LEVEL_DEBUG},
      {"info", zvec::ailego::Logger::LEVEL_INFO},
      {"warn", zvec::ailego::Logger::LEVEL_WARN},
      {"error", zvec::ailego::Logger::LEVEL_ERROR},
      {"fatal", zvec::ailego::Logger::LEVEL_FATAL}};
  string log_level = config_common["LogLevel"]
                         ? config_common["LogLevel"].as<string>()
                         : "debug";
  transform(log_level.begin(), log_level.end(), log_level.begin(), ::tolower);
  if (log_level_map.find(log_level) != log_level_map.end()) {
    zvec::ailego::LoggerBroker::SetLevel(log_level_map[log_level]);
  }

  // Calculate Bench
  size_t thread_count = config_common["BenchThreadCount"]
                            ? config_common["BenchThreadCount"].as<uint64_t>()
                            : 0;
  size_t iter_count = config_common["BenchIterCount"]
                          ? config_common["BenchIterCount"].as<uint64_t>()
                          : 10000;
  size_t batch_count = config_common["BenchBatchCount"]
                           ? config_common["BenchBatchCount"].as<uint64_t>()
                           : 0;
  g_debug_mode = config_common["DebugMode"]
                     ? config_common["DebugMode"].as<bool>()
                     : false;
  string topk_str = config_common["TopK"].as<string>();

  RetrievalMode retrieval_mode{RM_DENSE};
  if (config_common["RetrievalMode"]) {
    std::string retrieval_mode_str =
        config_common["RetrievalMode"].as<string>();
    if (retrieval_mode_str == "dense") {
      retrieval_mode = RM_DENSE;
    } else if (retrieval_mode_str == "sparse") {
      retrieval_mode = RM_SPARSE;
    }
  }

  FilterMode filter_mode{FM_NONE};
  if (config_common["FilterMode"]) {
    std::string filter_mode_str = config_common["FilterMode"].as<string>();
    if (filter_mode_str == "tag") {
      filter_mode = FM_TAG;
    }
  }

  vector<int32_t> topk_values;
  StringHelper::Split(topk_str, ",", &topk_values);
  size_t top_k = *topk_values.rbegin();
  string query_file = config_common["QueryFile"].as<string>();
  string first_sep = config_common["QueryFirstSep"]
                         ? config_common["QueryFirstSep"].as<string>()
                         : ";";
  string second_sep = config_common["QuerySecondSep"]
                          ? config_common["QuerySecondSep"].as<string>()
                          : " ";
  string query_type = config_common["QueryType"]
                          ? config_common["QueryType"].as<string>()
                          : "float";
  size_t bench_secs = config_common["BenchSecs"]
                          ? config_common["BenchSecs"].as<uint64_t>()
                          : 60;

  string index_dir = config_common["IndexPath"].as<string>();

  core_interface::Index::Pointer index;
  core_interface::BaseIndexQueryParam::Pointer query_param;
  if (0 !=
      parse_and_load_index_param(config_node, index_dir, index, query_param)) {
    LOG_ERROR("Failed to parse and load index param");
    return -1;
  }

  std::vector<core_interface::BaseIndexQueryParam::Pointer> cases;
  if (!parse_tool_query_cases(config_node, query_param, cases)) return -1;
  const int rounds =
      config_common["BenchRounds"] ? config_common["BenchRounds"].as<int>() : 1;
  std::vector<size_t> thread_counts{thread_count};
  if (auto counts = config_common["BenchThreadCounts"]) {
    if (!counts.IsSequence() || counts.size() == 0) return -1;
    thread_counts.clear();
    for (const auto &count : counts) {
      const int value = count.as<int>();
      if (value < 0) return -1;
      thread_counts.push_back(static_cast<size_t>(value));
    }
  }
  if (rounds < 1) return -1;
  for (int round = 0; round < rounds; ++round) {
    for (size_t position = 0; position < cases.size(); ++position) {
      const size_t case_id = round % 2 ? cases.size() - 1 - position : position;
      for (size_t run_threads : thread_counts) {
        std::cout << "BENCH_CASE case=" << case_id << " round=" << round
                  << " threads=" << run_threads << std::endl;
        if (retrieval_mode == RM_SPARSE) {
          bench_sparse(query_type, run_threads, batch_count, top_k, query_file,
                       first_sep, second_sep, bench_secs, iter_count, index,
                       cases[case_id], index_dir, filter_mode);

          cout << "Bench Sparse done." << endl;
        } else {
          bench(query_type, run_threads, batch_count, top_k, query_file,
                first_sep, second_sep, bench_secs, iter_count, index,
                cases[case_id], index_dir, retrieval_mode, filter_mode);

          cout << "Bench done." << endl;
        }
      }
    }
  }

  // Cleanup
  index->close();

  return 0;
}
