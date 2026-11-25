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

#include <cmath>
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include "core/interface/utils/utils.h"
#include "indexes/quantized_graph_index.h"

namespace zvec::core_interface {
namespace {
bool HasLegacyQuantizedGraphOptions(const ailego::JsonObject &json) {
  ailego::JsonValue value;
  for (const char *key :
       {"use_quantized_graph", "quantized_graph_model",
        "quantized_graph_margin", "quantized_graph_bucket_width",
        "quantized_graph_max_expansions", "quantized_graph_route_beam",
        "quantized_graph_uncertainty_bonus"}) {
    if (json.get(key, &value)) {
      LOG_ERROR(
          "%s belongs to the retired HNSW quantized-graph entry; use "
          "QuantizedGraphIndex and QuantizedGraphQuery",
          key);
      return true;
    }
  }
  return false;
}
}  // namespace


Index::Pointer IndexFactory::CreateAndInitIndex(const BaseIndexParam &param) {
  Index::Pointer ptr = nullptr;
  // if (param.index_type == IndexType::kIVF) {
  //   const IVFIndexParam *_param = dynamic_cast<const IVFIndexParam
  //   *>(&param); ptr = std::make_shared<IVFIndex>(param);

  //   if (_param->l1Index) {
  //     // TODO: create l1 index
  //   }
  //   if (_param->l2Index) {
  //     // TODO: create l2 index
  //   }
  // }
  // if (param.index_type == IndexType::kHNSW) {
  //   ptr = std::make_shared<HNSWIndex>(param);
  // }
  if (param.index_type == IndexType::kQuantizedGraph) {
    ptr = std::make_shared<QuantizedGraphIndex>();
  } else if (param.index_type == IndexType::kFlat) {
    // ptr = std::make_shared<FlatIndex>(param);
    ptr = std::make_shared<FlatIndex>();
  } else if (param.index_type == IndexType::kHNSW) {
    ptr = std::make_shared<HNSWIndex>();
  } else if (param.index_type == IndexType::kIVF) {
    ptr = std::make_shared<IVFIndex>();
  } else if (param.index_type == IndexType::kHNSWRabitq) {
    ptr = std::make_shared<HNSWRabitqIndex>();
  } else if (param.index_type == IndexType::kIVFRabitq) {
    ptr = std::make_shared<IVFRabitqIndex>();
#if DISKANN_SUPPORTED
  } else if (param.index_type == IndexType::kDiskAnn) {
    ptr = std::make_shared<DiskAnnIndex>();
#endif
  } else if (param.index_type == IndexType::kVamana) {
    ptr = std::make_shared<VamanaIndex>();
  } else {
    LOG_ERROR("Unsupported index type: ");
    return nullptr;
  }

  if (!ptr) {
    LOG_ERROR("Failed to create index");
    return nullptr;
  }
  if (0 != ptr->init(param)) {
    LOG_ERROR("Failed to init index");
    return nullptr;
  }
  return ptr;
}

BaseIndexParam::Pointer IndexFactory::DeserializeIndexParamFromJson(
    const std::string &json_str) {
  ailego::JsonValue json_value;
  if (!json_value.parse(json_str)) {
    LOG_ERROR("Failed to parse json string: %s", json_str.c_str());
    return nullptr;
  }
  ailego::JsonObject json_obj = json_value.as_object();
  if (HasLegacyQuantizedGraphOptions(json_obj)) return nullptr;
  ailego::JsonValue tmp_json_value;

  IndexType index_type;

  if (!extract_enum_from_json<IndexType>(json_obj, "index_type", index_type,
                                         tmp_json_value)) {
    LOG_ERROR("Failed to deserialize index type");
    return nullptr;
  }

  switch (index_type) {
    case IndexType::kQuantizedGraph: {
      auto param = std::make_shared<BaseIndexParam>(index_type);
      return param->deserialize_from_json(json_str) ? param : nullptr;
    }
    case IndexType::kFlat: {
      FlatIndexParam::Pointer param = std::make_shared<FlatIndexParam>();
      if (!param->deserialize_from_json(json_str)) {
        LOG_ERROR("Failed to deserialize flat index param");
        return nullptr;
      }
      return param;
    }
    case IndexType::kHNSW: {
      HNSWIndexParam::Pointer param = std::make_shared<HNSWIndexParam>();
      if (!param->deserialize_from_json(json_str)) {
        LOG_ERROR("Failed to deserialize hnsw index param");
        return nullptr;
      }
      return param;
    }
    case IndexType::kIVF: {
      IVFIndexParam::Pointer param = std::make_shared<IVFIndexParam>();
      if (!param->deserialize_from_json(json_str)) {
        LOG_ERROR("Failed to deserialize hnsw index param");
        return nullptr;
      }
      return param;
    }
    case IndexType::kHNSWRabitq: {
      HNSWRabitqIndexParam::Pointer param =
          std::make_shared<HNSWRabitqIndexParam>();
      if (!param->deserialize_from_json(json_str)) {
        LOG_ERROR("Failed to deserialize hnsqrabitq index param");
        return nullptr;
      }
      return param;
    }
    case IndexType::kIVFRabitq: {
      IVFRabitqIndexParam::Pointer param =
          std::make_shared<IVFRabitqIndexParam>();
      if (!param->deserialize_from_json(json_str)) {
        LOG_ERROR("Failed to deserialize ivf_rabitq index param");
        return nullptr;
      }
      return param;
    }
    case IndexType::kVamana: {
      VamanaIndexParam::Pointer param = std::make_shared<VamanaIndexParam>();
      if (!param->deserialize_from_json(json_str)) {
        LOG_ERROR("Failed to deserialize vamana index param");
        return nullptr;
      }
      return param;
    }
#if DISKANN_SUPPORTED
    case IndexType::kDiskAnn: {
      DiskAnnIndexParam::Pointer param = std::make_shared<DiskAnnIndexParam>();
      if (!param->deserialize_from_json(json_str)) {
        LOG_ERROR("Failed to deserialize diskann index param");
        return nullptr;
      }
      return param;
    }
#endif
    default:
      LOG_ERROR("Unsupported index type: %s",
                magic_enum::enum_name(index_type).data());
      return nullptr;
  }
}

template <typename QueryParamType,
          std::enable_if_t<
              std::is_base_of_v<BaseIndexQueryParam, QueryParamType>, bool> >
std::string IndexFactory::QueryParamSerializeToJson(const QueryParamType &param,
                                                    bool omit_empty_value) {
  ailego::JsonObject json_obj;

  // BaseIndexQueryParam
  // omit filter & bf_pks
  if (!omit_empty_value || param.topk != 0) {
    json_obj.set("topk", ailego::JsonValue(param.topk));
  }
  if (!omit_empty_value || param.fetch_vector) {
    json_obj.set("fetch_vector", ailego::JsonValue(param.fetch_vector));
  }
  if (!omit_empty_value || param.radius != 0.0f) {
    json_obj.set("radius", ailego::JsonValue(param.radius));
  }
  if (!omit_empty_value || param.is_linear) {
    json_obj.set("is_linear", ailego::JsonValue(param.is_linear));
  }

  IndexType index_type{IndexType::kNone};
  if constexpr (std::is_same_v<QueryParamType, FlatQueryParam>) {
    index_type = IndexType::kFlat;
  } else if constexpr (std::is_same_v<QueryParamType, QuantizedGraphQueryParam>) {
    index_type = IndexType::kQuantizedGraph;
    json_obj.set("capacity", ailego::JsonValue(param.capacity));
    json_obj.set("margin", ailego::JsonValue(param.margin));
    json_obj.set("bucket_width", ailego::JsonValue(param.bucket_width));
    json_obj.set("max_expansions", ailego::JsonValue(param.max_expansions));
    json_obj.set("route_beam", ailego::JsonValue(param.route_beam));
    json_obj.set("uncertainty_bonus", ailego::JsonValue(param.uncertainty_bonus));
  } else if constexpr (std::is_same_v<QueryParamType, HNSWQueryParam>) {
    if (!omit_empty_value || param.ef_search != 0) {
      json_obj.set("ef_search", ailego::JsonValue(param.ef_search));
    }
    if (!omit_empty_value || param.prefetch_offset != 0) {
      json_obj.set("prefetch_offset", ailego::JsonValue(param.prefetch_offset));
    }
    if (!omit_empty_value || param.prefetch_lines != 0) {
      json_obj.set("prefetch_lines", ailego::JsonValue(param.prefetch_lines));
    }
    index_type = IndexType::kHNSW;
  } else if constexpr (std::is_same_v<QueryParamType, IVFQueryParam>) {
    json_obj.set("nprobe", ailego::JsonValue(param.nprobe));
    index_type = IndexType::kIVF;
    // json_obj.set("l1QueryParam",
    // ailego::JsonValue(QueryParamSerializeToJson(param.l1QueryParam)));
    // json_obj.set("l2QueryParam",
    // ailego::JsonValue(QueryParamSerializeToJson(param.l2QueryParam)));
  } else if constexpr (std::is_same_v<QueryParamType, HNSWRabitqQueryParam>) {
    if (!omit_empty_value || param.ef_search != 0) {
      json_obj.set("ef_search", ailego::JsonValue(param.ef_search));
    }
    index_type = IndexType::kHNSWRabitq;
  } else if constexpr (std::is_same_v<QueryParamType, IVFRabitqQueryParam>) {
    if (!omit_empty_value || param.nprobe != 0) {
      json_obj.set("nprobe", ailego::JsonValue(param.nprobe));
    }
    index_type = IndexType::kIVFRabitq;
  } else if constexpr (std::is_same_v<QueryParamType, VamanaQueryParam>) {
    if (!omit_empty_value || param.ef_search != 0) {
      json_obj.set("ef_search", ailego::JsonValue(param.ef_search));
    }
    if (!omit_empty_value || param.prefetch_offset != 0) {
      json_obj.set("prefetch_offset", ailego::JsonValue(param.prefetch_offset));
    }
    if (!omit_empty_value || param.prefetch_lines != 0) {
      json_obj.set("prefetch_lines", ailego::JsonValue(param.prefetch_lines));
    }
    index_type = IndexType::kVamana;
#if DISKANN_SUPPORTED
  } else if constexpr (std::is_same_v<QueryParamType, DiskAnnQueryParam>) {
    if (!omit_empty_value || param.list_size != 0) {
      json_obj.set("list_size", ailego::JsonValue(param.list_size));
    }
    index_type = IndexType::kDiskAnn;
#endif
  }

  json_obj.set("index_type",
               ailego::JsonValue(magic_enum::enum_name(index_type).data()));

  return ailego::JsonValue(json_obj).as_json_string().as_stl_string();
}

template std::string
IndexFactory::QueryParamSerializeToJson<BaseIndexQueryParam>(
    const BaseIndexQueryParam &param, bool omit_empty_value);
template std::string IndexFactory::QueryParamSerializeToJson<FlatQueryParam>(
    const FlatQueryParam &param, bool omit_empty_value);
template std::string IndexFactory::QueryParamSerializeToJson<HNSWQueryParam>(
    const HNSWQueryParam &param, bool omit_empty_value);
template std::string IndexFactory::QueryParamSerializeToJson<IVFQueryParam>(
    const IVFQueryParam &param, bool omit_empty_value);

template <typename QueryParamType,
          std::enable_if_t<
              std::is_base_of_v<BaseIndexQueryParam, QueryParamType>, bool> >
typename QueryParamType::Pointer IndexFactory::QueryParamDeserializeFromJson(
    const std::string &json_str) {
  ailego::JsonValue tmp_json_value;
  if (!tmp_json_value.parse(json_str)) {
    LOG_ERROR("Failed to parse json string: %s", json_str.c_str());
    return nullptr;
  }
  ailego::JsonObject json_obj = tmp_json_value.as_object();
  if (HasLegacyQuantizedGraphOptions(json_obj)) return nullptr;

  auto parse_common_fields = [&](auto &param) -> bool {
    if (!extract_value_from_json(json_obj, "topk", param->topk,
                                 tmp_json_value)) {
      LOG_ERROR("Failed to deserialize topk");
      return false;
    }

    if (!extract_value_from_json(json_obj, "fetch_vector", param->fetch_vector,
                                 tmp_json_value)) {
      LOG_ERROR("Failed to deserialize fetch_vector");
      return false;
    }

    if (!extract_value_from_json(json_obj, "radius", param->radius,
                                 tmp_json_value)) {
      LOG_ERROR("Failed to deserialize radius");
      return false;
    }

    if (!extract_value_from_json(json_obj, "is_linear", param->is_linear,
                                 tmp_json_value)) {
      LOG_ERROR("Failed to deserialize is_linear");
      return false;
    }
    return true;
  };

  IndexType index_type;

  if (!extract_enum_from_json<IndexType>(json_obj, "index_type", index_type,
                                         tmp_json_value)) {
    LOG_ERROR("Failed to deserialize index type");
    return nullptr;
  }

  if constexpr (std::is_same_v<QueryParamType, BaseIndexQueryParam>) {
    if (index_type == IndexType::kQuantizedGraph) {
      return QueryParamDeserializeFromJson<QuantizedGraphQueryParam>(json_str);
    } else if (index_type == IndexType::kFlat) {
      auto param = std::make_shared<FlatQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      return param;
    } else if (index_type == IndexType::kHNSW) {
      auto param = std::make_shared<HNSWQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "ef_search", param->ef_search,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize ef_search");
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "prefetch_offset",
                                   param->prefetch_offset, tmp_json_value)) {
        LOG_ERROR("Failed to deserialize prefetch_offset");
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "prefetch_lines",
                                   param->prefetch_lines, tmp_json_value)) {
        LOG_ERROR("Failed to deserialize prefetch_lines");
        return nullptr;
      }
      return param;
    } else if (index_type == IndexType::kIVF) {
      auto param = std::make_shared<IVFQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "nprobe", param->nprobe,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize nprobe");
        return nullptr;
      }
      return param;
    } else if (index_type == IndexType::kHNSWRabitq) {
      auto param = std::make_shared<HNSWRabitqQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "ef_search", param->ef_search,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize ef_search");
        return nullptr;
      }
      return param;
    } else if (index_type == IndexType::kIVFRabitq) {
      auto param = std::make_shared<IVFRabitqQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "nprobe", param->nprobe,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize nprobe");
        return nullptr;
      }
      return param;
    } else if (index_type == IndexType::kVamana) {
      auto param = std::make_shared<VamanaQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "ef_search", param->ef_search,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize ef_search");
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "prefetch_offset",
                                   param->prefetch_offset, tmp_json_value)) {
        LOG_ERROR("Failed to deserialize prefetch_offset");
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "prefetch_lines",
                                   param->prefetch_lines, tmp_json_value)) {
        LOG_ERROR("Failed to deserialize prefetch_lines");
        return nullptr;
      }
      return param;
#if DISKANN_SUPPORTED
    } else if (index_type == IndexType::kDiskAnn) {
      auto param = std::make_shared<DiskAnnQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "list_size", param->list_size,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize list_size");
        return nullptr;
      }
      return param;
#endif
    } else {
      LOG_ERROR("Unsupported index type: %s",
                magic_enum::enum_name(index_type).data());
      return nullptr;
    }
  } else {
    auto param = std::make_shared<QueryParamType>();
    if (!parse_common_fields(param)) {
      return nullptr;
    }
    if constexpr (std::is_same_v<QueryParamType, FlatQueryParam>) {
    } else if constexpr (std::is_same_v<QueryParamType, HNSWQueryParam>) {
      if (!extract_value_from_json(json_obj, "ef_search", param->ef_search,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize ef_search");
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "prefetch_offset",
                                   param->prefetch_offset, tmp_json_value)) {
        LOG_ERROR("Failed to deserialize prefetch_offset");
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "prefetch_lines",
                                   param->prefetch_lines, tmp_json_value)) {
        LOG_ERROR("Failed to deserialize prefetch_lines");
        return nullptr;
      }
    } else if constexpr (std::is_same_v<QueryParamType, IVFQueryParam>) {
      if (!extract_value_from_json(json_obj, "nprobe", param->nprobe,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize nprobe");
        return nullptr;
      }
    } else if constexpr (std::is_same_v<QueryParamType, HNSWRabitqQueryParam>) {
      if (!extract_value_from_json(json_obj, "ef_search", param->ef_search,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize ef_search");
        return nullptr;
      }
    } else if constexpr (std::is_same_v<QueryParamType, IVFRabitqQueryParam>) {
      if (!extract_value_from_json(json_obj, "nprobe", param->nprobe,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize nprobe");
        return nullptr;
      }
    } else if constexpr (std::is_same_v<QueryParamType, QuantizedGraphQueryParam>) {
      if (index_type != IndexType::kQuantizedGraph) return nullptr;
      auto read_uint = [&](const char *key, uint32_t &out) {
        if (!json_obj.get(key, &tmp_json_value)) return true;
        if (!tmp_json_value.is_integer()) return false;
        const auto val = tmp_json_value.as_integer();
        if (val < 0 || static_cast<uint64_t>(val) > UINT32_MAX) return false;
        out = static_cast<uint32_t>(val);
        return true;
      };
      auto read_float = [&](const char *key, float &out) {
        return extract_value_from_json(json_obj, key, out, tmp_json_value);
      };
      if (!read_uint("capacity", param->capacity) || !param->capacity ||
          !read_uint("route_beam", param->route_beam) || param->route_beam > 4 ||
          !read_uint("max_expansions", param->max_expansions) ||
          !read_float("margin", param->margin) ||
          !read_float("bucket_width", param->bucket_width) ||
          !read_float("uncertainty_bonus", param->uncertainty_bonus) ||
          !std::isfinite(param->margin) ||
          (param->margin < 0 && param->margin != -1) ||
          !(param->bucket_width >= 0 && param->bucket_width <= 1) ||
          !(param->uncertainty_bonus >= 0 && param->uncertainty_bonus <= 1))
        return nullptr;
    } else if constexpr (std::is_same_v<QueryParamType, VamanaQueryParam>) {
      if (!extract_value_from_json(json_obj, "ef_search", param->ef_search,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize ef_search");
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "prefetch_offset",
                                   param->prefetch_offset, tmp_json_value)) {
        LOG_ERROR("Failed to deserialize prefetch_offset");
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "prefetch_lines",
                                   param->prefetch_lines, tmp_json_value)) {
        LOG_ERROR("Failed to deserialize prefetch_lines");
        return nullptr;
      }
#if DISKANN_SUPPORTED
    } else if constexpr (std::is_same_v<QueryParamType, DiskAnnQueryParam>) {
      if (!extract_value_from_json(json_obj, "list_size", param->list_size,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize list_size");
        return nullptr;
      }
#endif
    } else {
      LOG_ERROR("Unsupported index type: %s",
                magic_enum::enum_name(index_type).data());
      return nullptr;
    }
    return param;
  }
}

template BaseIndexQueryParam::Pointer
IndexFactory::QueryParamDeserializeFromJson<BaseIndexQueryParam>(
    const std::string &json_str);
template FlatQueryParam::Pointer IndexFactory::QueryParamDeserializeFromJson<
    FlatQueryParam>(const std::string &json_str);
template HNSWQueryParam::Pointer IndexFactory::QueryParamDeserializeFromJson<
    HNSWQueryParam>(const std::string &json_str);
template IVFQueryParam::Pointer IndexFactory::QueryParamDeserializeFromJson<
    IVFQueryParam>(const std::string &json_str);
template std::string IndexFactory::QueryParamSerializeToJson<QuantizedGraphQueryParam>(
    const QuantizedGraphQueryParam &param, bool omit_empty_value);
template QuantizedGraphQueryParam::Pointer
IndexFactory::QueryParamDeserializeFromJson<QuantizedGraphQueryParam>(
    const std::string &json_str);
template std::string IndexFactory::QueryParamSerializeToJson<VamanaQueryParam>(
    const VamanaQueryParam &param, bool omit_empty_value);
template VamanaQueryParam::Pointer IndexFactory::QueryParamDeserializeFromJson<
    VamanaQueryParam>(const std::string &json_str);
template std::string
IndexFactory::QueryParamSerializeToJson<IVFRabitqQueryParam>(
    const IVFRabitqQueryParam &param, bool omit_empty_value);
template IVFRabitqQueryParam::Pointer
IndexFactory::QueryParamDeserializeFromJson<IVFRabitqQueryParam>(
    const std::string &json_str);
#if DISKANN_SUPPORTED
template std::string IndexFactory::QueryParamSerializeToJson<DiskAnnQueryParam>(
    const DiskAnnQueryParam &param, bool omit_empty_value);
template DiskAnnQueryParam::Pointer IndexFactory::QueryParamDeserializeFromJson<
    DiskAnnQueryParam>(const std::string &json_str);
#endif

}  // namespace zvec::core_interface
