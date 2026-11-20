// Copyright 2026-present the zvec project
// Licensed under the Apache License, Version 2.0.
// Native QG benchmark. HNSW is used ONLY to export an immutable base graph.
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <zvec/core/framework/index_factory.h>
#include "algorithm/hnsw/hnsw_streamer_entity.h"
#include "algorithm/quantized_graph/quantized_graph_index.h"
#include "algorithm/quantized_graph/quantized_graph_query.h"

using namespace zvec;
using namespace zvec::core;
namespace {
void check(bool ok, const std::string &what) {
  if (!ok) throw std::runtime_error(what);
}
void status(int rc, const char *what) {
  if (rc) throw std::runtime_error(std::string(what) + ": " + std::to_string(rc));
}
template<class T> std::vector<T> read(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  check(bool(f), "open " + path);
  auto size = f.tellg();
  check(size >= 0 && size % sizeof(T) == 0, "size " + path);
  std::vector<T> v(size / sizeof(T));
  f.seekg(0);
  check(bool(f.read(reinterpret_cast<char *>(v.data()), size)), "read " + path);
  return v;
}
template<class T> void write(const std::string &path, const std::vector<T> &v) {
  std::ofstream f(path, std::ios::binary);
  check(bool(f.write(reinterpret_cast<const char *>(v.data()), v.size()*sizeof(T))), "write " + path);
}
// The anonymous-namespace type enables inlining. Arrays own all borrowed data.
struct Graph {
  struct Vector { const float *p = nullptr; const void *data() const { return p; } };
  struct Edges {
    const uint32_t *p; size_t n;
    size_t size() const { return n; }
    uint32_t operator[](size_t i) const { return p[i]; }
  };
  size_t dim, count;
  std::vector<float> vectors;
  std::vector<uint64_t> offsets, ids;
  std::vector<uint32_t> edges;
  Graph(const YAML::Node &c) : dim(c["dimension"].as<size_t>()),
      count(c["train_count"].as<size_t>()) {
    auto p = c["provider_prefix"].as<std::string>();
    vectors = read<float>(p+".vectors"); offsets = read<uint64_t>(p+".offsets");
    ids = read<uint64_t>(p+".ids"); edges = read<uint32_t>(p+".edges");
    check(dim && count && vectors.size()==dim*count && ids.size()==count &&
          offsets.size()==count+1 && offsets[0]==0 && offsets.back()==edges.size(), "provider shape");
    auto sorted = ids; std::sort(sorted.begin(), sorted.end());
    for (size_t i=0;i<count;++i) {
      check(sorted[i]==i && offsets[i]<=offsets[i+1], "provider IDs/offsets");
      double norm=0;
      for(size_t d=0;d<dim;++d) { float x=vectors[i*dim+d]; check(std::isfinite(x), "nonfinite vector"); norm+=double(x)*x; }
      if(c["metric"].as<std::string>()=="cosine") check(std::abs(norm-1)<2e-5, "cosine provider must be normalized exactly once upstream");
    }
    for(auto id:edges) check(id<count, "edge ID range");
  }
  bool valid(uint32_t id) const { return id<count; }
  int get_vector(uint32_t id, Vector &v) const {
    if(!valid(id)) return IndexError_NoExist;
    v.p=vectors.data()+size_t(id)*dim; return 0;
  }
  Edges neighbors(uint32_t id) const { return {edges.data()+offsets[id],size_t(offsets[id+1]-offsets[id])}; }
};
void export_graph(const YAML::Node &c) {
  auto base=c["base_graph"];
  auto storage=IndexFactory::CreateStorage("MMapFileReadStorage");
  check(bool(storage), "storage factory"); status(storage->init({}), "storage init");
  status(storage->open(base["source_index"].as<std::string>(),false), "base graph open");
  IndexStreamer::Stats stats;
  HnswMmapStreamerEntity entity(stats);
  entity.set_scaling_factor(HnswEntity::kDefaultScalingFactor);
  entity.set_upper_neighbor_cnt(base["m"].as<size_t>());
  entity.set_l0_neighbor_cnt(base["m"].as<size_t>()*2);
  entity.set_vector_size(c["dimension"].as<size_t>()*4+(c["metric"].as<std::string>()=="cosine"?4:0));
  entity.set_get_vector(true);
  status(entity.init(c["train_count"].as<size_t>()), "entity init");
  status(entity.open(storage, uint64_t{1}<<40, true), "entity open");
  const size_t n=entity.doc_cnt(), dim=c["dimension"].as<size_t>();
  check(n==c["train_count"].as<size_t>(), "base count");
  check(entity.vector_size()>=dim*4, "base precision/stride");
  check(entity.ef_construction()==base["ef_construction"].as<size_t>(), "base ef_construction");
  std::vector<float> vectors(n*dim);
  std::vector<uint64_t> ids(n), offsets{0};
  std::vector<uint32_t> edges;
  for(uint32_t id=0;id<n;++id) {
    ids[id]=entity.get_key(id); IndexStorage::MemoryBlock v;
    status(entity.get_vector(id,v), "base vector");
    std::memcpy(vectors.data()+size_t(id)*dim,v.data(),dim*4);
    auto e=entity.get_neighbors(0,id);
    for(size_t j=0;j<e.size();++j) edges.push_back(e[j]);
    offsets.push_back(edges.size());
  }
  const auto p=c["provider_prefix"].as<std::string>();
  write(p+".vectors",vectors); write(p+".ids",ids); write(p+".offsets",offsets); write(p+".edges",edges);
  std::cout<<"exported "<<n<<" vectors, "<<edges.size()<<" edges\n";
}
void reject_legacy(const YAML::Node &n) {
  if (n.IsMap()) for (const auto &entry : n) {
    const auto key=entry.first.as<std::string>();
    check(key.rfind("quantized_graph_",0)!=0 && key!="use_quantized_graph" &&
          key!="QuantizedGraph" && key!="QuantizedGraphRouter" && key!="ef_search",
          "legacy field rejected: "+key);
    reject_legacy(entry.second);
  }
  else if(n.IsSequence()) for(const auto &entry:n) reject_legacy(entry);
}
#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
QuantizedGraphBuildOptions build_options(const YAML::Node &p) {
  auto c=p["encoding"]; QuantizedGraphBuildOptions o;
  o.max_neighbors=c["max_neighbors"].as<size_t>(); o.threads=c["threads"].as<size_t>();
  o.rotation_views=c["rotation_views"].as<size_t>();
  auto codec=c["codec"].as<std::string>(), book=c["codebook"].as<std::string>();
  if(codec=="rabitq" || codec=="native") o.codec=decltype(o.codec)::RaBitQ;
  else if(codec=="global") o.codec=decltype(o.codec)::Global;
  else if(codec=="low_rank") o.codec=decltype(o.codec)::LowRank;
  else throw std::runtime_error("unknown codec");
  if(book=="hadamard") o.codebook=decltype(o.codebook)::Hadamard;
  else if(book=="gaussian") o.codebook=decltype(o.codebook)::Gaussian;
  else throw std::runtime_error("unknown codebook");
  o.block_dimension=c["block_dimension"].as<size_t>(); o.rank=c["rank"].as<size_t>();
  o.training_samples=c["training_samples"].as<size_t>(); o.seed=c["seed"].as<uint32_t>();
  o.center_weight=c["center_weight"].as<double>();
  check(o.valid() && o.threads>0, "build options");
  check(o.rotation_views==p["router"]["rotation_views"].as<size_t>(), "rotation_views mismatch");
  return o;
}
void build(const YAML::Node &c, const YAML::Node &p) {
  Graph g(c); auto o=build_options(p); bool cosine=c["metric"].as<std::string>()=="cosine";
  auto r=p["router"]; std::string codec,router;
  status(QuantizedGraphIndex::train_codec(g,g.count,g.dim,cosine,false,o,codec), "train codec");
  status(QuantizedGraphIndex::train_router(g,g.count,g.dim,cosine,r["anchors"].as<size_t>(),
      r["projected_dimension"].as<size_t>(),r["seed"].as<uint64_t>(),router,false,o.rotation_views), "train router");
  QuantizedGraphIndex index(g.dim,o.max_neighbors,cosine,g.dim*4,router,false,codec);
  status(index.prebuild(g,g.count,o.threads), "prebuild QG");
  const auto prefix=p["index_prefix"].as<std::string>();
  write(prefix+".router",std::vector<char>(router.begin(),router.end()));
  auto dumper=IndexFactory::CreateDumper("FileDumper"); check(bool(dumper), "dumper");
  status(dumper->init({}), "dumper init"); status(dumper->create(prefix+".qg"), "dumper create");
  status(index.dump(dumper), "dump QG"); status(dumper->close(), "dumper close");
  std::cout<<"built native QuantizedGraphIndex "<<prefix<<std::endl;
}
QuantizedGraphSearchOptions search_options(const YAML::Node &c) {
  QuantizedGraphSearchOptions o;
  o.margin=c["margin"].as<float>(); o.bucket_width=c["bucket_width"].as<float>();
  o.max_expansions=c["max_expansions"].as<uint32_t>(); o.route_beam=c["route_beam"].as<uint32_t>();
  o.uncertainty_bonus=c["uncertainty_bonus"].as<float>(); return o;
}
void query(const YAML::Node &c, const YAML::Node &p, const std::string &dest) {
  Graph g(c); build_options(p);
  const auto prefix=p["index_prefix"].as<std::string>();
  auto rb=read<char>(prefix+".router"); std::string router(rb.begin(),rb.end());
  auto storage=IndexFactory::CreateStorage("MMapFileReadStorage");
  status(storage->init({}), "init snapshot storage"); status(storage->open(prefix+".qg",false), "open snapshot");
  std::shared_ptr<QuantizedGraphIndex> index;
  bool cosine=c["metric"].as<std::string>()=="cosine";
  status(QuantizedGraphIndex::load(g,g.count,g.dim,cosine,g.dim*4,router,false,
      storage->get(QuantizedGraphIndex::kSnapshotSegment),index), "load new QG snapshot");
  auto metric=IndexFactory::CreateMetric(cosine?"InnerProduct":"SquaredEuclidean");
  check(bool(metric), "metric factory"); IndexMeta meta; meta.set_meta(IndexMeta::DT_FP32,g.dim);
  status(metric->init(meta,{}), "metric init"); auto distance=metric->distance();
  auto exact=[distance,cosine](const void *a,const void *b,size_t d,float *out){
    distance(a,b,d,out); if(cosine) *out+=1.f;
  };
  auto queries=read<float>(c["queries_file"].as<std::string>());
  const size_t nq=c["query_count"].as<size_t>(), k=c["protocol"]["topk"].as<size_t>();
  check(queries.size()==nq*g.dim, "query shape");
  for(size_t q=0;q<nq;++q) {
    double norm=0; for(size_t d=0;d<g.dim;++d) {float x=queries[q*g.dim+d];check(std::isfinite(x),"query finite");norm+=double(x)*x;}
    if(cosine) check(std::abs(norm-1)<2e-5,"query must be normalized exactly once upstream");
  }
  std::filesystem::create_directories(dest);
  QuantizedGraphQuery ctx(g.dim,exact);
  for(const auto &point:c["points"]) {
    if(point["profile"].as<std::string>()!=p["name"].as<std::string>()) continue;
    const auto name=point["name"].as<std::string>(); auto options=search_options(point["search"]);
    const size_t capacity=point["capacity"].as<size_t>(); auto scan=point["scan_limit"].as<uint64_t>();
    // One full warmup, then five measured rounds. Each process is one independent repetition.
    for(int round=-1;round<c["protocol"]["rounds"].as<int>();++round) {
      std::vector<uint64_t> ids(nq*k), ns(nq), expanded(nq), comparisons(nq);
      std::vector<float> scores(nq*k);
      for(size_t q=0;q<nq;++q) {
        auto start=std::chrono::steady_clock::now();
        status(ctx.reset(queries.data()+q*g.dim,k,capacity,options,scan), "reset query");
        status(index->search(g,ctx), "QG search");
        const auto &result=ctx.results();
        check(result.size()>=k,"fewer than topk results");
        for(size_t j=0;j<k;++j) {ids[q*k+j]=g.ids[result[j].first];scores[q*k+j]=result[j].second;}
        auto stop=std::chrono::steady_clock::now();
        ns[q]=std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        expanded[q]=ctx.expanded(); comparisons[q]=ctx.comparisons();
        check(expanded[q]>0,"QG path did not expand any nodes");
      }
      if(round<0) continue;
      auto file=dest+"/"+name+"-round"+std::to_string(round);
      write(file+".ids",ids);write(file+".scores",scores);write(file+".ns",ns);
      write(file+".expanded",expanded);write(file+".comparisons",comparisons);
      std::cout<<name<<" round "<<round<<" QPS "<<1e9*nq/std::accumulate(ns.begin(),ns.end(),double{0})<<std::endl;
    }
  }
}
#endif
} // namespace
int main(int argc,char **argv) {
  try {
    check(argc>=3,"usage: qg_bench export CONFIG | build CONFIG PROFILE | query CONFIG PROFILE OUT");
    auto c=YAML::LoadFile(argv[2]);
    reject_legacy(c);
    check(c["schema_version"].as<int>()==2 && c["engine"].as<std::string>()=="QuantizedGraphIndex", "new API config required");
    std::string mode=argv[1];
    if(mode=="export") { export_graph(c); return 0; }
#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
    check(QuantizedGraphSupported(),"AVX512/VNNI unavailable; refusing fallback");
    check(argc>=4,"profile required");
    auto p=c["profiles"][argv[3]]; check(bool(p),"unknown profile");
    if(mode=="build") build(c,p);
    else if(mode=="query") {check(argc==5,"output required");query(c,p,argv[4]);}
    else throw std::runtime_error("unknown command");
#else
    throw std::runtime_error("QG not compiled; refusing fallback");
#endif
  } catch(const std::exception &e) {std::cerr<<"qg_bench: "<<e.what()<<std::endl;return 1;}
}
