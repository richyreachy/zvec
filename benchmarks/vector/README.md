# Vector benchmarks

Vector index construction, recall evaluation, query throughput, and dataset conversion.

## Build

Follow the [build configuration](../README.md), then run from the repository root:

```bash
cmake --build build --target \
  txt2vecs local_builder recall bench \
  local_builder_original recall_original bench_original
```

| Target | Purpose | Configuration |
| --- | --- | --- |
| `txt2vecs` | Convert text vectors to binary data | Command-line flags |
| `local_builder`, `local_builder_original` | Build indexes | `BuilderCommon`, `BuilderParams` |
| `recall`, `bench` | Measure recall or query throughput via the index interface | `IndexCommon.IndexConfig`, `QueryConfig.QueryParam` |
| `recall_original`, `bench_original` | Measure recall or query throughput via framework plugins | `SearcherCommon`, `SearcherParams` |

Run YAML-based tools with `build/bin/<target> <config.yaml>`.

The COHERE example below uses the `_original` tools. Its search YAML is not
interchangeable with the `IndexCommon`/`QueryConfig` format used by `recall` and
`bench`. Run all local commands from the repository root.

## Building a quantized graph

ANN-Benchmarks provides machine-specific profiles for SIFT, GIST and GloVe.
See the [r7i presets](https://github.com/richyreachy/ann-benchmarks/tree/r7ibench_nice/ann_benchmarks/algorithms/zvec_qg/configs/r7i-native-builder)
and [g9i presets](https://github.com/richyreachy/ann-benchmarks/tree/g9ibench/ann_benchmarks/algorithms/zvec_qg/configs/g9i).

For dense FP32 Zvec `.vecs` input, save the following as `qg-build.yaml`
and run `build/bin/local_builder qg-build.yaml`.

```yaml
BuilderCommon:
  BuilderClass: QuantizedGraphBuilder
  BuildFile: train.vecs
  DumpPath: index.zvec
  MetricName: SquaredEuclidean
BuilderParams:
  zvec.quantized_graph.builder.threads: !!int 4
  zvec.quantized_graph.builder.encoding.max_neighbors: !!int 64
```

Use `Cosine` for cosine distance. Use `BuilderParams: {}` for defaults.
The index file preserves the input keys and can be queried with the native bench tools.

For internal IDs in Python, create the QG
index on a Collection, reopen the Collection read-only, and call
`collection.query_internal_ids("v", vectors, topk=10, param=query_param)`.
Pass a contiguous FP32 vector or a matrix of queries.

### Evaluating the QG index

Run `build/bin/recall qg-search.yaml` for recall and
`build/bin/bench qg-search.yaml` for throughput. The shared configuration
below reads dense FP32 queries from `id;v0 v1 ...` rows. For recall, supply
one ground-truth row `id;key0 key1 ...` per query in the same order.

```yaml
IndexCommon:
  IndexConfig: '{"index_type":"kQuantizedGraph","data_type":"DT_FP32","dimension":128,"metric_type":"kL2sq"}'
  IndexPath: index.zvec
  QueryFile: queries.txt
  GroundTruthFile: neighbors.txt
  TopK: 10
  CompareById: true
  RecallThreadCount: 1
  BenchThreadCount: 1
  BenchIterCount: 10000
QueryConfig:
  QueryParam: '{"index_type":"kQuantizedGraph","capacity":128,"max_expansions":96}'
```

Set the dimension and metric to match the index. Use `kCosine` with original
FP32 queries for cosine. `QueryConfig.QueryParams` accepts a sequence of query
JSON strings to measure multiple settings against the same open index.

## COHERE experiments

### Getting COHERE Data

Please download the COHERE 10M dataset to cohere_large_10m as follows:

```bash
... ...           
neighbors.parquet    
shuffle_train-00-of-10.parquet     
shuffle_train-01-of-10.parquet          
shuffle_train-02-of-10.parquet  
shuffle_train-03-of-10.parquet 
shuffle_train-04-of-10.parquet  
shuffle_train-05-of-10.parquet 
shuffle_train-06-of-10.parquet
shuffle_train-07-of-10.parquet
shuffle_train-08-of-10.parquet
shuffle_train-09-of-10.parquet
scalar_labels.parquet     
test.parquet      
```

For convenience, we prepared a docker image with cohere bench datasets: registry.cn-hongkong.cr.aliyuncs.com/zvec/cohere-bench-data.

You can run a container as follows:

```bash
docker run -it --net=host -d -e DEBUG_MODE=true  --user root --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /home/zvec/:/home/zvec/  -w /home/zvec --name=cohere_bench zvec-registry.cn-hongkong.cr.aliyuncs.com/zvec/cohere-bench-data:0.0.1 bash

docker exec -it cohere_bench bash
```

The datasets locate at /tmp/cohere/

### Preparing Environment

Clone code and init:

```bash
git clone git@github.com:alibaba/zvec.git
cd zvec
git submodule update --init
```

Configure and build the targets as described above.

### Converting Dataset

The conversion script reads `./cohere/10m` and writes `./10m.output` by default.
Adjust its `input_dir` and `output_dir` for your dataset location, then run:

```bash
mkdir -p 10m.output
python3 benchmarks/vector/convert_cohere_parquet.py
```

Convert vector data to binary formatted file.

```bash
build/bin/txt2vecs \
  --input=10m.output/cohere_768_10m_vector.train.txt \
  --output=10m.output/cohere_train_vector_10m.zvec.vecs --dimension=768
```

We've also prepared preprocessed binary formatted files, which can be found in the container below:

```bash
root@iZj6caifjouj5yu8xgsiysZ:/home/zvec# ls -al /tmp/cohere/*zvec 
/tmp/cohere/cohere_large_10m_zvec:
total 30204572
drwxr-xr-x 2 root root        4096 Feb  5 13:12 .
drwxr-xr-x 6 root root        4096 Feb  6 03:38 ..
-rw-r--r-- 1 root root     8664837 Feb  5 13:06 cohere_test_vector_10m.1000.new.txt
-rw-r--r-- 1 root root 30920004295 Feb  5 13:04 cohere_train_vector_10m.new.zvec.vecs
-rw-r--r-- 1 root root      792835 Feb  5 13:05 neighbors.txt

/tmp/cohere/cohere_medium_1m_zvec:
total 3028688
drwxr-xr-x 2 root root       4096 Feb  5 13:14 .
drwxr-xr-x 6 root root       4096 Feb  6 03:38 ..
-rw-r--r-- 1 root root    8661108 Feb  5 13:07 cohere_test_vector_1m.1000.new.txt
-rw-r--r-- 1 root root 3092004295 Feb  5 13:08 cohere_train_vector_1m.new.zvec.vecs
-rw-r--r-- 1 root root     692969 Feb  5 13:08 neighbors.txt
```

### Preparing Bench Config

Save as `build.yaml`, adjusting dataset and index paths for your environment.

```yaml
BuilderCommon:
    BuilderClass: HnswStreamer
    BuildFile: /tmp/cohere/cohere_large_10m_zvec/cohere_train_vector_10m.zvec.vecs
    NeedTrain: true 
    TrainFile: /tmp/cohere/cohere_large_10m_zvec/cohere_train_vector_10m.zvec.vecs
    DumpPath:  /home/zvec/bench/config/cohere_train_vector_10m.dump.index
    IndexPath: /home/zvec/bench/config/cohere_train_vector_10m.index

    ConverterName: CosineInt8Converter
    MetricName: Cosine

    ThreadCount: 16

BuilderParams: 
    proxima.general.builder.thread_count: !!int 16
    proxima.hnsw.builder.thread_count: !!int 16
```

Save as `search.yaml` for `recall_original` and `bench_original`.

```yaml
SearcherCommon:
    SearcherClass: HnswStreamer
    IndexPath: /home/zvec/bench/config/cohere_train_vector_10m.index
    TopK: 1,10,50,100 
    QueryFile: /tmp/cohere/cohere_large_10m_zvec/cohere_test_vector_1000.new.txt
    QueryType: float 
    QueryFirstSep: ";" 
    QuerySecondSep: " "
    GroundTruthFile: /tmp/cohere/cohere_large_10m_zvec/neighbors.txt
    RecallThreadCount: 1
    BenchThreadCount: 16 
    BenchIterCount: 1000000000 
    CompareById: true

SearcherParams: 
    proxima.hnsw.streamer.ef: !!int 250
```

### Building Index

Conduct Build

```bash
build/bin/local_builder_original ./build.yaml
```

### Performing Bench

Conduct Recall

```bash
build/bin/recall_original ./search.yaml
```

Conduct Bench

```bash
build/bin/bench_original ./search.yaml
```
