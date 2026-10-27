from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
M = R / "models68"
M.mkdir(exist_ok=False)
s = (R / "scripts/graph64_driver.cc").read_text()
begin = s.index("void search(")
end = s.index("\nint main(", begin)
body = 'void search(const std::string &path, const std::string &method, int, int, const std::string &prefix) {\n const size_t count=std::getenv("TRACE68_QUERY_COUNT")?std::stoul(std::getenv("TRACE68_QUERY_COUNT")):4096;\n check(count>0 && count<=N,"query count");\n auto index=make_index(method);ci::StorageOptions storage;storage.type=ci::StorageOptions::StorageType::kMMAP;storage.read_only=true;\n check(index->open(path,storage)==0,"open trace index");check(index->get_doc_count()==N,"count");\n std::ifstream graph(std::getenv("TRACE68_GRAPH_FILE"),std::ios::binary),vf(std::getenv("TRACE68_VECTOR_FILE"),std::ios::binary);\n uint32_t gh[8],vh[4];graph.read(reinterpret_cast<char*>(gh),32);vf.read(reinterpret_cast<char*>(vh),16);\n check(graph.good()&&vf.good()&&gh[0]==0x51474764&&gh[2]==N&&gh[3]==DIM&&vh[0]==0x51475664&&vh[2]==N&&vh[3]==DIM,"headers");\n std::vector<uint32_t>inverse(N,UINT32_MAX);std::vector<uint64_t>keys(N);\n for(size_t i=0;i<N;++i){uint32_t degree,adj[32];graph.read(reinterpret_cast<char*>(&keys[i]),8);graph.read(reinterpret_cast<char*>(&degree),4);graph.read(reinterpret_cast<char*>(adj),128);check(graph.good()&&keys[i]<N&&inverse[keys[i]]==UINT32_MAX,"keys");inverse[keys[i]]=i;}\n check(graph.peek()==EOF,"graph size");std::vector<float>vectors(N*DIM);vf.read(reinterpret_cast<char*>(vectors.data()),vectors.size()*4);check(vf.good()&&vf.peek()==EOF,"vector size");\n struct Record{uint32_t query;uint32_t ids[10];float distances[10];};static_assert(sizeof(Record)==84,"layout");std::vector<Record>output(count);\n auto param=std::make_shared<ci::HNSWQueryParam>();param->ef_search=800;param->topk=10;\n size_t self_hits=0;\n for(size_t i=0;i<count;++i){uint32_t id=(uint64_t(i)*2654435761ULL+6819)%N;auto &r=output[i];r.query=id;ci::SearchResult result;\n check(index->search({ci::DenseVector{vectors.data()+size_t(id)*DIM}},param,&result)==0,"trace search");check(result.doc_list_.size()==10,"result count");size_t j=0;std::unordered_set<uint32_t>seen;\n for(const auto &doc:result.doc_list_){check(doc.key()<N&&std::isfinite(doc.score()),"result");uint32_t target=inverse[doc.key()];check(seen.insert(target).second,"duplicate result");r.ids[j]=target;r.distances[j]=doc.score();self_hits+=target==id;++j;}}\n FILE *out=std::fopen((prefix+"_results.bin").c_str(),"wbx");check(out,"exclusive output");uint32_t h[8]={0x51474368,1,uint32_t(N),DIM,uint32_t(count),10,800,6819};check(std::fwrite(h,sizeof(h),1,out)==1&&std::fwrite(output.data(),sizeof(Record),count,out)==count,"output");check(std::fclose(out)==0,"close output");\n check(index->close()==0,"close index");std::cout<<"TRACE68 COMPLETE queries="<<count<<" self_hits="<<self_hits<<"; database proxies include self, no GT or retrieval QPS"<<std::endl;\n}\n'
s = s[:begin] + body + s[end:]
source = R / "scripts/trace68_driver.cc"
source.write_text(s)
flags = json.loads((R / "variants/bucket90_w16/metadata.json").read_text())[
    "defines"
] + ["QG_MULTI_TRACE=1"]
with (R / "logs/build_trace68_native.log").open("w") as f:
    subprocess.run(
        ["python3", str(R / "scripts/compile_variant.py"), "trace68_native", *flags],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
commands = json.loads((R / "results/build_commands.json").read_text())
oldobj = str(R / "objects/benchmark.o")
newobj = str(R / "objects/trace68_driver.o")
exe = str(R / "trace68_driver")
executed = []
c = next(x["argv"] for x in commands if "-c" in x["argv"] and oldobj in x["argv"])
c = [
    newobj
    if x == oldobj
    else str(source)
    if x == str(R / "scripts/benchmark.cc")
    else x
    for x in c
]
executed.append(c)
subprocess.run(c, check=True)
c = next(x["argv"] for x in commands if "-c" not in x["argv"] and oldobj in x["argv"])
c = [newobj if x == oldobj else exe if x == str(R / "benchmark") else x for x in c]
executed.append(c)
subprocess.run(c, check=True)
(R / "results/trace68_build_commands.json").write_text(
    json.dumps(executed, indent=2) + "\n"
)
logging.info("%s", "SETUP68 COMPLETE")
