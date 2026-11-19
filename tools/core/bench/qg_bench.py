#!/usr/bin/env python3
"""Config-driven independent QG runner and auditable full-query validation.

Requires numpy, h5py; isolation additionally requires psutil and docker.
No timing from prepare, preflight, or correctness checks enters the results.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import h5py
import numpy as np


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fingerprint(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=h.hexdigest())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_config(c):
    def reject_old(x):
        if isinstance(x, dict):
            for k, v in x.items():
                require(not k.startswith('quantized_graph_') and k not in
                        ('use_quantized_graph', 'QuantizedGraph', 'QuantizedGraphRouter', 'ef_search'),
                        'legacy field in executable config: ' + k)
                reject_old(v)
        elif isinstance(x, list):
            for v in x:
                reject_old(v)
    reject_old(c)
    require(c['schema_version'] == 2 and c['engine'] == 'QuantizedGraphIndex', 'new API required')
    require(c['precision'] == 'FP32' and c['metric'] in ('l2sq', 'cosine'), 'metric/precision')
    require(c['provider_format'] == 'csr-fp32-le-v1', 'provider format')
    require(c['id_convention'] == 'internal CSR row -> ids.u64 -> original HDF5 train row', 'ID mapping')
    proto = c['protocol']
    require(proto['topk'] == 10 and proto['threads'] == 1 and proto['warmup_rounds'] == 1 and
            proto['rounds'] == 5 and proto['repetitions'] == 3, 'protocol')
    names = set()
    for name, p in c['profiles'].items():
        e, r = p['encoding'], p['router']
        require(p['name'] == name and e['rotation_views'] == r['rotation_views'], 'profile/views')
        require(e['codec'] in ('native', 'global', 'low_rank') and e['codebook'] in ('hadamard', 'gaussian'), 'enum mapping')
        require(32 <= e['max_neighbors'] <= 256 and e['max_neighbors'] % 32 == 0 and
                e['rotation_views'] in (1, 2, 4), 'build shape')
    for p in c['points']:
        require(p['name'] not in names and p['profile'] in c['profiles'], 'point identity')
        names.add(p['name'])
        require(10 <= p['capacity'] <= 1048576 and p['scan_limit'] >= 0, 'capacity/scan_limit')
        s = p['search']
        require(set(s) == {'margin', 'bucket_width', 'max_expansions', 'route_beam', 'uncertainty_bonus'}, 'search fields')
        require(all(np.isfinite(v) for v in s.values()) and (s['margin'] >= 0 or s['margin'] == -1)
                and 0 <= s['bucket_width'] <= 1 and 0 <= s['route_beam'] <= 4
                and 0 <= s['uncertainty_bonus'] <= 1 and s['max_expansions'] >= 0, 'search range')


def config(path):
    raw = Path(path).read_text()
    for name in ('BENCH_ROOT', 'ARTIFACT_ROOT'):
        if '${' + name + '}' in raw:
            raw = raw.replace('${' + name + '}', os.environ[name])
    c = json.loads(raw)
    validate_config(c)
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    resolved = out / 'resolved.json'
    if resolved.exists():
        require(json.loads(resolved.read_text()) == c, 'configuration changed: use a new ARTIFACT_ROOT')
    else:
        save(resolved, c)
    return c, resolved


def run(binary, *args, log, cpu=None):
    cmd = [str(binary), *map(str, args)]
    if cpu is not None:
        cmd = ['numactl', '--physcpubind=' + str(cpu), '--membind=0', *cmd]
    with Path(log).open('w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))


def prepare(c, resolved, binary):
    out = Path(c['output_dir'])
    prefix = c['provider_prefix']
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    for p in c['profiles'].values():
        Path(p['index_prefix']).parent.mkdir(parents=True, exist_ok=True)
    require(fingerprint(c['base_graph']['source_index'])['sha256'] == c['base_graph']['source_sha256'], 'base graph source hash')
    require(not Path(prefix + '.vectors').exists(), 'provider exists; use existing verified manifest or new output directory')
    run(binary, 'export', resolved, log=out / 'export.log')
    n, d = c['train_count'], c['dimension']
    ids = np.fromfile(prefix + '.ids', dtype='<u8')
    require(np.array_equal(np.sort(ids), np.arange(n)), 'provider IDs not a permutation')
    vectors = np.memmap(prefix + '.vectors', dtype='<f4', mode='r', shape=(n, d))
    largest, norm_error = 0., 0.
    with h5py.File(c['dataset_file'], 'r') as f:
        require(f['train'].shape == (n, d) and f['test'].shape == (c['query_count'], d), 'full dataset shape')
        # Read corpus sequentially, compare each exported internal row using external IDs.
        inverse = np.argsort(ids)
        for start in range(0, n, 4096):
            raw = f['train'][start:start + 4096].astype(np.float64)
            actual = vectors[inverse[start:start + len(raw)]].astype(np.float64)
            if c['metric'] == 'cosine':
                raw /= np.linalg.norm(raw, axis=1)[:, None]
                norm_error = max(norm_error, float(np.max(np.abs(np.linalg.norm(actual, axis=1) - 1))))
            largest = max(largest, float(np.max(np.abs(actual - raw))))
        require(largest < (2e-6 if c['metric'] == 'cosine' else 1e-7), 'provider vectors differ from dataset: ' + str(largest))
        q = f['test'][:].astype(np.float64)
        if c['metric'] == 'cosine':
            q /= np.linalg.norm(q, axis=1)[:, None]
        np.asarray(q, dtype='<f4').tofile(c['queries_file'])
    files = [c['dataset_file'], c['base_graph']['source_index'], c['queries_file']]
    files += [prefix + ext for ext in ('.vectors', '.offsets', '.edges', '.ids')]
    save(out / 'provider.json', dict(verified=True, contract=provider_contract(c), checked_vectors=n, max_vector_error=largest,
         max_norm_error=norm_error, old_qg_snapshot_reused=False, base_graph_reused=True,
         files=[fingerprint(p) for p in files]))


def provider_contract(c):
    return {k: c[k] for k in ('provider_prefix', 'provider_format', 'dimension', 'train_count',
                              'metric', 'precision', 'id_convention', 'dataset_file', 'queries_file', 'query_count')}


def verify_provider(c):
    manifest = json.loads((Path(c['output_dir']) / 'provider.json').read_text())
    require(manifest['verified'] and manifest['contract'] == provider_contract(c), 'provider contract mismatch')
    verify_files(manifest['files'])


def build(c, resolved, binary):
    out = Path(c['output_dir'])
    verify_provider(c)
    for name, p in c['profiles'].items():
        require(not Path(p['index_prefix'] + '.qg').exists(), 'refusing to overwrite QG index')
        run(binary, 'build', resolved, name, log=out / ('build-' + name + '.log'))
        save(out / ('index-' + name + '.json'), dict(profile=p, source=source(binary),
             files=[fingerprint(p['index_prefix'] + ext) for ext in ('.qg', '.router')],
             provider_manifest=fingerprint(out / 'provider.json')))


def verify_files(files):
    for f in files:
        require(fingerprint(f['path']) == f, 'fingerprint mismatch: ' + f['path'])


def source(binary):
    repo = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True)
    patch = git('diff', 'HEAD', '--', 'src', 'tools/core', 'cmake', 'CMakeLists.txt')
    return dict(commit=git('rev-parse', 'HEAD').strip(), binary=fingerprint(binary),
                patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
                dirty=bool(git('status', '--porcelain')), timing='native_reset_search_sort_map_copy')


def validate_round(c, prefix, truth, thresholds, train, queries):
    n, k = c['query_count'], c['protocol']['topk']
    ids = np.fromfile(str(prefix) + '.ids', dtype='<u8').reshape(n, k)
    scores = np.fromfile(str(prefix) + '.scores', dtype='<f4').reshape(n, k)
    ns = np.fromfile(str(prefix) + '.ns', dtype='<u8')
    expanded = np.fromfile(str(prefix) + '.expanded', dtype='<u8')
    comparisons = np.fromfile(str(prefix) + '.comparisons', dtype='<u8')
    require(ns.shape == expanded.shape == comparisons.shape == (n,) and np.all(ns > 0)
            and np.all(expanded > 0) and np.all(comparisons >= k), 'timing or QG evidence')
    require(np.all(ids < c['train_count']) and np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0), 'result IDs')
    require(np.isfinite(scores).all(), 'nonfinite exact scores')
    independent = np.empty((n, k), dtype=np.float64)
    for start in range(0, n, 512):
        a = train[ids[start:start + 512]].astype(np.float64)
        q = queries[start:start + len(a)].astype(np.float64)
        if c['metric'] == 'cosine':
            independent[start:start + len(a)] = 1 - np.einsum('nkd,nd->nk', a, q) / (
                np.linalg.norm(a, axis=2) * np.linalg.norm(q, axis=1)[:, None])
        else:
            independent[start:start + len(a)] = np.sum((a - q[:, None, :]) ** 2, axis=2)
    require(np.allclose(scores, independent, rtol=3e-5, atol=3e-6), 'independent exact distance mismatch')
    ann_distance = np.sqrt(np.maximum(independent, 0)) if c['metric'] == 'l2sq' else independent
    strict = np.mean(np.any(ids[:, :, None] == truth[:, None, :], axis=2))
    ann = np.mean(ann_distance <= thresholds[:, None] + .001)
    return dict(qps=1e9 / float(ns.mean()), mean_query_us=float(ns.mean()) / 1000,
                ann_recall=float(ann), strict_id_recall=float(strict),
                max_exact_error=float(np.max(np.abs(scores - independent))),
                ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                scores_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),
                min_qg_expanded=int(expanded.min()), mean_qg_expanded=float(expanded.mean()),
                mean_exact_comparisons=float(comparisons.mean()),
                files=[fingerprint(str(prefix) + ext) for ext in ('.ids', '.scores', '.ns', '.expanded', '.comparisons')])


def measure(c, resolved, binary):
    out = Path(c['output_dir'])
    isolation = json.loads(Path(os.environ['QG_ISOLATION_MANIFEST']).read_text())
    require('restored_at' not in isolation, 'isolation has already ended')
    verify_provider(c)
    for name, p in c['profiles'].items():
        manifest = json.loads((out / ('index-' + name + '.json')).read_text())
        require(manifest['profile'] == p, 'index build/config mismatch')
        require(manifest['provider_manifest'] == fingerprint(out / 'provider.json'), 'index/provider binding mismatch')
        verify_files(manifest['files'])
    with h5py.File(c['dataset_file'], 'r') as f:
        train, queries = f['train'][:], f['test'][:]
        truth, thresholds = f['neighbors'][:, :10], f['distances'][:, 9]
    metadata = source(binary)
    save(out / 'measurement-source.json', metadata)
    rows = []
    # Migration first; only after all frozen points, run the predeclared equal-budget additions.
    for phase in ('migration', 'tuning'):
        active = copy.deepcopy(c)
        active['points'] = [p for p in c['points'] if p['phase'] == phase]
        active_path = out / (phase + '.json')
        save(active_path, active)
        for repeat in range(3):
            for name in c['profiles']:
                dest = out / 'raw' / phase / str(repeat) / name
                require(not dest.exists(), 'measurement directory exists; never mix batches')
                check_quiet()
                run(binary, 'query', active_path, name, dest,
                    log=out / f'query-{phase}-{repeat}-{name}.log', cpu=c['protocol']['cpu'])
                for point in active['points']:
                    if point['profile'] != name:
                        continue
                    rounds = [validate_round(c, dest / (point['name'] + '-round' + str(r)),
                                             truth, thresholds, train, queries) for r in range(5)]
                    require(len({r['ids_sha256'] for r in rounds}) == 1 and
                            len({r['scores_sha256'] for r in rounds}) == 1, 'query instability within repeat')
                    rows.append(dict(point=point, repeat=repeat, rounds=rounds,
                                     qps=max(r['qps'] for r in rounds)))
                save(out / 'measurements.json', rows)
    summary = []
    for p in c['points']:
        entries = sorted([r for r in rows if r['point']['name'] == p['name']], key=lambda r:r['repeat'])
        require(len(entries) == 3, 'missing repetitions')
        require(len({r['rounds'][0]['ids_sha256'] for r in entries}) == 1, 'query instability across repeats')
        qps = float(np.median([r['qps'] for r in entries]))
        r = entries[0]['rounds'][0]
        summary.append(dict(point=p, ann_recall=r['ann_recall'], strict_id_recall=r['strict_id_recall'],
                            qps_each=[e['qps'] for e in entries], qps=qps, mean_query_us=1e6/qps,
                            ids_sha256=r['ids_sha256'], full_query_count=c['query_count']))
    winners = []
    unique = {}
    for row in summary:
        point = row['point']
        signature = json.dumps({k: point[k] for k in ('profile', 'capacity', 'scan_limit', 'search')}, sort_keys=True)
        unique.setdefault(signature, row)
    for target in c['targets']:
        eligible = [r for r in unique.values() if r['ann_recall'] >= target]
        winners.append(dict(target=target, best=max(eligible, key=lambda r:r['qps']) if eligible else None))
    save(out / 'summary.json', dict(new_api_revalidated=True, dataset=c['dataset'], source=metadata,
         points=summary, winners=winners, index_rebuild_randomness_tested=False,
         metric='median of 3 independent process repetitions, each best of 5 full-query rounds',
         independent_distance_check_outside_timer=True, old_timing_directly_comparable=False))


def check_quiet():
    import docker
    import psutil
    running = [c.id for c in docker.from_env().containers.list()
               if c.attrs['Config']['Image'].startswith('ann-benchmarks-') and c.status == 'running']
    require(not running, 'background benchmark containers still running: ' + str(running))
    compiling = [p.pid for p in psutil.process_iter(['name', 'status'])
                 if p.info['name'] in ('cc1plus', 'cc1', 'g++', 'c++', 'clang++', 'ninja', 'cmake')
                 and p.info['status'] not in (psutil.STATUS_STOPPED, psutil.STATUS_ZOMBIE)]
    require(not compiling, 'compilation still active: ' + str(compiling))


def restore(path):
    import docker
    import psutil
    audit = json.loads(Path(path).read_text())
    client = docker.from_env()
    errors = []
    for entry in reversed(audit['containers']):
        try:
            c = client.containers.get(entry['id']); c.reload()
            if c.status == 'paused':
                c.unpause()
        except docker.errors.NotFound:
            pass
        except Exception as e:
            errors.append(str(e))
    for entry in reversed(audit['processes']):
        try:
            p = psutil.Process(entry['pid'])
            if p.create_time() == entry['created']:
                p.send_signal(signal.SIGCONT)
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            errors.append(str(e))
    audit.update(restored_at=time.time(), restore_errors=errors)
    save(path, audit)
    require(not errors, str(errors))


def isolate(path, command):
    import docker
    import psutil
    require(not Path(path).exists(), 'isolation manifest already exists')
    audit = dict(started_at=time.time(), processes=[], containers=[],
                 command=command, interrupted_background_timings_invalid=True)
    save(path, audit)
    own = {os.getpid(), *[p.pid for p in psutil.Process().parents()]}
    names = {'run_all_algorithms.py', 'run_sift_all_20261117.py', 'plot_all_points_20261117.py', 'run.py'}
    child = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('isolation interrupted by signal ' + str(signum))
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        for p in psutil.process_iter(['pid', 'cmdline', 'name']):
            if p.pid in own or not (p.info['name'] or '').startswith('python'):
                continue
            cmd = p.info['cmdline'] or []
            if any(Path(a).name in names for a in cmd) and p.status() != psutil.STATUS_STOPPED:
                audit['processes'].append(dict(pid=p.pid, created=p.create_time(), command=cmd))
                save(path, audit); p.send_signal(signal.SIGSTOP)
        client = docker.from_env()
        for c in client.containers.list():
            if c.attrs['Config']['Image'].startswith('ann-benchmarks-') and c.status == 'running':
                audit['containers'].append(dict(id=c.id, image=c.attrs['Config']['Image'], command=c.attrs['Config']['Cmd']))
                save(path, audit); c.pause()
        # Also stop host plotting descendants, but never stop our own process tree.
        for entry in list(audit['processes']):
            for p in psutil.Process(entry['pid']).children(recursive=True):
                if p.pid not in own and p.status() != psutil.STATUS_STOPPED:
                    audit['processes'].append(dict(pid=p.pid, created=p.create_time(), command=p.cmdline()))
                    save(path, audit); p.send_signal(signal.SIGSTOP)
        save(path, audit)
        env = dict(os.environ, QG_ISOLATION_MANIFEST=str(Path(path).resolve()))
        child = subprocess.Popen(command, env=env, start_new_session=True)
        require(child.wait() == 0, 'isolated command failed')
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        try:
            restore(path)
        finally:
            signal.signal(signal.SIGTERM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'prepare', 'build', 'measure', 'isolate', 'restore'])
    parser.add_argument('--config')
    parser.add_argument('--binary')
    parser.add_argument('--manifest')
    a, remainder = parser.parse_known_args()
    if a.command == 'restore':
        return restore(a.manifest)
    if a.command == 'isolate':
        require(bool(remainder), 'command required after --')
        return isolate(a.manifest, remainder[1:] if remainder[0] == '--' else remainder)
    require(not remainder, 'unexpected arguments: ' + str(remainder))
    c, resolved = config(a.config)
    if a.command == 'check':
        print(c['dataset'], len(c['points']), 'valid new API configurations')
    else:
        globals()[a.command](c, resolved, Path(a.binary).resolve())


if __name__ == '__main__':
    main()
