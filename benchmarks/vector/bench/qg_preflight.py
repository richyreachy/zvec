#!/usr/bin/env python3
# Copyright 2026-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small native-entry correctness checks. Never a performance measurement."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
from qg_bench import save, run, require, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True)
    p.add_argument('--config-dir', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for ds in ('sift', 'gist', 'glove'):
        c = json.loads((Path(a.config_dir) / (ds + '.json')).read_text())
        folder = out / ds
        folder.mkdir()
        n, nq, d = 128, 24, c['dimension']
        rng = np.random.default_rng(20261119)
        x = rng.normal(size=(n, d)).astype(np.float32)
        q = rng.normal(size=(nq, d)).astype(np.float32)
        if c['metric'] == 'cosine':
            x = (x / np.linalg.norm(x.astype(np.float64), axis=1)[:, None]).astype(np.float32)
            q = (q / np.linalg.norm(q.astype(np.float64), axis=1)[:, None]).astype(np.float32)
        c.update(train_count=n, query_count=nq, provider_prefix=str(folder / 'provider'),
                 queries_file=str(folder / 'queries.f32'), output_dir=str(folder))
        # Nonidentity mapping verifies that output IDs are external corpus rows.
        ids = rng.permutation(n).astype('<u8')
        x[ids].astype('<f4').tofile(c['provider_prefix'] + '.vectors')
        ids.tofile(c['provider_prefix'] + '.ids')
        np.arange(0, n * (n - 1) + 1, n - 1, dtype='<u8').tofile(c['provider_prefix'] + '.offsets')
        np.concatenate([np.delete(np.arange(n, dtype='<u4'), i) for i in range(n)]).tofile(c['provider_prefix'] + '.edges')
        q.tofile(c['queries_file'])
        profile = c['profiles']['n64']
        c['profiles'] = {'n64': profile}
        profile['index_prefix'] = str(folder / 'n64')
        profile['encoding'].update(threads=2, training_samples=64)
        profile['router']['anchors'] = 16
        profile['router']['projected_dimension'] = min(32, d)
        c['points'] = [dict(name='capacity-' + str(cap), phase='migration', target=1., profile='n64',
                           capacity=cap, scan_limit=0, search=dict(margin=1e6, bucket_width=0.,
                           max_expansions=512, route_beam=1, uncertainty_bonus=0.)) for cap in (10, 32, 128)]
        cfg = folder / 'config.json'
        save(cfg, c)
        run(a.binary, 'build', cfg, 'n64', log=folder / 'build.log')
        run(a.binary, 'query', cfg, 'n64', folder / 'query', log=folder / 'query.log')
        if c['metric'] == 'cosine':
            distances = 1 - q.astype(np.float64) @ x.astype(np.float64).T / (
                np.linalg.norm(q, axis=1)[:, None] * np.linalg.norm(x, axis=1)[None, :])
        else:
            distances = np.sum((q.astype(np.float64)[:, None, :] - x.astype(np.float64)[None, :, :]) ** 2, axis=2)
        expected = np.argsort(distances, axis=1)[:, :10]
        stable = []
        for cap in (10, 32, 128):
            for r in range(5):
                prefix = folder / 'query' / f'capacity-{cap}-round{r}'
                actual = np.fromfile(str(prefix) + '.ids', dtype='<u8').reshape(nq, 10)
                scores = np.fromfile(str(prefix) + '.scores', dtype='<f4').reshape(nq, 10)
                expanded = np.fromfile(str(prefix) + '.expanded', dtype='<u8')
                require(np.array_equal(actual, expected), ds + ': brute-force IDs differ')
                require(np.allclose(scores, np.take_along_axis(distances, expected, axis=1), rtol=3e-5, atol=3e-6), ds + ': exact distances differ')
                require(np.all(expanded > 0), 'not QG')
                stable.append(fingerprint(str(prefix) + '.ids')['sha256'])
        require(len(set(stable)) == 1, 'capacity/repetition instability on exhaustive small graph')
        negative = []
        for kind in ('legacy', 'codec', 'views'):
            bad = copy.deepcopy(c)
            if kind == 'legacy':
                bad['use_quantized_graph'] = True
            elif kind == 'codec':
                bad['profiles']['n64']['encoding']['codec'] = 'typo'
            else:
                bad['profiles']['n64']['router']['rotation_views'] = 4
            badfile = folder / (kind + '.json')
            save(badfile, bad)
            result = subprocess.run([a.binary, 'query', str(badfile), 'n64', str(folder / 'rejected')],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            require(result.returncode != 0, 'invalid config accepted: ' + kind)
            negative.append(dict(case=kind, returncode=result.returncode, output=result.stdout[-1000:]))
        records.append(dict(dataset=ds, full_train_used=False, preflight_only=True, performance_eligible=False,
                            brute_force_ids=True, exact_distances=True, nonidentity_id_mapping=True,
                            capacities=[10, 32, 128], repetitions_identical=True, negative_cases=negative))
    save(out / 'validation.json', dict(binary=fingerprint(a.binary), checks=records, passed=True))
    print('Native build/save/load/query preflight passed; no preflight QPS published.', flush=True)


if __name__ == '__main__':
    main()
