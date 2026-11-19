"""Validate archive structure and provenance, not runtime performance."""
import csv
import json
import math
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
summary = list(csv.DictReader((root / 'summary.csv').open()))
assert len(summary) == 21
validation = json.loads((root / 'validation.json').read_text())
assert validation['new_api_revalidated'] is False
assert len(validation['audits']) == 21
for name in ['sift', 'gist', 'glove']:
    data = json.loads((root / (name + '.json')).read_text())
    assert data['measured_revision'] == 'b44f0e09+python-qg-binding'
    assert data['new_api_revalidated'] is False
    assert [p['target_ann_recall'] for p in data['presets']] == [.3, .5, .7, .85, .9, .95, .99]
    for p in data['presets']:
        profile = data['profiles'][p['profile']]
        assert profile['native_build_options']['threads'] == 1
        assert len(profile['measured_index']['sha256']) == 64
        native = p['native_api_mapping']
        assert native['validated_on_new_api'] is False
        assert native['topk'] == 10 and native['capacity_candidate'] == p['legacy_query_params']['ef_search']
        assert set(native['search_options']) == {'margin', 'bucket_width', 'max_expansions', 'route_beam', 'uncertainty_bonus'}
        r = p['same_session_best']
        assert len(r['qps_each']) == len(r['evidence']) == 3
        assert {e['repeat'] for e in r['evidence']} == {0, 1, 2}
        assert r['ann_recall'] >= p['target_ann_recall']
        assert all(math.isfinite(q) and q > 0 for q in r['qps_each'])
        assert math.isclose(median(r['qps_each']), r['qps_median'])
        assert math.isclose(1e6 / r['qps_median'], r['mean_query_time_us'])
        row = next(row for row in summary if row['dataset'] == data['dataset'] and float(row['target_ann_recall']) == p['target_ann_recall'])
        assert row['profile'] == p['profile']
        assert math.isclose(float(row['qps_median']), r['qps_median'])
        assert math.isclose(float(row['actual_ann_recall']), r['ann_recall'])
print('Validated 21 historical g9i presets; new native API revalidation remains false.')
