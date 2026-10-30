"""After-phase drift canary: compare a rerun with the before-phase episode on the same layout, field by field.

  compare_canary.py A <before strict-metric.json> <canary strict-metric.json> <out.json>
  compare_canary.py B <before gate_trace.jsonl> <canary gate_trace.jsonl> <out.json>
Recorded for the reviewer, not a gate: repeated P0 requests are not bit-reproducible (noise floor 4-12 mm,
EXPERIMENT_REPORT §20), so a difference alone does not prove drift. v5 outcomes are scored offline with the locked scorer.
"""
import json
import sys
from pathlib import Path


def rows(kind, path):
    if kind == 'A':
        d = json.loads(Path(path).read_text())
        return d['trace'], {'initial_poses_world': d['initial_poses_world'], 'v4_stable_success_s': d['stable_success_s'],
                            'release_events': d['release_events']}
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()], {}


def flat(x, prefix=''):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(flat(v, f'{prefix}{k}.'))
        return out
    return {prefix[:-1]: x}


def numbers(v):
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return None
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, list):
        out = []
        for x in v:
            n = numbers(x)
            if n is None:
                return None
            out += n
        return out
    return None


kind, before, canary, out = sys.argv[1:]
(rb, mb), (rc, mc) = rows(kind, before), rows(kind, canary)
fields = {}
for i, (a, b) in enumerate(zip(rb, rc)):
    fa, fb = flat(a), flat(b)
    for k in sorted(set(fa) | set(fb)):
        f = fields.setdefault(k, {'identical': True, 'first_diff_row': None, 'max_abs_diff': 0.})
        if k in ('wall_s',):
            continue
        va, vb = fa.get(k), fb.get(k)
        if va == vb:
            continue
        if f['first_diff_row'] is None:
            f['first_diff_row'] = i
        f['identical'] = False
        na, nb = numbers(va), numbers(vb)
        d = max(abs(x - y) for x, y in zip(na, nb)) if na and nb and len(na) == len(nb) else None
        f['max_abs_diff'] = max(f['max_abs_diff'], d) if f['max_abs_diff'] is not None and d is not None else None
report = {'kind': kind, 'before': str(before), 'canary': str(canary), 'rows': [len(rb), len(rc)],
          'all_fields_identical': len(rb) == len(rc) and all(f['identical'] for k, f in fields.items() if k != 'wall_s'),
          'fields': fields, 'metadata_identical': {k: mb[k] == mc.get(k) for k in mb},
          'v4_stable_success_s': [mb.get('v4_stable_success_s'), mc.get('v4_stable_success_s')] if kind == 'A' else None,
          'note': 'Not a gate. v5 outcomes for both episodes are scored offline with the locked scorer.'}
Path(out).write_text(json.dumps(report, indent=1, default=str) + '\n')
print('CANARY_COMPARED', kind, 'rows', report['rows'], 'all_fields_identical', report['all_fields_identical'], flush=True)
