"""Score step 7 checkpoint selection (A on layouts 3000-3007) with the locked step 8 scorer, unchanged.

python score_select_step7.py --out DIR --stamp "$(date -u)" ROOT [ROOT ...]
Registers one extra phase, "select", in score_step8.PHASES (cases step7_select_A_{P0,c500,c1000,c1500,c2000}_l300{0-7}_s0_v1
from experiments/bowl_step7/run_select_v1.sh; pairs: each candidate vs P0) and runs score_step8.main. Every verdict, flag,
interval and pairing comes from the locked code; this file adds only the case pattern, the D2 rule and the review set.

D2 (EXPERIMENT_REPORT §22.6): P1 = the candidate with the most v5.1 first-rest wins out of 8; ties go to the earlier
checkpoint; if no candidate beats P0's count, D5 applies (best candidate goes on, labelled "P1, no A-validation
improvement"). S3: the binding counts are the reviewer-adjudicated ones (every v5.1 pass and every flagged fail reviewed).
The raw-count choice written here is provisional until the reviewer's verdicts are applied with --adjudicated.
Verdicts: a v5.1 pass gets OK (= CONFIRM) or REJECT (= DISAGREE: not a genuine v5.1 first-rest win, in the step 8 sense; a
regrasp after the first rest leaves the win standing, unlike the seed-1 training REJECT); a fail gets AGREE_FAIL or MISSED_WIN.
A choice is final only when adjudicated and every condition has 8 scored episodes; otherwise Miodrag decides.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_step8  # locked; imported, never modified

CANDIDATES = ('c500', 'c1000', 'c1500', 'c2000')
score_step8.PHASES['select'] = {
    'names': [re.compile(r'^step7_select_(?P<robot>A)_(?P<cond>P0|c500|c1000|c1500|c2000)_(?P<layout>l300[0-7])_s0_v1$')],
    'conditions': ('P0', *CANDIDATES), 'pairs': tuple((c, 'P0') for c in CANDIDATES)}


ALIASES = {'CONFIRM': 'OK', 'DISAGREE': 'REJECT'}
NOTES = {'selection_bias': "P1's validation count is selection-biased (the best of 4 candidates on the same 8 layouts).",
         'D5': 'If no candidate beats P0 on 3000-3007, P1 goes to step 8 labelled "P1, no A-validation improvement".'}


def choose(wins, final):
    """D2 on {condition: wins}. max() keeps the first of equal keys, and CANDIDATES is in checkpoint order."""
    best = max(CANDIDATES, key=lambda c: wins[c])
    return {'P1': best, 'wins': wins[best], 'P0_wins': wins['P0'], 'improves_on_P0': wins[best] > wins['P0'],
            'label': 'P1' if wins[best] > wins['P0'] else 'P1, no A-validation improvement', 'final': final}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--stamp', default='')
    ap.add_argument('--adjudicated', help='JSON {case: verdict} from the reviewer (S3): passes OK|CONFIRM or REJECT|DISAGREE '
                    '(not a genuine first-rest win), fails AGREE_FAIL|MISSED_WIN')
    ap.add_argument('roots', nargs='+')
    a = ap.parse_args(argv)
    r = score_step8.main(['--phase', 'select', '--out', a.out, '--stamp', a.stamp, *a.roots])
    per = r['per_condition']
    complete = all(per.get(f'A/{c}', {}).get('n') == 8 for c in ('P0', *CANDIDATES))
    raw = {c: per.get(f'A/{c}', {}).get('wins', 0) for c in ('P0', *CANDIDATES)}
    review = [{'case': e['case'], 'condition': e['condition'], 'layout': e['layout'], 'v5_1_pass': e['v5_1_pass'],
               'win_control': e['win_control'], 'review_flags': e['review_flags'], 'live_v4_pass': bool(e['live_v4']['pass'])}
              for e in r['episodes'] if e['scored'] and (e['v5_1_pass'] or e['review_flags'])]
    out = {'schema': 'step7-select-v1', 'stamp_utc': a.stamp, 'complete_8_of_8_each': complete, 'problems': r['problems'],
           'unscored': {c: per.get(f'A/{c}', {}).get('unscored') for c in ('P0', *CANDIDATES)},
           'raw_v5_1_wins': raw, 'raw_choice_provisional': choose(raw, False), 'review_set': review, 'notes': NOTES,
           'code_sha256': {**r['code_sha256'], 'score_select_step7.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    if a.adjudicated:
        verdicts = {k: ALIASES.get(v, v) for k, v in json.loads(Path(a.adjudicated).read_text()).items()}
        cases = {e['case']: e for e in r['episodes'] if e['scored']}
        assert verdicts.keys() <= cases.keys(), sorted(verdicts.keys() - cases.keys())
        assert {v['case'] for v in review} <= verdicts.keys(), 'every pass and flagged fail needs a verdict (S3)'
        for case, v in verdicts.items():
            allowed = ('OK', 'REJECT') if cases[case]['v5_1_pass'] else ('AGREE_FAIL', 'MISSED_WIN')
            assert v in allowed, (case, v, allowed)
        adj = {c: 0 for c in ('P0', *CANDIDATES)}
        for case, e in cases.items():
            v = verdicts.get(case)
            adj[e['condition']] += (e['v5_1_pass'] and v == 'OK') or (not e['v5_1_pass'] and v == 'MISSED_WIN')
        out.update(adjudicated_from=a.adjudicated,
                   adjudicated_sha256=hashlib.sha256(Path(a.adjudicated).read_bytes()).hexdigest(),
                   adjudicated_wins=adj, choice=choose(adj, complete))
    Path(a.out, 'select-step7.json').write_text(json.dumps(out, indent=1) + '\n')
    return out


if __name__ == '__main__':
    o = main()
    print(json.dumps({k: o[k] for k in ('complete_8_of_8_each', 'raw_v5_1_wins', 'raw_choice_provisional', 'problems')
                      + (('adjudicated_wins', 'choice') if 'choice' in o else ())}, indent=1))
