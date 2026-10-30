"""Step 8 stage metrics, revision 2. stages.py (sha256 f8c835b3…) is kept unchanged as history; only LIFT changes.

  lift  banana root origin >= LIFT_M above its position after control 1, with fingertip contact and NO table or bowl contact,
        on LIFT_POSTS consecutive post samples (0.2 s of simulation on both robots), all at or after the grasp control.
        The stage control is the last of those samples; over_bowl and later stages must come at or after it.
Grasp, over_bowl (r = 0.081 m), open, landed, win, attempts, M2 and the M5 helpers are unchanged.
"""
import numpy as np
from release_gate_v5_1 import replay
from stages import LIFT_M, OVER_BOWL_R_M, STAGES, P0_CONTROLS_PER_S, wilson, mcnemar_exact, rates, paired  # noqa: F401

LIFT_POSTS = 3


def stages(path):
    r = replay(path)
    rows, where = r['rows'], r['where']
    closed = [x['command'] > .5 for x in rows]
    touch = [bool(x['post']['contact']) for x in rows]
    rise = where[:, 0, 2] - where[0, 0, 2]
    free = [touch[k] and not x['post']['table_contact'] and not x['post']['bowl_contact'] and rise[k] >= LIFT_M for k, x in enumerate(rows)]
    dxy = np.linalg.norm(where[:, 0, :2] - where[:, 1, :2], axis=1)
    first = lambda ks, ok: next((k for k in ks if ok(k)), None)
    attempts, i, n = [], 0, len(rows)
    while i < n:
        if not closed[i]:
            i += 1
            continue
        s = i
        while i < n and closed[i]:
            i += 1
        edge = i if i < n else None
        hit = {'grasp': first(range(s + 2, i), lambda k: touch[k - 2] and touch[k - 1] and touch[k])}
        g = hit['grasp']
        hit['lift'] = None if g is None else first(range(g + LIFT_POSTS - 1, i), lambda k: all(free[k - j] for j in range(LIFT_POSTS)))
        hit['over_bowl'] = None if hit['lift'] is None else first(range(hit['lift'], i), lambda k: touch[k] and dxy[k] <= OVER_BOWL_R_M)
        hit['open'] = edge if hit['over_bowl'] is not None and edge in r['opens'] else None
        hit['landed'] = r['landed'].get(edge) if hit['open'] is not None else None
        hit['win'] = r['win'] if hit['landed'] is not None and r['qualifying_open'] == edge else None
        depth = next((j for j, st in enumerate(STAGES) if hit[st] is None), len(STAGES))
        attempts.append({'closed_from': rows[s]['step'], 'edge': rows[edge]['step'] if edge is not None else None, 'depth': depth,
                         **{st: rows[hit[st]]['step'] for st in STAGES[:depth]}})
    best = max(attempts, key=lambda a: a['depth'], default={'depth': 0})
    out = {'robot': r['robot'], 'rows': n, 'attempts': len(attempts), 'furthest': STAGES[best['depth'] - 1] if best['depth'] else None,
           **{st: best['depth'] > j for j, st in enumerate(STAGES)}, 'best_attempt': best}
    if r['win'] is not None:  # M2
        q = rows[r['qualifying_open']]['step']
        out.update(qualifying_open_control=q, time_to_win_P0_s=round(q / P0_CONTROLS_PER_S[r['robot']], 3),
                   win_control=rows[r['win']]['step'], win_simulation_s=round(rows[r['win']]['simulation_s'], 3))
    return out
