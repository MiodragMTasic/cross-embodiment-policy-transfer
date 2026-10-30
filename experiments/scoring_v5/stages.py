"""Step 8 stage metrics, computed the same way from A traces (strict-metric rows) and B traces (gate_trace + tracking.npz).

An attempt is one maximal run of P0 gripper command > 0.5, the open edge that ends it, and what follows from that release.
Within an attempt a stage counts only if every earlier stage occurred first, in order:
  grasp      fingertip contact on 3 consecutive closed-command controls (one shared flag: A's combined sensor, B's two tips)
  lift       banana root origin >= LIFT_M above its position after the first control, with fingertip contact
  over_bowl  banana root origin within OVER_BOWL_R_M horizontally of the bowl root origin, with fingertip contact
  open       the open edge ending the attempt is v5.1-eligible (armed, contact, closure > 0.2)
  landed     v5.1 landed_on_bowl from that release
  win        v5.1 win (first stable rest) from that release
Furthest stage = maximum over attempts. Positions are post-control root origins (A poses_env_local, B objects_banana_bowl).
"""
from math import comb
import numpy as np
from release_gate_v5_1 import replay

LIFT_M = .02
OVER_BOWL_R_M = .081  # declared: the bowl's outer half-width, 16.1 cm mesh bound / 2, rounded up
STAGES = ('grasp', 'lift', 'over_bowl', 'open', 'landed', 'win')
P0_CONTROLS_PER_S = {'A': 15, 'B': 180}  # B holds each P0 action for 12 controls at 15 controls/s


def stages(path):
    r = replay(path)
    rows, where = r['rows'], r['where']
    closed = [x['command'] > .5 for x in rows]
    touch = [bool(x['post']['contact']) for x in rows]
    rise = where[:, 0, 2] - where[0, 0, 2]
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
        hit['lift'] = None if hit['grasp'] is None else first(range(hit['grasp'], i), lambda k: touch[k] and rise[k] >= LIFT_M)
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


def wilson(k, n, z=1.959963984540054):
    """95% Wilson score interval for k successes in n."""
    if not n:
        return (0., 1.)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * ((p * (1 - p) + z * z / (4 * n)) / n) ** .5 / d
    return (round(max(0., c - h), 4), round(min(1., c + h), 4))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar (sign test) on discordant layouts: b only in the first condition, c only in the second.
    p < 0.05 needs at least 6 discordant layouts, all in one direction (5/0 gives 0.0625, 6/0 gives 0.03125)."""
    m = b + c
    return 1. if not m else min(1., 2 * sum(comb(m, j) for j in range(min(b, c) + 1)) / 2 ** m)


def rates(results):
    """Per-stage k/n with Wilson intervals over a list of stages() results."""
    n = len(results)
    return {st: {'k': sum(x[st] for x in results), 'n': n, 'wilson95': wilson(sum(x[st] for x in results), n)} for st in STAGES}


def paired(first, second, stage):
    """first/second: {layout: stages() result} on the same layouts (e.g. P1 vs P0). Delta, discordant counts, exact p."""
    layouts = sorted(first.keys() & second.keys())
    b = sum(first[l][stage] and not second[l][stage] for l in layouts)
    c = sum(second[l][stage] and not first[l][stage] for l in layouts)
    return {'stage': stage, 'layouts': len(layouts), 'delta': (b - c) / len(layouts) if layouts else None,
            'only_first': b, 'only_second': c, 'p_exact_mcnemar': mcnemar_exact(b, c)}
