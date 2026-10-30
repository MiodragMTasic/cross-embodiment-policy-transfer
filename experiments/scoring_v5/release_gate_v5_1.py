"""v5.1: v5 plus the reviewed changes of 23 October 2026; release_gate_v5.py is kept unchanged as history.

V1 the stable rest must be supported by the bowl alone: any table contact resets the 1 s window (a bowl/table straddle
   does not count). The end-of-trace placement uses the same test.
V3 landed_off_bowl and left_bowl need 3 consecutive post samples (0.2 s) of table contact without bowl contact. Landing is
   judged on post samples only, because each pre sample repeats the previous post.
V2 review_flags() lists the episodes a human must check.
Offline scorer only: v4 stop-on-success stays the episode stop rule. Primary verdict = first stable rest.
Run: python release_gate_v5_1.py <strict-metric.json | gate_trace.jsonl> ...  (needs ../bowl_step5 on sys.path)
"""
from dataclasses import dataclass
import json
from pathlib import Path
from release_gate_v4 import ReleaseGateV4
from release_gate_v5 import ReleaseGateV5, B_RELEASE_TIMEOUT_S

OFF_BOWL_POSTS = 3


@dataclass
class ReleaseGateV51(ReleaseGateV5):
    off_posts: int = 0

    def invalidate(self, t, reason):
        super().invalidate(t, reason)
        self.off_posts = 0

    def land(self, t, sample):
        if self.phase == 'idle' or self.detached_at is None:
            return
        self.off_posts = self.off_posts + 1 if sample['table_contact'] and not sample['bowl_contact'] else 0
        if not self.landed and sample['bowl_contact']:
            self.landed = True
            self.events.append({'time_s': t, 'event': 'landed_on_bowl'})
        elif self.off_posts >= OFF_BOWL_POSTS:
            self.invalidate(t, 'left_bowl' if self.landed else 'landed_off_bowl')

    def before(self, t, command, sample):
        return ReleaseGateV4.before(self, t, command, sample)  # V3: no landing check on pre samples

    def after(self, t, sample, raw_success, linear_speed, angular_speed):
        rest = dict(sample, bowl_support=sample['bowl_support'] and not sample['table_contact'])  # V1
        passed = ReleaseGateV4.after(self, t, rest, raw_success, linear_speed, angular_speed)
        self.land(t, sample)
        return passed


def load(path):
    """Rows plus post-control root positions [banana, bowl], built the same way for both robots."""
    import numpy as np
    path = Path(path)
    if path.name == 'gate_trace.jsonl':
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        return 'B', rows, np.load(path.with_name('tracking.npz'))['objects_banana_bowl'][:, :, :3].astype(float), dict
    from rim_support import support_signals
    rows = json.loads(path.read_text())['trace']
    def extend(s):  # historical A captures predate recorded support signals; same reviewed recompute as v4's replay
        s = dict(s)
        s.update(support_signals(s['boxes_env_local']['banana'], s['boxes_env_local']['bowl'], s['forces_world']['bowl'],
                                 s.get('force_sensor_reversed', {}).get('bowl', False)))
        return s
    return 'A', rows, np.array([[r['post']['poses_env_local'][n][0] for n in ('banana', 'bowl')] for r in rows], float), extend


def replay(path, gate_cls=ReleaseGateV51):
    """Per-control record of the full trace; the gate is never allowed to stop it."""
    robot, rows, where, extend = load(path)
    gate = gate_cls(release_timeout_s=B_RELEASE_TIMEOUT_S) if robot == 'B' else gate_cls()
    opens, landed, win, release, post, since = [], {}, None, None, None, 0
    for i, row in enumerate(rows):
        idle = gate.phase == 'idle'
        pre = extend(row['pre'])
        t = rows[i - 1]['simulation_s'] if i else 2 * row['simulation_s'] - rows[1]['simulation_s']  # previous post time, as recorded
        gate.before(t, row['command'], pre)
        if idle and gate.phase == 'opening':
            release, since = i, len(gate.events)
            opens.append(i)
        was = getattr(gate, 'landed', False)
        post = extend(row['post'])
        if gate.after(row['simulation_s'], post, row['upstream_predicate'], row['linear_m_s'], row['angular_rad_s']) and win is None:
            win, n_events = i, len(gate.events)
        if getattr(gate, 'landed', False) and not was:
            landed.setdefault(release, i)
    end = post['bowl_support'] and not post['table_contact'] and (rows[-1]['upstream_predicate'] or post['rim_support'])
    return {'robot': robot, 'rows': rows, 'where': where, 'opens': opens, 'landed': landed, 'win': win,
            'qualifying_open': max(o for o in opens if o <= win) if win is not None else None,
            'end_on_bowl': bool(end), 'events': gate.events, 'n_events': n_events if win is not None else len(gate.events),
            'events_since_last_open': since}


def score(path, gate_cls=ReleaseGateV51):
    r = replay(path, gate_cls)
    rows, where, events = r['rows'], r['where'], r['events'][:r['n_events']]
    def at_open(i):  # state before control i = after control i-1
        banana, bowl = where[max(i - 1, 0)]
        pre = rows[i]['pre']
        return {'control': rows[i]['step'], 'dxy_root_m': round(float(((banana[:2] - bowl[:2]) ** 2).sum() ** .5), 4),
                'centre_above_bowl_top_m': round(pre['centroid_height_above_bowl_m'], 4)}
    if r['win'] is not None:
        reason = None
    elif not any(e['event'] == 'lift_armed' for e in events):
        reason = 'never_lifted'
    else:
        last = [e for e in events[r['events_since_last_open']:] if e['event'] == 'invalidated']
        reason = last[-1]['reason'] if last else 'not_stable_by_end' if r['opens'] else 'no_open_while_held'
    return {'robot': r['robot'], 'pass': r['win'] is not None, 'success_control': rows[r['win']]['step'] if r['win'] is not None else None,
            'pass_at_end': r['win'] is not None and r['end_on_bowl'], 'end_on_bowl': r['end_on_bowl'],
            'qualifying_open': at_open(r['qualifying_open']) if r['win'] is not None else None,
            'eligible_opens': [rows[i]['step'] for i in r['opens']], 'fail_reason': reason, 'rows': len(rows),
            'events': events, 'after_first_rest': r['events'][r['n_events']:]}


def review_flags(v51, v4_pass):
    """V2: why a human must look at this episode; empty means the automatic verdict stands unreviewed."""
    return ([] + (['v5.1 pass'] if v51['pass'] else []) + (['v4 pass, v5.1 fail'] if v4_pass and not v51['pass'] else [])
            + (['fail, ends on bowl'] if not v51['pass'] and v51['end_on_bowl'] else [])
            + (['fail, had an eligible open'] if not v51['pass'] and v51['eligible_opens'] else []))


if __name__ == '__main__':
    import sys
    print(json.dumps({p: score(p) for p in sys.argv[1:]}, indent=1))
