"""v5: a measured, commanded release of a held, lifted banana that lands on the original bowl and settles there.

Changes from v4, nothing else:
- the open edge no longer needs the banana inside the bowl's geometric region, so a drop from above counts;
- after detachment the banana's first support must be the bowl (bowl contact before any table-only contact);
- after landing, table-only support invalidates (it fell off), replacing v4's geometric escape.
Arming, opening timeout, measured opening, regrasp and the 1 s stability window are v4's, unchanged.
Offline scorer only: v4 stop-on-success stays the episode stop rule and the runners stay byte-identical.
Two readings of "stays there" are reported: verdict fixed at the first stable rest (pass_first_rest), and that
plus the banana still supported in or on the bowl at the last recorded control (pass_at_end).
Run: python release_gate_v5.py <strict-metric.json | gate_trace.jsonl> ...  (needs ../bowl_step5 on sys.path)
"""
from dataclasses import dataclass
from release_gate_v4 import ReleaseGateV4


@dataclass
class ReleaseGateV5(ReleaseGateV4):
    landed: bool = False

    def invalidate(self, t, reason):
        super().invalidate(t, reason)
        self.landed = False

    def supported_region(self, sample):
        sample = super().supported_region(sample)
        sample['near_bowl'] = True  # where it ends up is judged by the landing, not by geometry at the open
        return sample

    def land(self, t, sample):
        if self.phase == 'idle' or self.detached_at is None:
            return
        off = sample['table_contact'] and not sample['bowl_contact']
        if not self.landed:
            if sample['bowl_contact']:
                self.landed = True
                self.events.append({'time_s': t, 'event': 'landed_on_bowl'})
            elif off:
                self.invalidate(t, 'landed_off_bowl')
        elif off:
            self.invalidate(t, 'left_bowl')

    def before(self, t, command, sample):
        self.land(t, sample)
        return super().before(t, command, sample)

    def after(self, t, sample, raw_success, linear_speed, angular_speed):
        passed = super().after(t, sample, raw_success, linear_speed, angular_speed)
        self.land(t, sample)  # after, so the detachment sample itself counts; a pass needs bowl support, which land() never revokes
        return passed

    def status(self):
        return {**super().status(), 'landed_on_bowl': self.landed}


B_RELEASE_TIMEOUT_S = .5 + 24 / 15  # as the B port of v4: A's 0.5 s + B's measured full gripper travel


def score(path, gate_cls=ReleaseGateV5):
    """Replay one saved trace. A: RoboLab strict-metric.json; B: gate_trace.jsonl beside tracking.npz."""
    import json
    from pathlib import Path
    from rim_support import support_signals
    path = Path(path)
    robot = 'B' if path.name == 'gate_trace.jsonl' else 'A'
    if robot == 'B':
        import numpy as np
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        objects = np.load(path.with_name('tracking.npz'))['objects_banana_bowl']
        gate = gate_cls(release_timeout_s=B_RELEASE_TIMEOUT_S)
        def where(i):  # state before control i = after control i-1
            o = objects[max(i - 1, 0)]
            return o[0, :3].astype(float), o[1, :3].astype(float)
        extend = dict
    else:
        rows = json.loads(path.read_text())['trace']
        gate = gate_cls()
        def where(i):
            p = rows[i]['pre']['poses_env_local']
            return [*map(float, p['banana'][0])], [*map(float, p['bowl'][0])]
        def extend(s):  # historical A captures predate recorded support signals; same reviewed recompute as v4's replay
            s = dict(s)
            s.update(support_signals(s['boxes_env_local']['banana'], s['boxes_env_local']['bowl'], s['forces_world']['bowl'],
                                     s.get('force_sensor_reversed', {}).get('bowl', False)))
            return s
    opens, passed_at = [], None
    for i, row in enumerate(rows):
        idle = gate.phase == 'idle'
        pre = extend(row['pre'])
        t = rows[i - 1]['simulation_s'] if i else 2 * row['simulation_s'] - rows[1]['simulation_s']  # pre time = previous post time, as recorded
        gate.before(t, row['command'], pre)
        if idle and gate.phase == 'opening':
            banana, bowl = where(i)
            opens.append({'control': row['step'], 'time_s': t,
                          'dxy_root_m': round(float(((banana[0] - bowl[0]) ** 2 + (banana[1] - bowl[1]) ** 2) ** .5), 4),
                          'centre_above_bowl_top_m': round(pre['centroid_height_above_bowl_m'], 4),
                          'touching_bowl': pre['bowl_contact'], 'touching_table': pre['table_contact']})
        post = extend(row['post'])
        if gate.after(row['simulation_s'], post, row['upstream_predicate'], row['linear_m_s'], row['angular_rad_s']) and passed_at is None:
            passed_at, passed_t, n_events = row['step'], row['simulation_s'], len(gate.events)
    end = post['bowl_support'] and (rows[-1]['upstream_predicate'] or post['rim_support'])  # same supported-placement test as the window
    if passed_at:
        reason = None
    elif not any(e['event'] == 'lift_armed' for e in gate.events):
        reason = 'never_lifted'
    else:
        last = [e for e in gate.events if e['event'] == 'invalidated']
        reason = (last[-1]['reason'] if last and (not opens or last[-1]['time_s'] >= opens[-1]['time_s'])
                  else 'not_stable_by_end' if opens else 'no_open_while_held')
    qualifying = [o for o in opens if passed_at and o['time_s'] < passed_t][-1:] or [None]
    return {'robot': robot, 'pass': passed_at is not None, 'pass_first_rest': passed_at is not None, 'success_control': passed_at,
            'pass_at_end': passed_at is not None and bool(end), 'end_in_or_on_bowl': bool(end), 'end_fingertip_contact': post['contact'],
            'qualifying_open': qualifying[0], 'opens': opens, 'fail_reason': reason, 'rows': len(rows),
            'events': gate.events[:n_events] if passed_at else gate.events, 'after_first_rest': gate.events[n_events:] if passed_at else []}


if __name__ == '__main__':
    import json, sys
    print(json.dumps({p: score(p) for p in sys.argv[1:]}, indent=1))
