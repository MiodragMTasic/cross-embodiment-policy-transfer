"""Retrospective development analysis only: no fresh trial or score replacement."""
import hashlib
import json
from pathlib import Path
import sys
from release_gate_v4 import ReleaseGateV4
from rim_support import support_signals

results=[]
for arg in sys.argv[1:]:
    p=Path(arg); d=json.loads(p.read_text()); g=ReleaseGateV4(); first=None
    for row in d['trace']:
        def extend(s):
            s=dict(s)
            # Reviewed pinned banana__bowl sensor for historical A captures.
            # New v4 captures explicitly record sensor order.
            reverse=s.get('force_sensor_reversed',{}).get('bowl',False)
            s.update(support_signals(s['boxes_env_local']['banana'],s['boxes_env_local']['bowl'],s['forces_world']['bowl'],reverse))
            return s
        g.before((row['step']-1)/15,row['command'],extend(row['pre']))
        passed=g.after(row['simulation_s'],extend(row['post']),row['upstream_predicate'],row['linear_m_s'],row['angular_rad_s'])
        if passed and first is None:first=row['simulation_s']
    results.append({'case':p.parent.parent.name,'source_metric_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                    'source_metric':str(p),'recorded_schema':d['schema'], 'recorded_first_pass_s':d['stable_success_s'],
                    'retrospective_v4_first_pass_s':first,'events':g.events})
print(json.dumps({'classification':'development retrospective, not independent trial; original results unchanged',
                  'historical_force_order':'Pinned A banana__bowl sensor forces on banana; confirmed source convention',
                  'code_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                 for n in ['release_gate_v3.py','release_gate_v4.py','rim_support.py','stability.py']},
                  'cases':results},indent=2))
