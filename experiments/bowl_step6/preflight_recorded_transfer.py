"""Bounded CPU kinematic preflight; optimistic command-following seed, no physics."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from rigid_chain import RigidChain,rigid_inverse
from local_ik import solve,physical_limits,pose_error
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
paths={'graphs':args.root/'assets/report/evidence/2026-10-21/bowl-step6/kinematics_export_v1/joint_graphs.json',
 'frames':args.root/'runs/step6_fixed_frames_cpu_v1/frame_proposal.json','initial':args.root/'runs/step6_fixed_frames_cpu_v1/ik_results.json',
 'A_recording':args.root/'datasets/training_layout2002_v4.npz'}
expected={'graphs':'51d10a57316bed35b79792211eea4eeff530b2aec79f9d1af9cf14b94216b66e','frames':'ac3c64b604f85e3295d0f256eecd850a9a6ec9985f30300d87c0b2641d5f41f1','initial':'a5e57b96f7aa50445252cd78e7b8bba45c8f26ee56e1533be3bb84e2265a42dc','A_recording':'4cfcf48999cfac4e0b1883be5cf052de9581a9c1bd6fc31a5b5a14aaaf690d50'}
for k,v in paths.items():assert hashlib.sha256(v.read_bytes()).hexdigest()==expected[k], k
graphs=json.loads(paths['graphs'].read_text());frames=json.loads(paths['frames'].read_text());data=np.load(paths['A_recording'])
A=RigidChain(graphs['A']);B=RigidChain(graphs['B']);limits=physical_limits(graphs['B']);al=physical_limits(graphs['A'])
CA,CB=[np.asarray(frames['raw_T_canonical'][r]) for r in ('A','B')]
# Fixed robot roots are identity for this preflight; physics replay must verify them separately.
assert all(np.allclose(frames['root_world'][r],np.eye(4)) for r in ('A','B'))
actions=data['action.executed_joint_gripper'];assert data['complete'] and actions.shape==(301,8)
assert np.isfinite(actions).all() and np.all(actions[:,:7]>=al[:,0]) and np.all(actions[:,:7]<=al[:,1])
initial=json.loads(paths['initial'].read_text())
# Explicit reviewed matched-home solution stored by frame derivation.
assert initial[0]['name']=='matched_A_home' and initial[0]['forward']['success']
initialB=np.asarray(initial[0]['forward']['q'])
command=initialB.copy();rows=[];complete=False
try:
 with (args.output/'ik_trace.jsonl').open('w') as log:
  for i,action in enumerate(actions):
   target=A.fk(action[:7])@CA@rigid_inverse(CB)
   seed=command.copy();r=solve(B,target,seed,limits)
   log.write(json.dumps({'row':i,'seed_assumption':'previous limited command, perfect tracking; measured physics seed unavailable','seed':seed.tolist(),'target':target.tolist(),'result':r})+'\n');log.flush()
   if not r['success']:raise RuntimeError(f'IK failed at row {i}: '+r['reason'])
   goal=np.asarray(r['q']);previous=command.copy();command+=np.clip(goal-command,-.04,.04)
   assert (command>=limits[:,0]).all() and (command<=limits[:,1]).all()
   rows.append({'A_action':action,'B_goal':goal,'B_command':np.r_[command,action[7]],'target':target,'limited':bool(np.max(abs(goal-previous))>.04),
    'goal_change_rad':goal-previous,'canonical_ik_error':pose_error(target@CB,B.fk(goal)@CB),'canonical_command_error':pose_error(target@CB,B.fk(command)@CB)})
 complete=True
finally:
 if rows:np.savez_compressed(args.output/'mapping.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
 summary={'complete':complete,'rows':len(rows),'scope':'CPU feasibility only; previous limited command stands in for unavailable measured B seed. No collision/contact/dynamics/replay/task success demonstrated.',
 'input_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},'source_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['preflight_recorded_transfer.py','rigid_chain.py','local_ik.py']}}
 if rows:
  summary.update({'limited_rows':sum(r['limited'] for r in rows),'max_goal_change_rad':float(max(np.max(abs(r['goal_change_rad'])) for r in rows)),
   'max_canonical_ik_m':float(max(np.linalg.norm(r['canonical_ik_error'][:3]) for r in rows)),
   'max_canonical_command_m':float(max(np.linalg.norm(r['canonical_command_error'][:3]) for r in rows))})
 (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
