"""Stage 1 of the painted shadow check: shadow_capture_B.py unchanged except that a collision-free visual copy of A is
in the scene and, at each capture row, the exterior view is additionally rendered with A painted at the virtual-A
joints (paint_A.py). Both the actual B exterior and the painted exterior are saved. No policy server, no inference.
--case step6_B_paint_probe_v2 runs rows 1-24 only and sweeps renders per visibility switch (development probe)."""
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path
ROOT=Path.home()/'robot-transfer/bowl';REPO=ROOT/'RoboLab'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(ROOT/'experiments/bowl_step5'));os.chdir(REPO)
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--case',required=True);AppLauncher.add_app_launcher_args(p);args=p.parse_args()
assert args.case in ('step6_B_paint_probe_v2','step6_B_paint_capture_v1');PROBE=args.case.startswith('step6_B_paint_probe')
out=ROOT/'runs'/args.case;out.mkdir(parents=True,exist_ok=False)
args.enable_cameras=True;app=AppLauncher(args).app
env=video=trace=capture=None;rows=[];metric=[];complete=False
REPEAT=12;HOLD_EXTRA=24;LOGICAL_STEPS=24 if PROBE else 88;REQUEST_ROWS=list(range(8,LOGICAL_STEPS+1,8))
# Capture rows hold their target 24 extra ticks: B settles to ~0.5 mm by ~30 ticks after a step (mapped_tracking_B_v1),
# while the 12th tick still carries 2-6 mm of controller lag (temporal_B_v1).
SCHEDULE=[(r,k) for r in range(LOGICAL_STEPS) for k in range(REPEAT+(HOLD_EXTRA if r+1 in REQUEST_ROWS else 0))]
PHYSICAL_STEPS=len(SCHEDULE);assert PHYSICAL_STEPS==LOGICAL_STEPS*REPEAT+len(REQUEST_ROWS)*HOLD_EXTRA==(360 if PROBE else 1320)
# Row 0 is excluded: the first B chunk already matched A, and reset-time camera data needs a step to refresh.
VALID={'banana_m':.005,'canonical_m':.002,'canonical_rad':.01}
try:
 import numpy as np,torch,h5py,cv2
 from robolab.constants import set_output_dir
 from robolab.core.environments.config import parse_env_cfg
 from robolab.core.environments.factory import get_envs
 from robolab.core.environments.runtime import create_env
 from robolab.core.observations.observation_utils import unpack_image_obs,unpack_viewport_cams
 from robolab.core.utils.video_utils import VideoWriter
 from robolab.registrations.kinova.auto_env_registrations_jointpos import auto_register_kinova_envs
 from robolab.core.sensors.contact_sensor_utils import create_contact_sensors,get_contact_sensor_with_order
 from robolab.core.world.world_state import get_world
 from robolab.robots.kinova_gen3 import GRIPPER_JOINT_NAME
 from robolab.core.task.instruction import primary_instruction,instruction_text
 from policies.gr00t.client import GR00TDroidJointposClient,DROID_EEF_ROTATION_CORRECT
 from rigid_chain import RigidChain,pose_matrix,rigid_inverse
 from local_ik import pose_error,solve,physical_limits
 from paint_A import add_painted_A,Painter,EXTERIOR

 def arr(x):return x.detach().cpu().numpy().copy()
 def tensor(x):return torch.as_tensor(np.asarray(x),device=args.device,dtype=torch.float32)
 def save(n,x):(out/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
 def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 folder=REPO/'output/step5_training_layout2002_v4/BananaInBowlUniformInitPose10cmTask'
 refpath=folder/'run_0.hdf5';metricpath=folder/'strict-metric.json';npzpath=ROOT/'datasets/training_layout2002_v4.npz'
 syncpath=folder/'synchronized.hdf5';SYNC_SHA='0d2096b1939d15ff086c1c7de8c6ce638f5a29ed5ab46a7e2942ecf5cd9eafc1'
 graphpath=ROOT/'assets/report/evidence/2026-10-21/bowl-step6/kinematics_export_v1/joint_graphs.json'
 expected=json.loads((Path(__file__).with_name('A_replay_inputs.json')).read_text())
 paths={'recording':refpath,'metric':metricpath,'dataset':npzpath,'graphs':graphpath}
 for k,path in paths.items():assert digest(path)==expected[k],k
 assert digest(syncpath)==SYNC_SHA,'synchronized'
 reference=json.loads(metricpath.read_text());dataset=np.load(npzpath);actions=dataset['action.executed_joint_gripper']
 assert dataset['complete'] and actions.shape==(301,8) and np.isfinite(actions).all()
 initial={};refstates={}
 with h5py.File(refpath) as f:
  g=f['data/demo_0'];assert np.array_equal(g['actions'][:],actions)
  g['initial_state'].visititems(lambda n,x:initial.update({n:x[:]}) if isinstance(x,h5py.Dataset) else None)
  g['states'].visititems(lambda n,x:refstates.update({n:x[:]}) if isinstance(x,h5py.Dataset) else None)
 with h5py.File(syncpath) as f:
  syncq=f['state.joint_position'][:];syncfresh=f['fresh_model_request'][:];sync_instruction=f.attrs['instruction']
 # A's measured post-state after action r is exactly the observation P0 received at row r+1.
 measured=refstates['articulation/robot/joint_position'][:,:7]
 assert np.array_equal(measured[:LOGICAL_STEPS],syncq[1:LOGICAL_STEPS+1])
 assert all(syncfresh[r] for r in REQUEST_ROWS) and np.flatnonzero(syncfresh[:LOGICAL_STEPS+1]).tolist()==list(range(0,LOGICAL_STEPS+1,8))
 # Pick by verified pre-first-action evidence, not arbitrary reset index.
 candidates=[]
 for i in range(len(initial['rigid_object/banana/root_pose'])):
  if all(np.array_equal(initial['rigid_object/'+n+'/root_pose'][i],np.asarray(reference['initial_poses_world'][n][:7],dtype=np.float32)) and np.array_equal(initial['rigid_object/'+n+'/root_velocity'][i],np.asarray(reference['initial_poses_world'][n][7:],dtype=np.float32)) for n in ('banana','bowl')):candidates.append(i)
 assert len(candidates)==1,candidates
 idx=candidates[0]
 graphs=json.loads(graphpath.read_text());A=RigidChain(graphs['A']);B=RigidChain(graphs['B']);Alim=physical_limits(graphs['A']);Blim=physical_limits(graphs['B'])
 setup={'frames':ROOT/'runs/step6_fixed_frames_cpu_v1/frame_proposal.json','initial_ik':ROOT/'runs/step6_fixed_frames_cpu_v1/ik_results.json','A_inventory':ROOT/'runs/step6_A_tracking_v3/inventory.json'}
 setup_hash={'frames':'ac3c64b604f85e3295d0f256eecd850a9a6ec9985f30300d87c0b2641d5f41f1','initial_ik':'a5e57b96f7aa50445252cd78e7b8bba45c8f26ee56e1533be3bb84e2265a42dc','A_inventory':'f26eb999aa5652ee442953bc6dbb84a1f5180ab3ee0d0ceaeab35f1161bfa4c7'}
 for k,path in setup.items():assert digest(path)==setup_hash[k],k
 frames=json.loads(setup['frames'].read_text());initial_ik=json.loads(setup['initial_ik'].read_text())
 assert initial_ik[0]['name']=='matched_A_home' and initial_ik[0]['forward']['success']
 initialB=np.asarray(initial_ik[0]['forward']['q']);CA,CB=(np.asarray(frames['raw_T_canonical'][r]) for r in ('A','B'))
 Acam=np.asarray(frames['A_raw_T_camera_opengl']);virtual_seed=np.asarray(json.loads(setup['A_inventory'].read_text())['default_arm'])
 Aroot=pose_matrix(initial['articulation/robot/root_pose'][idx,:3],initial['articulation/robot/root_pose'][idx,3:7]);assert np.allclose(Aroot,frames['root_world']['A'],atol=1e-7,rtol=0)
 set_output_dir(str(out));auto_register_kinova_envs(task_dirs=['randomize_initial_pose'],task='BananaInBowlUniformInitPose10cmTask')
 names=get_envs(task='BananaInBowlTask');assert names==['BananaInBowlUniformInitPose10cmTaskKinovaJointPosition'],names
 cfg=parse_env_cfg(names[0],device=args.device,seed=2002,num_envs=1,use_fabric=True)
 assert cfg.episode_length_s==50
 original_success=cfg.terminations.success;rawfunc=original_success.func;rawparams=dict(original_success.params)
 cfg.terminations.success=None # Unscored capture.
 cfg.episode_length_s=90 # 1320 controls = 88 s; original 50 s cannot fit the 12x dwell plus capture holds.
 for n in list(vars(cfg.scene)):
  if n.startswith('gripper__'):setattr(cfg.scene,n,None)
 tips={s:f'robotiq_85_{s}_finger_tip_link' for s in ('left','right')}
 cfg.contact_gripper={s:f'{{ENV_REGEX_NS}}/robot/{n}' for s,n in tips.items()}
 cfg.contact_gripper['gripper']=['left','right'];create_contact_sensors(cfg)
 cfg.scene.robot.init_state.joint_pos.update({f'joint_{i+1}':float(v) for i,v in enumerate(initialB)})
 cam=frames['B_camera_cfg'];cfg.scene.wrist_cam.prim_path=cam['prim_path']
 cfg.scene.wrist_cam.offset.pos=tuple(cam['offset']['pos']);cfg.scene.wrist_cam.offset.rot=tuple(cam['offset']['rot']);cfg.scene.wrist_cam.offset.convention='opengl'
 for n in ('width','height'):setattr(cfg.scene.wrist_cam,n,cam[n])
 for n in ('focal_length','horizontal_aperture','vertical_aperture'):setattr(cfg.scene.wrist_cam.spawn,n,cam[n])
 for n in ('wrist_cam','over_shoulder_left_camera'):getattr(cfg.scene,n).update_latest_camera_pose=True
 add_painted_A(cfg)
 env,cfg=create_env(cfg,instruction_type='default');obs,_=env.reset()
 _,instruction=primary_instruction(cfg.instruction);assert instruction_text(instruction)==sync_instruction,(instruction,sync_instruction)
 assert np.isclose(env.step_dt,1/15) and env.action_manager.total_action_dim==8
 robot=env.scene['robot'];jn=list(robot.data.joint_names);ids=[jn.index(n) for n in B.names];grip_id=jn.index(GRIPPER_JOINT_NAME)
 assert [jn[i] for i in env.action_manager.get_term('arm')._joint_ids]==B.names
 assert np.allclose(dataset['state.joint_position'][0],initial['articulation/robot/joint_position'][idx,:7],atol=1e-7,rtol=0)
 # Direct state restore, including velocities, followed by forward/render without a physics step.
 for n in ('banana','bowl','table'):
  obj=env.scene[n];state=np.r_[initial['rigid_object/'+n+'/root_pose'][idx],initial['rigid_object/'+n+'/root_velocity'][idx]]
  obj.write_root_state_to_sim(tensor(state)[None])
 root=arr(robot.data.root_state_w)[0].copy();root[7:]=0
 assert np.allclose(pose_matrix(root[:3],root[3:7]),frames['root_world']['B'],atol=1e-7,rtol=0)
 robot.write_root_state_to_sim(tensor(root)[None])
 q0=arr(robot.data.joint_pos)[0].copy();q0[:]=0;q0[ids]=initialB;qd0=np.zeros_like(q0)
 robot.write_joint_state_to_sim(tensor(q0)[None],tensor(qd0)[None]);robot.set_joint_position_target(tensor(q0)[None])
 env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.0)
 env.sim.render() # Render only; no warmup action or simulated-time advance.
 assert int(env.episode_length_buf[0])==0
 restored={'selected_initial_row':idx,'physics_warmup_steps':0,'robot_q':arr(robot.data.joint_pos)[0].tolist(),'robot_qd':arr(robot.data.joint_vel)[0].tolist(),'robot_root':arr(robot.data.root_state_w)[0].tolist(),'objects':{n:arr(env.scene[n].data.root_state_w)[0].tolist() for n in ('banana','bowl','table')}}
 save('initialization.json',restored)
 painter=Painter(env);save('painter.json',painter.inventory)
 assert np.allclose(restored['robot_q'],q0,atol=1e-6,rtol=0) and np.allclose(restored['robot_qd'],qd0,atol=1e-6,rtol=0)
 assert np.allclose(restored['robot_root'],root,atol=1e-6,rtol=0)
 for n in ('banana','bowl','table'):assert np.allclose(restored['objects'][n],np.r_[initial['rigid_object/'+n+'/root_pose'][idx],initial['rigid_object/'+n+'/root_velocity'][idx]],atol=1e-6,rtol=0)
 save('protocol.json',{'scope':'Stage 1 of the matched shadow test. Captures B observations only; no P0 inference here, no policy output executed, no score, no adapter freeze.',
  'environment':names[0],'input_sha256':{**{k:digest(v) for k,v in paths.items()},'synchronized':SYNC_SHA},'setup_sha256':setup_hash,
  'steps':PHYSICAL_STEPS,'logical_steps':LOGICAL_STEPS,'repeat_factor':REPEAT,'capture_hold_extra_ticks':HOLD_EXTRA,'hz':15,'physical_duration_s':PHYSICAL_STEPS/15,'physics_warmup_steps':0,'selected_initial_row':idx,
  'native_B_controller_unchanged':True,'native_B_gripper_unchanged':True,'success_termination_disabled':True,'original_timeout_s':50,'diagnostic_timeout_s':90,
  'arm_target':'A2002 MEASURED post-state after logical action r (equals P0 observation at row r+1), not the commanded action: A lags its commands by up to ~21 mm in approach, and the goal is pose-matched images. FK -> A canonical world -> B raw root -> IK seeded from measured B q each tick -> unchanged 0.04 rad/tick limiter',
  'gripper':'A2002 executed gripper command for logical action r, repeated 12 times; native B binary target',
  'observation_mapping':'identical to run_fixed_bridge_temporal.py: B measured canonical -> virtual A local IK with continuous seed from A home; eef_9d from virtual FK with DROID_EEF_ROTATION_CORRECT; gripper = clip(robotiq_85_left_knuckle_joint/0.8,0,1); remounted wrist camera',
  'request_rows':REQUEST_ROWS,'capture_point':'observation returned by the last of the 36 controls (12 + 24 hold) for logical action r-1, i.e. B settled at A2002 row r pose',
  'validity_predeclared':{'banana_xyz_vs_A2002_max_m':VALID['banana_m'],'canonical_position_vs_A2002_max_m':VALID['canonical_m'],'canonical_rotation_vs_A2002_max_rad':VALID['canonical_rad']},
  'excluded':'Request row 0 (reset-time camera refresh; first chunk already matched A in live runs). Rows after 88.',
  'failure_bounds':{'measured_joint_soft_margin_rad':.03,'root_translation_m':1e-4,'root_rotation_rad':1e-4,'FK_translation_m':1e-4,'FK_rotation_rad':1e-3,'IK_raw_translation_m':1e-4,'IK_raw_rotation_rad':1e-3,'camera_translation_m':1e-4,'camera_rotation_rad':1e-3},
  'stop':'Exactly 1320 physical steps (360 in the probe) unless a guard/IK/termination fails. Preserve partial run, no automatic retry.',
  'painting':'At each capture row, after saving the unpainted request: teleport the visual A copy to virtual-A q with gripper closure*pi/4, hide B visuals, render the exterior, restore. Painted exterior saved separately; nothing is executed.'})
 tool_id=list(robot.data.body_names).index(graphs['B']['raw_tool_body'].split('/')[-1])
 video=VideoWriter(str(out/'three_views.mp4'),15);soft=arr(robot.data.soft_joint_pos_limits)[0];command=initialB.copy()
 trace=(out/'ik_trace.jsonl').open('x');capture=h5py.File(out/'captures.hdf5','x');capture.attrs['instruction']=sync_instruction
 root0=pose_matrix(root[:3],root[3:7]);world=get_world(env)
 def state():
  q=arr(robot.data.joint_pos)[0];qd=arr(robot.data.joint_vel)[0];rs=arr(robot.data.root_state_w)[0]
  rt=pose_matrix(rs[:3],rs[3:7]);ts=arr(robot.data.body_state_w)[0,tool_id];tool=rigid_inverse(rt)@pose_matrix(ts[:3],ts[3:7])
  objects=np.stack([arr(env.scene[n].data.root_state_w)[0] for n in ('banana','bowl','table')])
  assert all(np.isfinite(x).all() for x in (q,qd,rs,tool,objects))
  assert (q>=soft[:,0]-.03).all() and (q<=soft[:,1]+.03).all()
  re=pose_error(root0,rt);fe=pose_error(B.fk(q[ids]),tool)
  assert np.linalg.norm(re[:3])<1e-4 and np.linalg.norm(re[3:])<1e-4
  assert np.linalg.norm(fe[:3])<1e-4 and np.linalg.norm(fe[3:])<1e-3
  return q,qd,rs,rt,tool,objects
 def contacts():
  forces={};reversed_flags={}
  for side,name in tips.items():
   sensor=env.scene[side+'__banana'];force=sensor.data.force_matrix_w
   assert force is not None and tuple(force.shape)==(1,1,1,3)
   assert list(sensor.body_names)==[name] and sensor.cfg.filter_prim_paths_expr==[cfg.scene.banana.prim_path]
   forces['banana_on_'+side+'_tip']=arr(force)[0,0,0].tolist()
  for n in ('bowl','table'):
   sensor,rev=get_contact_sensor_with_order(env.scene,'banana',n);forces[n]=arr(sensor.data.force_matrix_w)[0].tolist();reversed_flags[n]=bool(rev)
  boxes={n:arr(world.get_bbox(n)[0][0]).tolist() for n in ('banana','bowl','table')}
  assert all(np.isfinite(v).all() for v in list(forces.values())+list(boxes.values()))
  return {'forces_world':forces,'force_sensor_reversed':reversed_flags,'boxes_env_local':boxes}
 tick12={};last_painted=np.zeros((720,1280,3),np.uint8)+64
 for i,(logical_index,repeat_index) in enumerate(SCHEDULE):
  last=repeat_index==REPEAT+(HOLD_EXTRA if logical_index+1 in REQUEST_ROWS else 0)-1
  qA=measured[logical_index].astype(float);grip=float(actions[logical_index,7]);assert grip in (0,1)
  preq,preqd,prers,prert,pretool,preobjects=state();precontacts=contacts()
  target=rigid_inverse(prert)@Aroot@A.fk(qA)@CA@rigid_inverse(CB)
  solution=solve(B,target,preq[ids],Blim)
  record={'step':i+1,'direction':'A2002_measured_to_B','logical_index_0based':logical_index,'repeat_index_0based':repeat_index,'seed':preq[ids].tolist(),'target':target.tolist(),**solution}
  trace.write(json.dumps(record)+'\n');trace.flush()
  if not solution['success']:save('failed_solve.json',record)
  assert solution['success'],(i+1,solution['reason'])
  goal=np.asarray(solution['q']);previous=command.copy();command+=np.clip(goal-command,-.04,.04)
  assert np.isfinite(command).all() and (command>=Blim[:,0]).all() and (command<=Blim[:,1]).all()
  assert np.max(np.abs(command-previous))<=.04000001
  executed=np.r_[command,grip]
  obs,_,term,trunc,_=env.step(tensor(executed)[None])
  assert int(env.episode_length_buf[0])==i+1 and not bool(term.any()) and not bool(trunc.any()),'Unexpected reset/termination'
  q,qd,rs,rt,tool,objects=state();postcontacts=contacts()
  # Virtual-A observation exactly as the live bridge maps it, every tick for seed continuity.
  targetA=rigid_inverse(Aroot)@rt@tool@CB@rigid_inverse(CA)
  inverse=solve(A,targetA,virtual_seed,Alim)
  vrecord={'step':i+1,'direction':'observation_B_to_A','seed':virtual_seed.tolist(),'target':targetA.tolist(),**inverse}
  trace.write(json.dumps(vrecord)+'\n');trace.flush()
  if not inverse['success']:save('failed_solve.json',vrecord)
  assert inverse['success'],(i+1,'observation',inverse['reason'])
  virtual_seed=np.asarray(inverse['q']);virtual_tool=A.fk(virtual_seed)
  raw=bool(rawfunc(env,**rawparams)[0])
  metric.append({'step':i+1,'logical_index_0based':logical_index,'repeat_index_0based':repeat_index,'simulation_s':(i+1)/15,'upstream_predicate_not_success_score':raw,'command':grip,'closure_left_knuckle':float(q[grip_id]/.8),'pre':precontacts,'post':postcontacts})
  referencecanonical=Aroot@A.fk(qA)@CA;actualcanonical=rt@tool@CB
  banana_error=objects[0,:3]-refstates['rigid_object/banana/root_pose'][logical_index,:3]
  if repeat_index==REPEAT-1:tick12[logical_index+1]=pose_error(referencecanonical,actualcanonical)
  rows.append({'step':i+1,'logical_index_0based':logical_index,'repeat_index_0based':repeat_index,'physical_time_s':(i+1)/15,'A_measured_target':qA,'action_B':executed,'pre_q':preq,'pre_tool_root':pretool,
   'q':q,'qd':qd,'root':rs,'tool_root':tool,'objects':objects,'B_goal':goal,'previous_command':previous,'target_B':target,
   'limited':bool(np.max(np.abs(goal-previous))>.04),'gripper_closure':q[grip_id]/.8,'virtual_A_q':virtual_seed,
   'reference_A_canonical':referencecanonical,'actual_B_canonical':actualcanonical,
   'reference_A_tracking_error':pose_error(referencecanonical,actualcanonical),'banana_reference_error_m':banana_error})
  images=unpack_image_obs(obs)
  if last and logical_index+1 in REQUEST_ROWS:
   r=logical_index+1
   camdata=env.scene['wrist_cam'].data;actualcam=pose_matrix(arr(camdata.pos_w)[0],arr(camdata.quat_w_opengl)[0])
   ce=pose_error(rt@tool@CB@rigid_inverse(CA)@Acam,actualcam);vce=pose_error(Aroot@virtual_tool@Acam,actualcam)
   assert np.linalg.norm(ce[:3])<1e-4 and np.linalg.norm(ce[3:])<1e-3,('camera',r,ce.tolist())
   assert np.linalg.norm(vce[:3])<3e-4 and np.linalg.norm(vce[3:])<.002,('virtual camera',r,vce.tolist())
   assert np.allclose(arr(camdata.intrinsic_matrices)[0],frames['expected_camera_K'],atol=1e-4,rtol=0)
   eef=np.r_[virtual_tool[:3,3],(virtual_tool[:3,:3]@DROID_EEF_ROTATION_CORRECT)[:2,:].reshape(-1)].astype(np.float32)
   closure=np.asarray([np.clip(q[grip_id]/.8,0,1)],dtype=np.float32)
   extracted={'external_image':images['over_shoulder_left_camera'],'wrist_image':images['wrist_cam'],
              'joint_position':virtual_seed.astype(np.float32),'gripper_position':closure,'eef_9d':eef}
   # _pack_request uses no instance state; calling it unbound keeps this stage server-free.
   packed=GR00TDroidJointposClient._pack_request(None,extracted,instruction)
   assert packed['annotation.language.language_instruction']==[sync_instruction]
   pe=pose_error(referencecanonical,actualcanonical)
   checks={'banana_error_m':float(np.linalg.norm(banana_error)),'canonical_position_error_m':float(np.linalg.norm(pe[:3])),'canonical_rotation_error_rad':float(np.linalg.norm(pe[3:]))}
   valid=checks['banana_error_m']<=VALID['banana_m'] and checks['canonical_position_error_m']<=VALID['canonical_m'] and checks['canonical_rotation_error_rad']<=VALID['canonical_rad']
   group=capture.create_group(str(r))
   for k,v in packed.items():
    if k.startswith(('video.','state.')):group.create_dataset(k,data=v,compression='lzf')
   for k,v in {'A2002_joint_position':syncq[r],'B_q':q,'virtual_A_q':virtual_seed,'B_canonical_world':actualcanonical,'A2002_canonical_world':referencecanonical,
               'camera_error':ce,'virtual_camera_error':vce,'objects_banana_bowl_table':objects}.items():group.create_dataset(k,data=np.asarray(v))
   group.attrs.update({'request_row':r,'physical_step':i+1,'valid':bool(valid),**checks,
                       'tick12_canonical_position_error_m':float(np.linalg.norm(tick12[r][:3])),'tick12_canonical_rotation_error_rad':float(np.linalg.norm(tick12[r][3:]))})
   # Sanity: a forced re-read of the exterior camera must equal the observation image before any painting.
   painter.camera.update(0.0,force_recompute=True);reread=painter.camera.data.output['rgb'][0,...,:3].detach().cpu().numpy()
   group.attrs['exterior_reread_max_abs_diff']=int(np.abs(reread.astype(int)-images[EXTERIOR].astype(int)).max())
   if PROBE:
    written=painter.pose(virtual_seed,float(closure[0]));painter._show(True);sweep=[]
    for k in range(8):env.sim.render();painter.camera.update(0.0,force_recompute=True);sweep.append(painter.camera.data.output['rgb'][0,...,:3].detach().cpu().numpy().copy())
    painter.park();painter._show(False);restore=[]
    for k in range(8):env.sim.render();painter.camera.update(0.0,force_recompute=True);restore.append(painter.camera.data.output['rgb'][0,...,:3].detach().cpu().numpy().copy())
    env.scene['wrist_cam'].update(0.0,force_recompute=True);group.create_dataset('probe_wrist_after_restore',data=env.scene['wrist_cam'].data.output['rgb'][0,...,:3].detach().cpu().numpy(),compression='lzf')
    group.create_dataset('probe_painted_sweep',data=np.stack(sweep),compression='lzf');group.create_dataset('probe_restore_sweep',data=np.stack(restore),compression='lzf')
    painted=sweep[painter.frames-1]
   else:painted,written=painter.paint(virtual_seed,float(closure[0]))
   ppack=GR00TDroidJointposClient._pack_request(None,{**extracted,'external_image':painted},instruction)
   group.create_dataset('painted.video.exterior_image_1_left',data=ppack['video.exterior_image_1_left'],compression='lzf')
   group.create_dataset('painted_A_joint_position',data=written)
   for name,img in (('actual_exterior',images[EXTERIOR]),('painted_exterior',painted)):assert cv2.imwrite(str(out/f'row{r:03d}_{name}.png'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
   last_painted=painted
   capture.flush()
   print('SHADOW_CAPTURE',r,json.dumps({'valid':bool(valid),**checks}),flush=True)
  panels=[]
  for label,img in zip(('B EXTERIOR (actual)','P0 EXTERIOR (painted, last capture)','WRIST'),(images['over_shoulder_left_camera'],last_painted,images['wrist_cam'])):
   assert img.dtype==np.uint8 and (img.std()>1 or img is last_painted)
   frame=cv2.resize(img,(640,360));cv2.putText(frame,f'B PAINT CAPTURE {i+1}/{PHYSICAL_STEPS} ROW {logical_index+1}.{repeat_index} {label}',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(frame)
  video.write(cv2.hconcat(panels))
  if (i+1)%50==0:print('B_SHADOW_CAPTURE_PROGRESS',i+1,flush=True)
 complete=True
 save('result.json',{'complete':True,'steps':PHYSICAL_STEPS,'logical_steps':LOGICAL_STEPS,'repeat_factor':REPEAT,'task_score':None,'captured_rows':sorted(int(k) for k in capture),
  'valid_rows':sorted(int(k) for k in capture if capture[k].attrs['valid']),'rate_limited_rows':sum(r['limited'] for r in rows),
  'max_banana_reference_error_m':float(max(np.linalg.norm(r['banana_reference_error_m']) for r in rows)),
  'interpretation':'Observation capture only. Policy queries happen in stage 2 against these saved inputs; nothing here is a P0 trial or success estimate.'})
 print('B_SHADOW_CAPTURE_COMPLETE',flush=True)
except BaseException:
 error=traceback.format_exc();(out/'failure.txt').write_text(error);print(error,flush=True);raise
finally:
 if rows:np.savez_compressed(out/'tracking.npz',complete=np.asarray(complete),**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
 if metric:(out/'contact_trace.json').write_text(json.dumps(metric,indent=2)+'\n')
 if capture is not None:capture.close()
 if trace is not None:trace.close()
 if video is not None:video.release()
 if env is not None:env.close()
 app.close()
