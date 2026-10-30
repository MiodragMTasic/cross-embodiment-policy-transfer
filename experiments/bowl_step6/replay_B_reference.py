"""One unscored B replay diagnostic using all301 A2002 commands. No policy/model or fitting."""
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path
ROOT=Path.home()/'robot-transfer/bowl';REPO=ROOT/'RoboLab'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(ROOT/'experiments/bowl_step5'));os.chdir(REPO)
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--case',required=True);AppLauncher.add_app_launcher_args(p);args=p.parse_args()
assert args.case=='step6_B_recorded_replay_v1'
out=ROOT/'runs'/args.case;out.mkdir(parents=True,exist_ok=False)
args.enable_cameras=True;app=AppLauncher(args).app
env=video=trace=None;rows=[];metric=[];complete=False
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
 from robolab.core.task.conditionals import object_in_container
 from rigid_chain import RigidChain,pose_matrix,rigid_inverse
 from local_ik import pose_error,solve,physical_limits

 def arr(x):return x.detach().cpu().numpy().copy()
 def tensor(x):return torch.as_tensor(np.asarray(x),device=args.device,dtype=torch.float32)
 def save(n,x):(out/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
 def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 folder=REPO/'output/step5_training_layout2002_v4/BananaInBowlUniformInitPose10cmTask'
 refpath=folder/'run_0.hdf5';metricpath=folder/'strict-metric.json';npzpath=ROOT/'datasets/training_layout2002_v4.npz'
 graphpath=ROOT/'assets/report/evidence/2026-10-21/bowl-step6/kinematics_export_v1/joint_graphs.json'
 expected=json.loads((Path(__file__).with_name('A_replay_inputs.json')).read_text())
 paths={'recording':refpath,'metric':metricpath,'dataset':npzpath,'graphs':graphpath}
 for k,path in paths.items():assert digest(path)==expected[k],k
 reference=json.loads(metricpath.read_text());dataset=np.load(npzpath);actions=dataset['action.executed_joint_gripper']
 assert dataset['complete'] and actions.shape==(301,8) and np.isfinite(actions).all()
 initial={};refstates={}
 with h5py.File(refpath) as f:
  g=f['data/demo_0'];assert np.array_equal(g['actions'][:],actions)
  g['initial_state'].visititems(lambda n,x:initial.update({n:x[:]}) if isinstance(x,h5py.Dataset) else None)
  g['states'].visititems(lambda n,x:refstates.update({n:x[:]}) if isinstance(x,h5py.Dataset) else None)
 # Pick by verified pre-first-action evidence, not arbitrary reset index.
 candidates=[]
 for i in range(len(initial['rigid_object/banana/root_pose'])):
  if all(np.array_equal(initial['rigid_object/'+n+'/root_pose'][i],np.asarray(reference['initial_poses_world'][n][:7],dtype=np.float32)) and np.array_equal(initial['rigid_object/'+n+'/root_velocity'][i],np.asarray(reference['initial_poses_world'][n][7:],dtype=np.float32)) for n in ('banana','bowl')):candidates.append(i)
 assert len(candidates)==1,candidates
 idx=candidates[0]
 graphs=json.loads(graphpath.read_text());A=RigidChain(graphs['A']);B=RigidChain(graphs['B']);Blim=physical_limits(graphs['B'])
 setup={'frames':ROOT/'runs/step6_fixed_frames_cpu_v1/frame_proposal.json','initial_ik':ROOT/'runs/step6_fixed_frames_cpu_v1/ik_results.json'}
 setup_hash={'frames':'ac3c64b604f85e3295d0f256eecd850a9a6ec9985f30300d87c0b2641d5f41f1','initial_ik':'a5e57b96f7aa50445252cd78e7b8bba45c8f26ee56e1533be3bb84e2265a42dc'}
 for k,path in setup.items():assert digest(path)==setup_hash[k],k
 frames=json.loads(setup['frames'].read_text());initial_ik=json.loads(setup['initial_ik'].read_text())
 assert initial_ik[0]['name']=='matched_A_home' and initial_ik[0]['forward']['success']
 initialB=np.asarray(initial_ik[0]['forward']['q']);CA,CB=(np.asarray(frames['raw_T_canonical'][r]) for r in ('A','B'))
 Aroot=pose_matrix(initial['articulation/robot/root_pose'][idx,:3],initial['articulation/robot/root_pose'][idx,3:7]);assert np.allclose(Aroot,frames['root_world']['A'],atol=1e-7,rtol=0)
 set_output_dir(str(out));auto_register_kinova_envs(task_dirs=['randomize_initial_pose'],task='BananaInBowlUniformInitPose10cmTask')
 names=get_envs(task='BananaInBowlTask');assert names==['BananaInBowlUniformInitPose10cmTaskKinovaJointPosition'],names
 cfg=parse_env_cfg(names[0],device=args.device,seed=2002,num_envs=1,use_fabric=True)
 assert cfg.episode_length_s==50
 original_success=cfg.terminations.success;rawfunc=original_success.func;rawparams=dict(original_success.params)
 cfg.terminations.success=None # Collect full301 rows; explicitly unscored diagnostic.
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
 env,cfg=create_env(cfg,instruction_type='default');obs,_=env.reset()
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
 assert np.allclose(restored['robot_q'],q0,atol=1e-6,rtol=0) and np.allclose(restored['robot_qd'],qd0,atol=1e-6,rtol=0)
 assert np.allclose(restored['robot_root'],root,atol=1e-6,rtol=0)
 for n in ('banana','bowl','table'):assert np.allclose(restored['objects'][n],np.r_[initial['rigid_object/'+n+'/root_pose'][idx],initial['rigid_object/'+n+'/root_velocity'][idx]],atol=1e-6,rtol=0)
 save('protocol.json',{'scope':'Unscored prerecorded B action-mapping/dynamics diagnostic. No P0 inference, policy success, tuning, or adapter freeze.',
  'environment':names[0],'input_sha256':{k:digest(v) for k,v in paths.items()},'setup_sha256':setup_hash,
  'steps':301,'hz':15,'physics_warmup_steps':0,'selected_initial_row':idx,'all_object_states_identical_to_A':True,
  'native_B_controller_unchanged':True,'native_B_gripper_unchanged':True,'success_termination_disabled':True,'native_timeout_s':50,
  'action_mapping':'A recorded joint FK -> A canonical world -> B raw root -> IK seeded from measured B q each tick -> unchanged previous-command 0.04rad/tick limiter',
  'gripper':'Original A binary sequence at same15Hz, no hold extension; B native six-joint binary target',
  'scoring':None,'gripper_observability':'B two distal fingertip sensors; differs from native A single finger sensor. Descriptive forces only, not a validated comparable success score.',
  'failure_bounds':{'measured_joint_soft_margin_rad':.03,'root_translation_m':1e-4,'root_rotation_rad':1e-4,'FK_translation_m':1e-4,'FK_rotation_rad':1e-3,'IK_raw_translation_m':1e-4,'IK_raw_rotation_rad':1e-3},
  'stop':'Exactly301 steps unless guard/IK/termination fails; preserve every partial run; no automatic retry.'})
 tool_id=list(robot.data.body_names).index(graphs['B']['raw_tool_body'].split('/')[-1])
 video=VideoWriter(str(out/'three_views.mp4'),15);soft=arr(robot.data.soft_joint_pos_limits)[0];command=initialB.copy()
 trace=(out/'ik_trace.jsonl').open('x');root0=pose_matrix(root[:3],root[3:7]);world=get_world(env)
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
 for i,action in enumerate(actions):
  preq,preqd,prers,prert,pretool,preobjects=state();precontacts=contacts()
  target=rigid_inverse(prert)@Aroot@A.fk(action[:7])@CA@rigid_inverse(CB)
  solution=solve(B,target,preq[ids],Blim)
  record={'step':i+1,'seed':preq[ids].tolist(),'target':target.tolist(),**solution}
  trace.write(json.dumps(record)+'\n');trace.flush()
  if not solution['success']:save('failed_solve.json',record)
  assert solution['success'],(i+1,solution['reason'])
  goal=np.asarray(solution['q']);previous=command.copy();command+=np.clip(goal-command,-.04,.04)
  assert np.isfinite(command).all() and (command>=Blim[:,0]).all() and (command<=Blim[:,1]).all()
  assert np.max(np.abs(command-previous))<=.04000001
  executed=np.r_[command,action[7]];assert action[7] in (0,1)
  obs,_,term,trunc,_=env.step(tensor(executed)[None])
  assert int(env.episode_length_buf[0])==i+1 and not bool(term.any()) and not bool(trunc.any()),'Unexpected reset/termination'
  q,qd,rs,rt,tool,objects=state();postcontacts=contacts()
  raw=bool(rawfunc(env,**rawparams)[0]);simulation_s=(i+1)/15
  metric.append({'step':i+1,'simulation_s':simulation_s,'upstream_predicate_not_success_score':raw,'command':float(action[7]),'closure_left_knuckle':float(q[grip_id]/.8),'pre':precontacts,'post':postcontacts})
  refq=refstates['articulation/robot/joint_position'][i,:7];refroot=refstates['articulation/robot/root_pose'][i]
  referencecanonical=pose_matrix(refroot[:3],refroot[3:7])@A.fk(refq)@CA;actualcanonical=rt@tool@CB
  rows.append({'step':i+1,'action_A':action,'action_B':executed,'pre_q':preq,'pre_qd':preqd,'pre_tool_root':pretool,'pre_objects':preobjects,
   'q':q,'qd':qd,'root':rs,'tool_root':tool,'objects':objects,'B_goal':goal,'previous_command':previous,'target_B':target,
   'limited':bool(np.max(np.abs(goal-previous))>.04),'gripper_closure':q[grip_id]/.8,
   'command_tracking_error':pose_error(B.fk(command)@CB,tool@CB),
   'target_tracking_error':pose_error(target@CB,tool@CB),
   'reference_A_canonical':referencecanonical,'actual_B_canonical':actualcanonical,
   'reference_A_tracking_error':pose_error(referencecanonical,actualcanonical),
   'banana_reference_error_m':objects[0,:3]-refstates['rigid_object/banana/root_pose'][i,:3]})
  images=unpack_image_obs(obs);panels=[]
  for label,img in zip(('EXTERIOR','WRIST','VIEWPORT'),(images['over_shoulder_left_camera'],images['wrist_cam'],unpack_viewport_cams(obs)['combined_image'])):
   assert img.dtype==np.uint8 and img.std()>1
   frame=cv2.resize(img,(640,360));cv2.putText(frame,f'B REPLAY {i+1} {label}',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(frame)
  video.write(cv2.hconcat(panels))
  if (i+1)%50==0:print('B_REPLAY_PROGRESS',i+1,flush=True)
 complete=True
 save('result.json',{'complete':True,'steps':301,'task_score':None,'rate_limited_rows':sum(r['limited'] for r in rows),
  'max_banana_z_m':float(max(r['objects'][0,2] for r in rows)),
  'max_command_canonical_tracking_m':float(max(np.linalg.norm(r['command_tracking_error'][:3]) for r in rows)),
  'max_reference_A_canonical_discrepancy_m':float(max(np.linalg.norm(r['reference_A_tracking_error'][:3]) for r in rows)),
  'interpretation':'Prerecorded diagnostic only. Independent visual/contact/trajectory review required. No closed-loop P0 success claim.'})
 print('B_REPLAY_COMPLETE',flush=True)
except BaseException:
 error=traceback.format_exc();(out/'failure.txt').write_text(error);print(error,flush=True);raise
finally:
 if rows:np.savez_compressed(out/'tracking.npz',complete=np.asarray(complete),**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
 if metric:(out/'contact_trace.json').write_text(json.dumps(metric,indent=2)+'\n')
 if trace is not None:trace.close()
 if video is not None:video.release()
 if env is not None:env.close()
 app.close()
