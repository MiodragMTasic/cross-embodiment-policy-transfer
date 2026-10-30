"""One exact-action A2002 replay control. No policy/model, fitting, or B execution."""
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path
ROOT=Path.home()/'robot-transfer/bowl';REPO=ROOT/'RoboLab'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(ROOT/'experiments/bowl_step5'));os.chdir(REPO)
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--case',required=True);AppLauncher.add_app_launcher_args(p);args=p.parse_args()
assert args.case=='step6_A_recorded_replay_v1'
out=ROOT/'runs'/args.case;out.mkdir(parents=True,exist_ok=False)
args.enable_cameras=True;app=AppLauncher(args).app
env=video=None;rows=[];metric=[];complete=False
try:
 import numpy as np,torch,h5py,cv2
 from robolab.constants import set_output_dir
 from robolab.core.environments.config import parse_env_cfg
 from robolab.core.environments.factory import get_envs
 from robolab.core.environments.runtime import create_env
 from robolab.core.observations.observation_utils import unpack_image_obs,unpack_viewport_cams
 from robolab.core.utils.video_utils import VideoWriter
 from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
 from robolab.core.task.conditionals import object_in_container
 from rigid_chain import RigidChain,pose_matrix,rigid_inverse
 from local_ik import pose_error
 from replay_release_signals_A import release_signals
 from release_gate_v4 import ReleaseGateV4

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
 graph=json.loads(graphpath.read_text())['A'];chain=RigidChain(graph)
 set_output_dir(str(out));auto_register_droid_envs(task_dirs=['randomize_initial_pose'],task='BananaInBowlUniformInitPose10cmTask')
 names=get_envs(task='BananaInBowlTask');assert names==['BananaInBowlUniformInitPose10cmTask'],names
 cfg=parse_env_cfg(names[0],device=args.device,seed=2002,num_envs=1,use_fabric=True)
 assert cfg.episode_length_s==50
 original_success=cfg.terminations.success;rawfunc=original_success.func;rawparams=dict(original_success.params)
 cfg.terminations.success=None # Collect complete301 rows; evaluate unchanged v4 separately.
 for n in ('wrist_cam','over_shoulder_left_camera'):getattr(cfg.scene,n).update_latest_camera_pose=True
 env,cfg=create_env(cfg,instruction_type='default');obs,_=env.reset()
 assert np.isclose(env.step_dt,1/15) and env.action_manager.total_action_dim==8
 robot=env.scene['robot'];jn=list(robot.data.joint_names);ids=[jn.index(n) for n in chain.names]
 assert [jn[i] for i in env.action_manager.get_term('body')._joint_ids]==chain.names
 assert np.allclose(dataset['state.joint_position'][0],initial['articulation/robot/joint_position'][idx,ids],atol=1e-7,rtol=0)
 # Direct state restore, including velocities, followed by forward/render without a physics step.
 for n in ('banana','bowl','table'):
  obj=env.scene[n];state=np.r_[initial['rigid_object/'+n+'/root_pose'][idx],initial['rigid_object/'+n+'/root_velocity'][idx]]
  obj.write_root_state_to_sim(tensor(state)[None])
 root=np.r_[initial['articulation/robot/root_pose'][idx],initial['articulation/robot/root_velocity'][idx]]
 robot.write_root_state_to_sim(tensor(root)[None])
 q0=initial['articulation/robot/joint_position'][idx];qd0=initial['articulation/robot/joint_velocity'][idx]
 robot.write_joint_state_to_sim(tensor(q0)[None],tensor(qd0)[None]);robot.set_joint_position_target(tensor(q0)[None])
 env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.0)
 env.sim.render() # Render only; no warmup action or simulated-time advance.
 assert int(env.episode_length_buf[0])==0
 restored={'selected_initial_row':idx,'physics_warmup_steps':0,'robot_q':arr(robot.data.joint_pos)[0].tolist(),'robot_qd':arr(robot.data.joint_vel)[0].tolist(),'robot_root':arr(robot.data.root_state_w)[0].tolist(),'objects':{n:arr(env.scene[n].data.root_state_w)[0].tolist() for n in ('banana','bowl','table')}}
 save('initialization.json',restored)
 assert np.allclose(restored['robot_q'],q0,atol=1e-6,rtol=0) and np.allclose(restored['robot_qd'],qd0,atol=1e-6,rtol=0)
 assert np.allclose(restored['robot_root'],root,atol=1e-6,rtol=0)
 for n in ('banana','bowl','table'):assert np.allclose(restored['objects'][n],np.r_[initial['rigid_object/'+n+'/root_pose'][idx],initial['rigid_object/'+n+'/root_velocity'][idx]],atol=1e-6,rtol=0)
 save('protocol.json',{'scope':'Prerecorded A replay control, not new closed-loop policy success','input_sha256':{k:digest(v) for k,v in paths.items()},'steps':301,'hz':15,'physics_warmup_steps':0,'selected_reset_row':idx,'native_controller_unchanged':True,'native_gripper_unchanged':True,'early_success_termination_disabled':True,'replay_criteria':{'max_arm_joint_absolute_rad':.02,'max_raw_tool_position_m':.005,'max_raw_tool_rotation_rad':.02,'max_banana_position_m':.02,'v4_supported_release_by_step301':True},'criteria_scope':'Predeclared diagnostic resemblance limits, not bitwise identity or statistical transfer evidence.'})
 tool_id=list(robot.data.body_names).index(graph['raw_tool_body'].split('/')[-1])
 gate=ReleaseGateV4();video=VideoWriter(str(out/'three_views.mp4'),15);soft=arr(robot.data.soft_joint_pos_limits)[0]
 for i,action in enumerate(actions):
  pre=release_signals(env);gate.before(i/15,float(action[7]),pre)
  obs,_,term,trunc,_=env.step(tensor(action)[None])
  assert int(env.episode_length_buf[0])==i+1 and not bool(term.any()) and not bool(trunc.any()),'Unexpected reset/termination'
  q=arr(robot.data.joint_pos)[0];qd=arr(robot.data.joint_vel)[0];rootstate=arr(robot.data.root_state_w)[0]
  tool=rigid_inverse(pose_matrix(rootstate[:3],rootstate[3:7]))@pose_matrix(arr(robot.data.body_state_w)[0,tool_id,:3],arr(robot.data.body_state_w)[0,tool_id,3:7])
  objects=np.stack([arr(env.scene[n].data.root_state_w)[0] for n in ('banana','bowl','table')])
  assert all(np.isfinite(x).all() for x in (q,qd,rootstate,tool,objects))
  assert (q>=soft[:,0]-.03).all() and (q<=soft[:,1]+.03).all()
  post=release_signals(env);banana=env.scene['banana']
  raw=bool(rawfunc(env,**rawparams)[0]);linear=float(torch.linalg.vector_norm(banana.data.root_lin_vel_w[0]));angular=float(torch.linalg.vector_norm(banana.data.root_ang_vel_w[0]));simulation_s=(i+1)/15
  passed=gate.after(simulation_s,post,raw,linear,angular)
  metric.append({'step':i+1,'simulation_s':simulation_s,'upstream_predicate':raw,'linear_m_s':linear,'angular_rad_s':angular,'pre':pre,'post':post,'command':float(action[7]),'status':gate.status(),'passed':bool(passed)})
  refq=refstates['articulation/robot/joint_position'][i];ref_tool=chain.fk(refq[ids]);err=pose_error(ref_tool,tool)
  rows.append({'step':i+1,'action':action,'q':q,'qd':qd,'root':rootstate,'tool':tool,'objects':objects,'arm_error_rad':q[ids]-refq[ids],'tool_error':err,'banana_error_m':objects[0,:3]-refstates['rigid_object/banana/root_pose'][i,:3]})
  images=unpack_image_obs(obs);panels=[]
  for label,img in zip(('EXTERIOR','WRIST','VIEWPORT'),(images['over_shoulder_left_camera'],images['wrist_cam'],unpack_viewport_cams(obs)['combined_image'])):
   assert img.dtype==np.uint8 and img.std()>1
   frame=cv2.resize(img,(640,360));cv2.putText(frame,f'A REPLAY {i+1} {label}',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(frame)
  video.write(cv2.hconcat(panels))
  if (i+1)%50==0:print('A_REPLAY_PROGRESS',i+1,flush=True)
 complete=True
 vals={'max_arm_joint_absolute_rad':float(max(np.max(abs(r['arm_error_rad'])) for r in rows)),'max_raw_tool_position_m':float(max(np.linalg.norm(r['tool_error'][:3]) for r in rows)),'max_raw_tool_rotation_rad':float(max(np.linalg.norm(r['tool_error'][3:]) for r in rows)),'max_banana_position_m':float(max(np.linalg.norm(r['banana_error_m']) for r in rows)),'v4_supported_release_by_step301':any(r['passed'] for r in metric)}
 criteria=vals['max_arm_joint_absolute_rad']<=.02 and vals['max_raw_tool_position_m']<=.005 and vals['max_raw_tool_rotation_rad']<=.02 and vals['max_banana_position_m']<=.02 and vals['v4_supported_release_by_step301']
 save('result.json',{'complete':True,'steps':301,'replay_criteria_met':bool(criteria),'measurements':vals,'events':gate.events,'interpretation':'Independent result review required before B replay; differences remain evidence.'});print('A_REPLAY_COMPLETE',bool(criteria),flush=True)
except BaseException:
 error=traceback.format_exc();(out/'failure.txt').write_text(error);print(error,flush=True);raise
finally:
 if rows:np.savez_compressed(out/'tracking.npz',complete=np.asarray(complete),**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
 if metric:(out/'release_trace.json').write_text(json.dumps(metric,indent=2)+'\n')
 if video is not None:video.release()
 if env is not None:env.close()
 app.close()
