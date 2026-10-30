"""One unscored closed-loop P0/B temporal diagnostic. Fixed 12x pacing, no fitting or retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path.home()/'robot-transfer/bowl'
REPO = ROOT/'RoboLab'
sys.path.insert(0,str(REPO)); os.chdir(REPO)
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--case',required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
assert args.case == 'step6_B_P0_feedback_v1' and all(c.isalnum() or c=='_' for c in args.case)
output=ROOT/'runs'/args.case
output.mkdir(parents=True,exist_ok=False)
args.enable_cameras=True
app=AppLauncher(args).app
rows=[]; env=video=client=capture=trace=None; complete=False
try:
    import cv2
    import h5py
    import numpy as np
    import torch
    import zmq
    from rigid_chain import RigidChain,pose_matrix,rigid_inverse
    from local_ik import solve,physical_limits,pose_error
    from robolab.constants import set_output_dir
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.core.observations.observation_utils import unpack_image_obs,unpack_viewport_cams
    from robolab.core.utils.video_utils import VideoWriter
    from robolab.core.sensors.contact_sensor_utils import create_contact_sensors
    from robolab.core.task.instruction import primary_instruction
    from robolab.registrations.kinova.auto_env_registrations_jointpos import auto_register_kinova_envs
    from robolab.robots.kinova_gen3 import _to_torch,GRIPPER_JOINT_NAME
    from policies.gr00t.client import GR00TDroidJointposClient,DROID_EEF_ROTATION_CORRECT,_get_action_value,_as_action_chunk

    def arr(x):return _to_torch(x).detach().cpu().numpy().copy()
    def pose(x):return pose_matrix(x[:3],x[3:7])
    def save(name,value): (output/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def log_solve(step,direction,result,target,seed):
        result={**result,'target':np.asarray(target).tolist(),'seed':np.asarray(seed).tolist()}
        trace.write(json.dumps({'step':step,'direction':direction,**result})+'\n');trace.flush()
        if not result['success']:save('failed_solve.json',{'step':step,'direction':direction,**result})
        assert result['success'], (step,direction,result['reason'])

    paths={'frames':ROOT/'runs/step6_fixed_frames_cpu_v1/frame_proposal.json',
           'initial_ik':ROOT/'runs/step6_fixed_frames_cpu_v1/ik_results.json',
           'A_inventory':ROOT/'runs/step6_A_tracking_v3/inventory.json',
           'graph':ROOT/'assets/report/evidence/2026-10-21/bowl-step6/kinematics_export_v1/joint_graphs.json'}
    frames=json.loads(paths['frames'].read_text()); initial_ik=json.loads(paths['initial_ik'].read_text())
    graphs=json.loads(paths['graph'].read_text()); A=RigidChain(graphs['A']); B=RigidChain(graphs['B'])
    Alim=physical_limits(graphs['A']); Blim=physical_limits(graphs['B'])
    CA,CB=(np.asarray(frames['raw_T_canonical'][r]) for r in ('A','B'))
    Aroot=np.asarray(frames['root_world']['A']); Acam=np.asarray(frames['A_raw_T_camera_opengl'])
    initialB=np.asarray(initial_ik[0]['forward']['q']);assert initial_ik[0]['name']=='matched_A_home' and initial_ik[0]['forward']['success']
    virtual_seed=np.asarray(json.loads(paths['A_inventory'].read_text())['default_arm'])
    set_output_dir(str(output));auto_register_kinova_envs(task='BananaInBowlTask')
    names=get_envs(task='BananaInBowlTask');assert len(names)==1
    cfg=parse_env_cfg(names[0],device=args.device,seed=0,num_envs=1,use_fabric=True)
    # 301 logical actions repeated 12 times: 240.8 simulated seconds, unscored.
    # Preserve every other native termination; stop explicitly before the timeout.
    original_success=str(cfg.terminations.success)
    cfg.terminations.success=None
    cfg.episode_length_s=242
    for n in list(vars(cfg.scene)):
        if n.startswith('gripper__'):setattr(cfg.scene,n,None)
    tips={s:f'robotiq_85_{s}_finger_tip_link' for s in ('left','right')}
    cfg.contact_gripper={s:f'{{ENV_REGEX_NS}}/robot/{n}' for s,n in tips.items()}
    cfg.contact_gripper['gripper']=['left','right'];create_contact_sensors(cfg)
    cfg.scene.robot.init_state.joint_pos.update({f'joint_{i+1}':float(v) for i,v in enumerate(initialB)})
    for n in ('wrist_cam','over_shoulder_left_camera'):getattr(cfg.scene,n).update_latest_camera_pose=True
    cam=frames['B_camera_cfg'];cfg.scene.wrist_cam.prim_path=cam['prim_path']
    cfg.scene.wrist_cam.offset.pos=tuple(cam['offset']['pos']);cfg.scene.wrist_cam.offset.rot=tuple(cam['offset']['rot'])
    cfg.scene.wrist_cam.offset.convention='opengl'
    for n in ('width','height'):setattr(cfg.scene.wrist_cam,n,cam[n])
    for n in ('focal_length','horizontal_aperture','vertical_aperture'):setattr(cfg.scene.wrist_cam.spawn,n,cam[n])
    env,cfg=create_env(cfg,instruction_type='default');obs,_=env.reset()
    robot=env.scene['robot'];joints=list(robot.data.joint_names);bodies=list(robot.data.body_names)
    ids=[joints.index(n) for n in B.names];tool_id=bodies.index(graphs['B']['raw_tool_body'].split('/')[-1]);grip_id=joints.index(GRIPPER_JOINT_NAME)
    assert [joints[i] for i in env.action_manager.get_term('arm')._joint_ids]==B.names
    assert env.action_manager.total_action_dim==8 and np.isclose(env.step_dt,1/15)
    root0=pose(arr(robot.data.root_state_w)[0]);assert np.allclose(root0,frames['root_world']['B'],atol=1e-7,rtol=0)
    assert np.allclose(arr(robot.data.joint_pos)[0,ids],initialB,atol=1e-6,rtol=0)
    soft=arr(robot.data.soft_joint_pos_limits)[0];command=initialB.copy()
    _,instruction=primary_instruction(cfg.instruction)
    client=GR00TDroidJointposClient(remote_host='127.0.0.1',remote_port=15555,open_loop_horizon=1)
    client.client.socket.setsockopt(zmq.RCVTIMEO,60000);client.client.socket.setsockopt(zmq.SNDTIMEO,60000);client.client.socket.setsockopt(zmq.LINGER,0)
    video=VideoWriter(str(output/'three_views.mp4'),15)
    capture=h5py.File(output/'requests.hdf5','x');trace=(output/'ik_trace.jsonl').open('x')
    (output/'responses').mkdir()
    inventory={'scope':'Unscored unchanged-P0 fixed-adapter development attempt. Not a benchmark success estimate or learned transfer.',
        'environment':names[0],'layout_seed':0,'policy_seed':0,'instruction':instruction,'max_steps':3612,'logical_actions':301,'repeat_factor':12,'replan_every_physical_ticks':12,'replan_interval_sim_s':0.8,'initialization_steps':1,'total_simulator_steps':3613,'hz':15,'open_loop_horizon':1,
        'initial_B':initialB.tolist(),'root_initial':root0.tolist(),'arm_indices':ids,'joint_names':joints,
        'camera_remount':cam,'command_max_increment_rad':.04,'measured_joint_margin_rad':.03,
        'gripper_observation':'clip measured robotiq_85_left_knuckle_joint / 0.8 to [0,1]; A observation corruption is disabled',
        'gripper_action':'P0 >0.5 closes, otherwise opens; native B six-joint binary target',
        'termination_change':{'removed_success':original_success,'native_timeout_replaced_s':242,'policy_duration_s':240.8,'total_simulator_duration_s':3613/15},
        'action_mapping':'A predicted joint FK -> A canonical world -> B raw root -> local IK seeded from measured B; rate limit from previous command',
        'observation_mapping':'B measured canonical world -> virtual A local IK, continuous previous virtual seed; internally consistent virtual FK pose',
        'auxiliary_eef_action':'saved, never executed','setup_hashes':{str(p):digest(p) for p in paths.values()},
        'source_hashes':{n:digest(Path(__file__).with_name(n)) for n in ('run_fixed_bridge_feedback.py','local_ik.py','rigid_chain.py')},
        'client_source_hash':digest(REPO/'policies/gr00t/client.py'),'fixed_frames':frames}
    save('inventory.json',inventory)
    def state():
        q=arr(robot.data.joint_pos)[0];qd=arr(robot.data.joint_vel)[0]
        root=pose(arr(robot.data.root_state_w)[0]);tool=rigid_inverse(root)@pose(arr(robot.data.body_state_w)[0,tool_id])
        assert all(np.isfinite(v).all() for v in (q,qd,root,tool))
        re=pose_error(root0,root);fe=pose_error(B.fk(q[ids]),tool)
        assert np.linalg.norm(re[:3])<1e-4 and np.linalg.norm(re[3:])<1e-4
        assert np.linalg.norm(fe[:3])<1e-4 and np.linalg.norm(fe[3:])<1e-3
        assert (q>=soft[:,0]-.03).all() and (q<=soft[:,1]+.03).all()
        return q,qd,root,tool
    def startup_snapshot(label,current_obs):
        q,qd,root,tool=state()
        data=env.scene['wrist_cam'].data
        camera=pose_matrix(arr(data.pos_w)[0],arr(data.quat_w_opengl)[0])
        expected=root@tool@CB@rigid_inverse(CA)@Acam
        error=pose_error(expected,camera)
        record={'q':q.tolist(),'qd':qd.tolist(),'root':root.tolist(),'tool_root':tool.tolist(),
                'camera_world':camera.tolist(),'expected_camera_world':expected.tolist(),
                'camera_error':error.tolist(),'K':arr(data.intrinsic_matrices)[0].tolist(),
                'objects':{n:arr(env.scene[n].data.root_state_w)[0].tolist() for n in ('banana','bowl')},
                'episode_step':int(env.episode_length_buf[0])}
        save('startup_'+label+'.json',record)
        imgs=unpack_image_obs(current_obs)
        imgs={n:imgs[n] for n in ('wrist_cam','over_shoulder_left_camera')}
        imgs['viewport']=unpack_viewport_cams(current_obs)['combined_image']
        for n,img in imgs.items():
            assert img.dtype==np.uint8 and img.ndim==3 and img.shape[2]==3
            assert cv2.imwrite(str(output/f'startup_{label}_{n}.png'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        return record
    startup_snapshot('before',obs)
    # Exactly one declared initialization action; never add adaptive warmup/retry steps.
    warm_action=torch.as_tensor(np.r_[initialB,0.],device=env.device,dtype=torch.float32)[None]
    obs,_,term,trunc,_=env.step(warm_action)
    assert not bool(term.any()) and not bool(trunc.any()) and int(env.episode_length_buf[0])==1
    startup=startup_snapshot('after',obs)
    error=np.asarray(startup['camera_error'])
    assert np.linalg.norm(error[:3])<1e-4 and np.linalg.norm(error[3:])<1e-3, 'Camera still mismatched after one initialization step'
    assert np.allclose(startup['K'],frames['expected_camera_K'],atol=1e-4,rtol=0)
    for step in range(3612):
        logical_step=step//12
        repeat_index=step%12
        q,qd,root,tool=state()
        targetA=rigid_inverse(Aroot)@root@tool@CB@rigid_inverse(CA)
        inverse=solve(A,targetA,virtual_seed,Alim);log_solve(step,'observation_B_to_A',inverse,targetA,virtual_seed)
        virtual_seed=np.asarray(inverse['q']);virtual_tool=A.fk(virtual_seed)
        eef=np.r_[virtual_tool[:3,3],(virtual_tool[:3,:3]@DROID_EEF_ROTATION_CORRECT)[:2,:].reshape(-1)].astype(np.float32)
        images=unpack_image_obs(obs)
        closure=np.asarray([np.clip(q[grip_id]/.8,0,1)],dtype=np.float32)
        extracted={'external_image':images['over_shoulder_left_camera'],'wrist_image':images['wrist_cam'],
                   'joint_position':virtual_seed.astype(np.float32),'gripper_position':closure,'eef_9d':eef}
        packed=client._pack_request(extracted,instruction)
        camdata=env.scene['wrist_cam'].data
        actualcam=pose_matrix(arr(camdata.pos_w)[0],arr(camdata.quat_w_opengl)[0])
        ce=pose_error(root@tool@CB@rigid_inverse(CA)@Acam,actualcam)
        vce=pose_error(Aroot@virtual_tool@Acam,actualcam)
        if not (np.linalg.norm(ce[:3])<1e-4 and np.linalg.norm(ce[3:])<1e-3):
            save('failed_camera.json',{'policy_step':step,'actual':actualcam.tolist(),'physical_expected':(root@tool@CB@rigid_inverse(CA)@Acam).tolist(),'error':ce.tolist()})
        assert np.linalg.norm(ce[:3])<1e-4 and np.linalg.norm(ce[3:])<1e-3
        assert np.linalg.norm(vce[:3])<3e-4 and np.linalg.norm(vce[3:])<.002
        assert np.allclose(arr(camdata.intrinsic_matrices)[0],frames['expected_camera_K'],atol=1e-4,rtol=0)
        fresh=step%12==0
        if fresh:
            group=capture.create_group(str(step));group.attrs['instruction']=instruction
            for k,v in packed.items():
                if k.startswith(('video.','state.')):group.create_dataset(k,data=v,compression='lzf')
            capture.flush()
            response=client.client.get_action(packed)
            chunks={n:_as_action_chunk(_get_action_value(response[0],n),name=n) for n in ('joint_position','gripper_position','eef_9d')}
            np.savez_compressed(output/'responses'/f'{step:04d}.npz',**chunks)
            assert all(v.shape==(40,d) and np.isfinite(v).all() for (n,v),d in zip(chunks.items(),(7,1,9)))
            chunk=client._postprocess_chunk(client._unpack_response(response));assert chunk.shape==(40,8)
        predicted=chunk[0];Aq=predicted[:7].astype(float)
        assert (Aq>=Alim[:,0]).all() and (Aq<=Alim[:,1]).all(), 'P0 A joint target outside physical limits'
        targetB=rigid_inverse(root)@Aroot@A.fk(Aq)@CA@rigid_inverse(CB)
        forward=solve(B,targetB,q[ids],Blim);log_solve(step,'action_A_to_B',forward,targetB,q[ids])
        goal=np.asarray(forward['q']);previous=command.copy();command+=np.clip(goal-command,-.04,.04)
        assert np.max(np.abs(command-previous))<=.04000001
        assert np.isfinite(command).all() and (command>=Blim[:,0]).all() and (command<=Blim[:,1]).all()
        action=torch.as_tensor(np.r_[command,predicted[7]],device=env.device,dtype=torch.float32)[None]
        obs,_,term,trunc,_=env.step(action)
        assert int(env.episode_length_buf[0])==step+2 and not bool(term.any()) and not bool(trunc.any()), 'Unexpected termination/reset'
        postq,postqd,postroot,posttool=state()
        forces=[]
        for side,name in tips.items():
            sensor=env.scene[side+'__banana'];force=sensor.data.force_matrix_w
            assert force is not None and tuple(force.shape)==(1,1,1,3)
            assert list(sensor.body_names)==[name] and sensor.cfg.filter_prim_paths_expr==[cfg.scene.banana.prim_path]
            forces.append(arr(force)[0,0,0])
        objects=np.stack([arr(env.scene[n].data.root_state_w)[0] for n in ('banana','bowl')])
        assert np.isfinite(objects).all() and np.isfinite(forces).all()
        rows.append({'step':step+1,'simulator_step':step+2,'fresh_request':fresh,'chunk_row':0,'logical_index_0based':logical_step,'repeat_index_0based':repeat_index,'physical_time_s':(step+1)/15,'logical_elapsed_s':(step+1)/180,'pre_q':q,'pre_tool_root':tool,'q':postq,'qd':postqd,
            'root':postroot,'tool_root':posttool,'virtual_A_q':virtual_seed,'virtual_A_tool':virtual_tool,'state_eef_9d':eef,
            'gripper_observation':closure,'P0_action':predicted,'B_goal':goal,'B_action':arr(action)[0],
            'rate_limited':bool(np.max(np.abs(goal-previous))>.04),'target_B':targetB,
            'tracking_error':pose_error(targetB,posttool),'camera_mapping_error':ce,'virtual_camera_error':vce,
            'pre_camera_world':actualcam,'objects_banana_bowl':objects,'banana_force_on_tips':np.stack(forces)})
        images=unpack_image_obs(obs);panels=[]
        for label,img in zip(('EXTERIOR','WRIST','VIEWPORT'),(images['over_shoulder_left_camera'],images['wrist_cam'],unpack_viewport_cams(obs)['combined_image'])):
            assert img.dtype==np.uint8 and img.ndim==3 and img.shape[2]==3 and img.std()>1
            frame=cv2.resize(img,(640,360));cv2.putText(frame,f'B P0 FB1 SLOW12 {step+1}/3612 L{logical_step} {label}',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(frame)
        video.write(cv2.hconcat(panels))
        if (step+1)%50==0:print('B_P0_TEMPORAL_PROGRESS',step+1,flush=True)
    complete=True
    save('checks.json',{'complete':True,'steps':len(rows),'requests':len(capture),'initialization_steps':1,'total_simulator_steps':3613,'task_score':None,
                        'scope':'Integration diagnostic; controlled-release scoring and full adapter approval remain pending.'})
    print('B_P0_TEMPORAL_DIAGNOSTIC_COMPLETE',len(rows),flush=True)
except BaseException:
    error=traceback.format_exc();(output/'failure.txt').write_text(error);print(error,flush=True)
    raise
finally:
    if rows:np.savez_compressed(output/'tracking.npz',complete=np.asarray(complete),**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
    if trace is not None:trace.close()
    if capture is not None:capture.close()
    if video is not None:video.release()
    if client is not None:client.close()
    if env is not None:env.close()
    app.close()
